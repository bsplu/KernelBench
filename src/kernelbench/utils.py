########################
# Utils Functions
########################

import multiprocessing
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env early so query_server can see them
import subprocess
import re
import random
import tempfile
from pathlib import Path
import re
import math
import os
import json
from tqdm import tqdm
from importlib.resources import files, as_file

# API clients
from openai import OpenAI
from litellm import completion

import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import time
import concurrent
from functools import cache


from concurrent.futures import ProcessPoolExecutor, as_completed

SGLANG_KEY = os.environ.get("SGLANG_API_KEY")
SGLANG_ADDRESS = (
    os.environ.get("SGLANG_API_ADDRESS")
    or "localhost:30000"
)


# def _maybe_debug_cot_payload(response, server_type: str, model_name: str):
#     """
#     Temporary debug hook: dump raw response payload to inspect potential CoT fields.
#     Enable with env: KB_DEBUG_COT=1
#     """
#     flag = os.environ.get("KB_DEBUG_COT", "0").lower() in {"1", "true", "yes"}
#     if not flag:
#         return
#     try:
#         dump = None
#         if hasattr(response, "model_dump"):
#             dump = response.model_dump()
#         elif hasattr(response, "dict"):
#             dump = response.dict()
#         else:
#             dump = {"repr": repr(response)}

#         os.makedirs("results/eval_logs", exist_ok=True)
#         path = os.path.join("results/eval_logs", f"cot_debug_{os.getpid()}.log")
#         with open(path, "a", encoding="utf-8") as f:
#             f.write("\n" + "=" * 80 + "\n")
#             f.write(f"server_type={server_type} model_name={model_name}\n")
#             # quick signal checks for common reasoning/coT fields
#             text = json.dumps(dump, ensure_ascii=False, default=str)
#             markers = [
#                 "reasoning_content",
#                 "reasoning",
#                 "reasoning_tokens",
#                 "thought",
#                 "thinking",
#             ]
#             found = [m for m in markers if m in text]
#             f.write(f"cot_markers_found={found}\n")
#             f.write(text + "\n")
#     except Exception as e:
#         print(f"[KB_DEBUG_COT] failed to dump payload: {e}")

# def _extract_minimax_think(content: str) -> tuple[str | None, str]:
#     """
#     Extract Minimax-style CoT embedded in content:
#     <think> ... </think>
#     Returns (reasoning_text_or_none, content_without_think).
#     """
#     if not isinstance(content, str) or not content:
#         return None, content
#     m = re.search(r"<think>([\s\S]*?)</think>", content, re.IGNORECASE)
#     if not m:
#         return None, content
#     reasoning = m.group(1).strip()
#     final_content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
#     return reasoning if reasoning else None, final_content


# def _maybe_debug_minimax_cot(content: str, model_name: str, label: str = "single"):
#     """
#     Temporary debug helper for Minimax models whose CoT is embedded in `content`.
#     Enable with KB_DEBUG_COT=1.
#     """
#     flag = os.environ.get("KB_DEBUG_COT", "0").lower() in {"1", "true", "yes"}
#     if not flag:
#         return
#     if "minimax" not in (model_name or "").lower():
#         return
#     reasoning, final_content = _extract_minimax_think(content)
#     if reasoning is None:
#         return
#     try:
#         os.makedirs("results/eval_logs", exist_ok=True)
#         path = os.path.join("results/eval_logs", f"cot_debug_{os.getpid()}.log")
#         with open(path, "a", encoding="utf-8") as f:
#             f.write("\n" + "-" * 80 + "\n")
#             f.write(f"minimax_think_detected[{label}]=True\n")
#             f.write(f"minimax_think_len={len(reasoning)}\n")
#             f.write("minimax_think_excerpt:\n")
#             f.write(reasoning[:2000] + "\n")
#             f.write("minimax_final_content_excerpt:\n")
#             f.write((final_content or "")[:1000] + "\n")
#     except Exception as e:
#         print(f"[KB_DEBUG_COT] failed to dump minimax think content: {e}")

def _strip_think_block_from_content(content: str) -> str:
    """
    Remove <think>...</think> blocks from assistant content before returning to caller.
    """
    if not isinstance(content, str) or "<think>" not in content.lower():
        return content
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
    return cleaned


########################################################
# Inference Helpers
########################################################

def set_gpu_arch(arch_list: list[str] | str):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """
    # pydra CLI may pass scalar string (e.g., gpu_arch=Ampere); normalize to list.
    if isinstance(arch_list, str):
        # Support both single value ("Ampere") and comma-separated ("Ampere,Ada").
        arch_list = [a.strip() for a in arch_list.split(",") if a.strip()]

    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")
    
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

def query_server(
    prompt: str | list[dict],  # string if normal prompt, list of dicts if chat prompt,
    system_prompt: str = "You are a helpful assistant",  # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0, # nucleus sampling
    top_k: int = 50, 
    max_tokens: int = 128,  # max output tokens to generate
    num_completions: int = 1,
    server_address: str = SGLANG_ADDRESS,
    server_type: str = "sglang",
    model_name: str = "default",  # specify model type

    # for reasoning models
    is_reasoning_model: bool = False, # indiactor of using reasoning models
    budget_tokens: int = 0, # for claude thinking
    reasoning_effort: str = None, # only for o1 and o3 / more reasoning models in the future
):
    """
    Query various sort of LLM inference API providers
    Done through liteLLM:
    - Local Server (SGLang, vLLM, Tokasaurus)
    """
    # Accept address as:
    # - host:port (e.g. localhost:30000)
    # - http(s)://host:port
    # - http(s)://host:port/v1
    def _to_openai_base_url(addr: str) -> str:
        addr = (addr or "").strip()
        if not addr:
            addr = "localhost:30000"
        if not addr.startswith(("http://", "https://")):
            addr = f"http://{addr}"
        addr = addr.rstrip("/")
        if addr.endswith("/v1"):
            return addr
        return f"{addr}/v1"

    # Local Server (SGLang, vLLM, Tokasaurus) - special handling
    if server_type == "local":
        base_url = _to_openai_base_url(server_address)
        client = OpenAI(
            api_key=SGLANG_KEY, base_url=base_url, timeout=None, max_retries=0
        )
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        create_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "n": num_completions,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if is_reasoning_model:
            model_name_l = (model_name or "").lower()
            # DeepSeek-compatible thinking flags (matches curl payload style).
            if "deepseek" in model_name_l:
                create_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
                    "enable_thinking": True,
                }

            elif reasoning_effort:
                # For OpenAI-style reasoning models where this field is accepted.
                create_kwargs["reasoning_effort"] = reasoning_effort

        response = client.chat.completions.create(**create_kwargs)
        # _maybe_debug_cot_payload(response, server_type=server_type, model_name=model_name)
        outputs = [choice.message.content for choice in response.choices]
        # for i, out in enumerate(outputs):
        #     _maybe_debug_minimax_cot(out, model_name=model_name, label=f"local_choice_{i}")
        outputs = [_strip_think_block_from_content(out) for out in outputs]
        
        # output processing
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    
    # All other providers - use LiteLLM unified interface
    # Build messages list with system prompt first (if not already present)
    messages = []
    
    # Check if prompt is already a list with a system message
    if isinstance(prompt, list) and prompt and prompt[0].get("role") == "system":
        # Prompt already has system message, use it directly
        messages = prompt
    else:
        # Add system prompt first if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Then add the actual prompt
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)
    
    try:
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "n": num_completions,
        }
        
        # Reasoning models (o1, o3, etc.) don't support standard sampling params
        if is_reasoning_model:
            # Note: o1/o3 models don't support temperature, top_p, top_k
            # LiteLLM will pass through reasoning_effort for OpenAI o1/o3 models
            if reasoning_effort:
                completion_kwargs["reasoning_effort"] = reasoning_effort
            # Claude extended thinking uses "thinking" parameter with dict structure
            # Format: {"type": "enabled", "budget_tokens": <int>}
            if budget_tokens > 0 and "anthropic" in model_name.lower():
                completion_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            elif "deepseek" in model_name.lower():
                completion_kwargs["chat_template_kwargs"] = {"enable_thinking": True, 'thinking': True}
                completion_kwargs["enable_thinking"] = True 
                completion_kwargs["temperature"] = temperature      
                completion_kwargs["top_p"] = top_p   
                completion_kwargs["top_k"] = top_k      
        else:
            # Standard models support temperature and top_p
            completion_kwargs["temperature"] = temperature
            completion_kwargs["top_p"] = top_p
            
            # top_k is not supported by OpenAI models
            if "openai/" not in model_name.lower() and "gpt" not in model_name.lower():
                completion_kwargs["top_k"] = top_k
        
        response = completion(**completion_kwargs)
        # _maybe_debug_cot_payload(response, server_type=server_type, model_name=model_name)
        
        # output processing
        if num_completions == 1:
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(f"LLM returned None content for model {model_name}. finish_reason: {response.choices[0].finish_reason}")
            # _maybe_debug_minimax_cot(content, model_name=model_name, label="litellm_single")
            return _strip_think_block_from_content(content)
        else:
            contents = [choice.message.content for choice in response.choices]
            if any(c is None for c in contents):
                raise ValueError(f"LLM returned None content in one or more completions for model {model_name}")
            # for i, content in enumerate(contents):
            #     _maybe_debug_minimax_cot(content, model_name=model_name, label=f"litellm_choice_{i}")
            return [_strip_think_block_from_content(content) for content in contents]
    except Exception as e:
        print(f"Error in query_server for model {model_name}: {e}")
        raise


# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 1.6, 
        "model_name": "deepseek/deepseek-coder",
        "max_tokens": 4096
    },
    "google": {
        "model_name": "gemini/gemini-2.5-flash",
        "temperature": 0.7, # need to experiment with temperature
        "max_tokens": 16384,
    },
    "together": { # mostly for Llama 3.1
        "model_name": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "local": {  # this is for running locally (SGLang, vLLM, Tokasaurus), mostly for Llama
        "temperature": 0.8, # human eval pass@N temperature
        "server_address": SGLANG_ADDRESS,
        "max_tokens": 8192,
    },
    "anthropic": {  # for Claude 3.7 Sonnet
        "model_name": "anthropic/claude-3-7-sonnet-20250219",
        "temperature": 0.8,
        "max_tokens": 8192,
    },
    "openai": {
        "model_name": "gpt-4o-2024-08-06",
        # "model_name": "o1-preview-2024-09-12", # be careful with this one
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "fireworks": {
        "model_name": "fireworks_ai/llama-v3p1-70b-instruct",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
}


def create_inference_server_from_presets(server_type: str = None, 
                                         greedy_sample: bool = False,   
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         model_name: str = None,
                                         **kwargs,
                                         ) -> callable:
    """
    Return a callable function that queries LLM with given settings
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[server_type].copy()
        
        if model_name is not None and model_name != "None":
            server_args["model_name"] = model_name
        
        if kwargs:
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != "None"}
            if "server_port" in filtered_kwargs:
                print("Warning: 'server_port' is deprecated and ignored. Use 'server_address' with host:port.")
                filtered_kwargs.pop("server_port")
            server_args.update(filtered_kwargs)
        
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        
        if verbose:
            print(f"Querying server {server_type} with model {server_args['model_name']} and args: {server_args}")
        
        if time_generation:
            start_time = time.time()
            response = query_server(
                prompt, server_type=server_type, **server_args
            )
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return query_server(
                prompt, server_type=server_type, **server_args
            )
    
    return _query_llm

"""
Model output processing
#  TODO: add unit tests
"""


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


########################################################
# Path Resolution Helpers
########################################################
REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def get_package_resource_path(relative_path: str) -> str:
    """
    Get absolute path to a kernelbench package resource.
    Works for all three usage modes:
      - Running from repo directly
      - As a git submodule
      - As an installed pip/uv dependency
    
    Args:
        relative_path: Path relative to kernelbench/, e.g. "prompts/prompts.toml"
    """
    # Try importlib.resources first (installed package)
    try:
        resource = files("kernelbench").joinpath(relative_path)
        with as_file(resource) as path:
            if path.exists():
                return str(path)
    except (TypeError, FileNotFoundError):
        pass
    
    # Try repo path (running from source / submodule)
    repo_path = os.path.join(REPO_TOP_PATH, "src/kernelbench", relative_path)
    if os.path.exists(repo_path):
        return repo_path
        
    raise FileNotFoundError(f"Could not find resource: {relative_path}")


def resolve_path(rel: str) -> str:
    """
    Resolve a relative path to absolute. Handles paths like "src/kernelbench/prompts/..."
    from prompts.toml which reference files relative to repo root.
    """
    if os.path.isabs(rel):
        return rel
    
    # Convert "src/kernelbench/..." paths to package-relative
    if rel.startswith("src/kernelbench/"):
        return get_package_resource_path(rel[len("src/kernelbench/"):])
    
    # Otherwise treat as repo-relative
    repo_path = os.path.join(REPO_TOP_PATH, rel)
    if os.path.exists(repo_path):
        return repo_path
    
    raise FileNotFoundError(f"Could not resolve path: {rel}")


def print_messages(messages):
    for message in messages:
        print(message["role"])
        print(message["content"])
        print("-" * 50)
        print("\n\n")


def extract_python_code(text):
    """
    Extract python code from model output
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""


def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    if output_string is None:
        return None
    
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None

def extract_code_blocks(text, code_language_types: list[str]) -> str:
    '''
    Extract all code blocks from text, combine them to return as a single string
    '''
    pattern = r'```.*?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type):].strip()
        combined_code.append(code)
    
    return " \n ".join(combined_code) if combined_code else ""

################################################################################
# Scale up experiments in parallel
################################################################################

def maybe_multithread(func, instances, num_workers, time_interval=0.0, *shared_args, **shared_kwargs):
    """
    Multithreaded execution of func, with optional time interval between queries
    Ideal for querying LLM APIs, does not provide process isolation
    """
    output_data = []
    if num_workers not in [1, None]:
        with tqdm(total=len(instances), smoothing=0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

                # Submit tasks one at a time with delay between them
                futures = []
                for instance in instances:
                    futures.append(
                        executor.submit(
                            func,
                            instance,
                            *shared_args,
                            **shared_kwargs
                        )
                    )
                    time.sleep(time_interval)  # sleep between submitting each task



                # Wait for each future to complete
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            output_data.append(result)
                    except Exception as e:
                        print("Got an error!", e)
                        continue
    else:
        for instance in tqdm(instances):
            output = func(instance, *shared_args, **shared_kwargs)
            if output is not None: output_data.append(output)

    return output_data


def maybe_multiprocess_cuda(
    func, instances, num_workers, *shared_args, **shared_kwargs
):
    """
    From monkeys, but modified to work with CUDA
    """
    output_data = []
    multiprocessing.set_start_method(
        "spawn", force=True
    )  # this is necessary for CUDA to work

    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(func, instance, *shared_args, **shared_kwargs): None
                for instance in instances
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                    if result is not None:
                        output_data.append(result)
                except Exception as e:
                    print("Got an error!", e)
                    continue
    return output_data

# src/random_inputs.py
import os, torch, itertools
from torch.distributions import Normal, Uniform, Laplace, Exponential, LogNormal

# Pick which distributions are allowed in “random” mode.
_DEFAULT_RANDOM_POOL = (
    ("normal",      lambda shape: Normal(0, 1).sample(shape)),
    ("uniform",     lambda shape: Uniform(-1, 1).sample(shape)),
    ("laplace",     lambda shape: Laplace(0, 1).sample(shape)),
    ("exponential", lambda shape: Exponential(1).sample(shape)),   # strictly >0
    ("lognormal",   lambda shape: LogNormal(0, 1).sample(shape)),  # strictly >0
)


def sample(shape, mode="random"):
    """
    shape : torch.Size or tuple
    mode  : "random"  – draw from a rotating pool of distributions
            "target"  – return a tensor from a randomly chosen edge-case pattern
            <dist>    – force a single distribution name, e.g. "laplace"
    """
    if mode == "random":
        # Round-robin through default pool
        idx = int(torch.empty((), dtype=torch.int64).random_()) % len(_DEFAULT_RANDOM_POOL)
        _, fn = _DEFAULT_RANDOM_POOL[idx]
        return fn(shape)

    # Explicit distribution name
    pool = dict(_DEFAULT_RANDOM_POOL)
    if mode not in pool:
        raise ValueError(f"Unknown distribution {mode}")
    return pool[mode](shape)


# ------------------------------------------------------------------
# Public helper: rand_mix / rand_mix_like
# ------------------------------------------------------------------

def rand_mix(*size, dist: str = "random", device=None, dtype=None, requires_grad: bool = False):
    """Return a tensor drawn from a chosen distribution (or randomly chosen).

    Parameters
    ----------
    *size : int or tuple
        Dimensions of the output tensor (same semantics as ``torch.randn``).
    dist : str, optional
        • "random"   – randomly cycle through the default pool defined above.
        • "target"   – pick from the specialised _TARGETED_CASES pool.
        • any key in the default pool ("normal", "uniform", "laplace", ...).
    device, dtype, requires_grad : any
        Forwarded to ``Tensor.to`` / ``Tensor.requires_grad_`` for convenience.
    """
    # normalise *size → shape tuple
    shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, torch.Size)) else size

    t = sample(shape, mode=dist)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    if requires_grad:
        t.requires_grad_(True)
    return t

def rand_mix_like(tensor: torch.Tensor, dist: str = "random", **kwargs):
    """rand_mix variant that infers shape from *tensor*."""
    return rand_mix(*tensor.shape, dist=dist, **kwargs)

# Register convenience aliases under torch namespace (does not shadow existing fns)
setattr(torch, "rand_mix", rand_mix)
setattr(torch, "rand_mix_like", rand_mix_like)
