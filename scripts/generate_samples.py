import json
import os, sys
from dataclasses import dataclass

import pydra
import torch
import tomli

from pydra import Config, REQUIRED

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref
from kernelbench.prompt_constructor_toml import get_prompt_for_backend, get_custom_prompt
from kernelbench.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    extract_last_code,
    get_package_resource_path,
    maybe_multithread,
    set_gpu_arch,
)
from kernelbench.kernel_static_checker import validate_kernel_static

"""
Batch Generate Samples for Particular Level

Assume 1 sample per problem here
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

SELF_REVIEW_PROMPT_TEMPLATE = None
SELF_REVIEW_FEEDBACK_TEMPLATE = None
SELF_REVIEW_REWRITE_TEMPLATE = None
_prompts_toml = get_package_resource_path("prompts/prompts.toml")
with open(_prompts_toml, "rb") as _f:
    _prompts_data = tomli.load(_f)
SELF_REVIEW_PROMPT_TEMPLATE = (  # backward-compatible alias
    _prompts_data.get("templates", {})
    .get("common", {})
    .get("self_review_template")
)
SELF_REVIEW_FEEDBACK_TEMPLATE = (
    _prompts_data.get("templates", {})
    .get("common", {})
    .get("self_review_feedback_template")
)
SELF_REVIEW_REWRITE_TEMPLATE = (
    _prompts_data.get("templates", {})
    .get("common", {})
    .get("self_review_rewrite_template")
)


class GenerationConfig(Config):
    def __init__(self):

        self.dataset_src = REQUIRED  # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED

        # subset of problems to generate, otherwise generate on all problems in the level
        self.subset = (
            None,
            None,
        )  # (start_id, end_id), both inclusive - logical 1-indexed IDs

        self.run_name = REQUIRED  # name of the run

        # num of thread pool to call inference server in parallel
        self.num_workers = 64
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = None
        self.model_name = None
        self.max_tokens = None
        self.temperature = 0.0
        
        # Reasoning model specific parameters
        self.is_reasoning_model = False  # set to True for o1, o3, Gemini 2.5 thinking, etc.
        self.reasoning_effort = "low"  # for o1/o3: "low", "medium", "high"
        self.budget_tokens = 0  # for Claude extended thinking mode

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        self.verbose = False
        self.store_type = "local"  # TODO: add Database Integration

        # Number of samples to generate per problem for pass@k analysis
        self.num_samples = 1  # Default to 1 sample per problem
        self.generation_retries = 3  # retry model generation when code extraction fails
        self.enable_self_review = True  # whether to query model self-review feedback before rewrite
        self.self_review_passes = 1  # deprecated: no longer used in generation loop

        self.log_prompt = False
        self.log_model_output = False  # save raw model outputs for debugging

        self.backend = "cuda"
        
        self.precision = "fp32"
        self.prompt_option = "one_shot"  # zero_shot, one_shot, few_shot
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.custom_prompt_key = None

        self.check_kernel = True  # [experimental] optional static checker catching potential hacking patterns

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int  # logically indexed
    sample_id: int


def _extract_kernel_candidate(generation) -> str | None:
    """Extract code from model output with tolerant fallbacks."""
    if generation is None:
        return None

    if not isinstance(generation, str):
        generation = str(generation)

    custom_kernel = extract_first_code(generation, ["python", "cpp"])
    if custom_kernel is None:
        custom_kernel = extract_last_code(generation, ["python", "cpp"])

    if custom_kernel is None and isinstance(generation, str):
        candidate = generation.strip()
        if any(
            marker in candidate
            for marker in ["class ModelNew", "load_inline", "__global__", "torch.utils.cpp_extension"]
        ):
            custom_kernel = candidate

    return custom_kernel


def _build_self_review_feedback_prompt(
    ref_arch_src: str,
    original_task_prompt: str,
    candidate_code: str,
    backend: str,
    precision: str,
) -> str:
    """Prompt for analysis-only self-review feedback (no code output)."""
    if not SELF_REVIEW_FEEDBACK_TEMPLATE:
        raise ValueError(
            "Missing templates.common.self_review_feedback_template in prompts.toml; "
            "self-review feedback prompt must be defined in src/kernelbench/prompts/prompts.toml."
        )
    return SELF_REVIEW_FEEDBACK_TEMPLATE.format(
        backend_display=backend.upper(),
        precision_display=precision.upper(),
        ref_arch_src=ref_arch_src,
        original_task_prompt=original_task_prompt,
        candidate_code=candidate_code,
    ).strip()

def _build_self_review_rewrite_prompt(
    ref_arch_src: str,
    original_task_prompt: str,
    candidate_code: str,
    review_feedback: str,
    backend: str,
    precision: str,
) -> str:
    """Prompt for rewrite pass conditioned on review feedback."""
    template = SELF_REVIEW_REWRITE_TEMPLATE or SELF_REVIEW_PROMPT_TEMPLATE
    if not template:
        raise ValueError(
            "Missing templates.common.self_review_rewrite_template (or self_review_template) in prompts.toml."
        )
    return template.format(
        backend_display=backend.upper(),
        precision_display=precision.upper(),
        ref_arch_src=ref_arch_src,
        original_task_prompt=original_task_prompt,
        review_feedback=review_feedback,
        candidate_code=candidate_code,
    ).strip()


def _append_model_output_log(
    run_dir: str,
    level: int,
    problem_id: int,
    sample_id: int,
    stage: str,
    content,
):
    """
    Append raw model output to per-sample log file for debugging.
    """
    log_path = os.path.join(
        run_dir,
        f"level_{level}_problem_{problem_id}_sample_{sample_id}_model_output.log",
    )
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2, default=str)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {stage} =====\n")
        f.write(text)
        f.write("\n")


def _format_static_check_feedback(
    errors,
    warnings,
    attempt: int,
    max_retries: int,
) -> str:
    """
    Build concise, actionable feedback text for model rewrite after static-check failures.
    """
    errors = errors or []
    warnings = warnings or []
    error_lines = "\n".join([f"- {msg}" for msg in errors]) if errors else "- (none)"
    warning_lines = "\n".join([f"- {msg}" for msg in warnings]) if warnings else "- (none)"
    return (
        f"Static check failed at generation attempt {attempt}/{max_retries}.\n\n"
        f"Blocking errors:\n{error_lines}\n\n"
        f"Warnings:\n{warning_lines}\n\n"
        "Please rewrite the candidate code to resolve all blocking errors first while preserving "
        "the reference Model interface and semantics."
    )


def generate_sample_single(
    work: WorkArgs,
    config: GenerationConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
) -> bool:
    # 1. Fetch Problem - unified interface
    problem = dataset.get_problem_by_id(work.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name

    if config.custom_prompt_key:
        custom_prompt = get_custom_prompt(
            config.custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    else:
        custom_prompt = get_prompt_for_backend(
            ref_arch_src,
            config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    if config.log_prompt:
        prompt_path = os.path.join(
            run_dir,
            f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt",
        )
        with open(prompt_path, "w") as f:
            f.write(custom_prompt)

    # Query server with constructed prompt (with retries for extraction/static-check failures)
    max_retries = getattr(config, "generation_retries", 3)
    if isinstance(max_retries, str):
        try:
            max_retries = int(max_retries)
        except ValueError:
            max_retries = 3
    max_retries = max(1, max_retries)

    log_model_output = getattr(config, "log_model_output", False)
    if isinstance(log_model_output, str):
        log_model_output = log_model_output.lower() in ["true", "1", "yes"]

    enable_self_review = getattr(config, "enable_self_review", True)
    if isinstance(enable_self_review, str):
        enable_self_review = enable_self_review.lower() in ["true", "1", "yes"]

    custom_kernel = None
    last_generation = None
    current_prompt = custom_prompt
    last_static_check_status = True
    last_static_errors = []
    last_static_warnings = []

    for attempt in range(1, max_retries + 1):
        generation = inference_server(current_prompt)
        last_generation = generation
        if log_model_output:
            _append_model_output_log(
                run_dir=run_dir,
                level=config.level,
                problem_id=work.problem_id,
                sample_id=work.sample_id,
                stage=f"generation_attempt_{attempt}",
                content=generation,
            )

        candidate_kernel = _extract_kernel_candidate(generation)
        if candidate_kernel is None:
            if config.verbose:
                print(
                    f"Code extraction failed for problem {work.problem_id} sample {work.sample_id} "
                    f"(attempt {attempt}/{max_retries}), retrying..."
                )
            custom_kernel = None
            continue

        static_check_status = True
        static_errors = []
        static_warnings = []
        if config.check_kernel:
            static_check_status, static_errors, static_warnings = validate_kernel_static(
                candidate_kernel,
                backend=config.backend,
                precision=config.precision,
            )
            if static_warnings:
                print(
                    f"Static check warnings for sample {work.sample_id} for problem {work.problem_id}: "
                    f"{problem_name}. Warnings: {static_warnings}"
                )

        static_feedback = ""
        if config.check_kernel and not static_check_status:
            static_feedback = _format_static_check_feedback(
                errors=static_errors,
                warnings=static_warnings,
                attempt=attempt,
                max_retries=max_retries,
            )
            if log_model_output:
                _append_model_output_log(
                    run_dir=run_dir,
                    level=config.level,
                    problem_id=work.problem_id,
                    sample_id=work.sample_id,
                    stage=f"static_check_failure_attempt_{attempt}",
                    content=static_feedback,
                )

        review_feedback_bundle = static_feedback
        if enable_self_review:
            feedback_prompt = _build_self_review_feedback_prompt(
                ref_arch_src=ref_arch_src,
                original_task_prompt=custom_prompt,
                candidate_code=candidate_kernel,
                backend=config.backend,
                precision=config.precision,
            )
            if static_feedback:
                feedback_prompt = (
                    f"{feedback_prompt}\n\n"
                    f"Static check failure details to incorporate:\n```text\n{static_feedback}\n```"
                )
            review_feedback = inference_server(feedback_prompt)
            review_feedback_text = str(review_feedback).strip()
            if log_model_output:
                _append_model_output_log(
                    run_dir=run_dir,
                    level=config.level,
                    problem_id=work.problem_id,
                    sample_id=work.sample_id,
                    stage=f"review_feedback_attempt_{attempt}",
                    content=review_feedback,
                )
            review_feedback_bundle = (
                f"{static_feedback}\n\nModel review feedback:\n{review_feedback_text}"
                if static_feedback else review_feedback_text
            )

        rewrite_prompt = _build_self_review_rewrite_prompt(
            ref_arch_src=ref_arch_src,
            original_task_prompt=custom_prompt,
            candidate_code=candidate_kernel,
            review_feedback=review_feedback_bundle,
            backend=config.backend,
            precision=config.precision,
        )
        rewritten_generation = inference_server(rewrite_prompt)
        if log_model_output:
            _append_model_output_log(
                run_dir=run_dir,
                level=config.level,
                problem_id=work.problem_id,
                sample_id=work.sample_id,
                stage=f"review_rewrite_attempt_{attempt}",
                content=rewritten_generation,
            )

        rewritten_kernel = _extract_kernel_candidate(rewritten_generation)
        custom_kernel = rewritten_kernel if rewritten_kernel is not None else candidate_kernel
        current_prompt = rewrite_prompt

        if config.check_kernel:
            last_static_check_status, last_static_errors, last_static_warnings = validate_kernel_static(
                custom_kernel,
                backend=config.backend,
                precision=config.precision,
            )
        else:
            last_static_check_status, last_static_errors, last_static_warnings = True, [], []

    # check LLM is able to generate custom backend code
    assert custom_kernel is not None, (
        f"Custom {config.backend} code generation failed after {max_retries} attempts; "
        f"last output type={type(last_generation).__name__}"
    )

    if config.check_kernel:
        assert last_static_check_status, (
            f"Static check failed for sample {work.sample_id} for problem {work.problem_id}: "
            f"{problem_name}. Error: {last_static_errors}. Warnings: {last_static_warnings}"
        )

    if config.verbose:
        print(
            f"Generated sample {work.sample_id} for problem {work.problem_id}: {problem_name}"
        )

    # Store to local file
    kernel_path = os.path.join(
        run_dir,
        f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py",
    )
    with open(kernel_path, "w") as f:
        f.write(custom_kernel)

    return True


def generate_sample_launcher(
    work: WorkArgs,
    config: GenerationConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
):
    try:
        return generate_sample_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating sample {work.problem_id} {work.sample_id}: {e}")
        return None


def check_kernel_exists(
    run_dir: str, level: int, problem_id: int, sample_id: int
) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )
    return os.path.exists(kernel_path)


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    from kernelbench.utils import SERVER_PRESETS
    
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", "None")
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", "None")
    
    # Convert string boolean to actual boolean for reasoning model flag
    if isinstance(config.is_reasoning_model, str):
        config.is_reasoning_model = config.is_reasoning_model.lower() in ['true', '1', 'yes']
    
    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        if trimmed.lower() in {"", "none"}:
            custom_prompt_key = None
        else:
            custom_prompt_key = trimmed
    config.custom_prompt_key = custom_prompt_key

    include_hardware = config.include_hardware_info
    if isinstance(include_hardware, str):
        include_hardware = include_hardware.lower() in ["true", "1", "yes"]
    config.include_hardware_info = include_hardware
    if isinstance(config.log_prompt, str):
        config.log_prompt = config.log_prompt.lower() in ["true", "1", "yes"]
    if isinstance(config.log_model_output, str):
        config.log_model_output = config.log_model_output.lower() in ["true", "1", "yes"]

    supported_backends = {"cuda", "triton", "cute", "tilelang", "thunderkittens"}
    backend = config.backend.lower()
    if backend not in supported_backends:
        raise ValueError(
            f"Unsupported backend: {config.backend}. Must be one of {sorted(supported_backends)}."
        )
    config.backend = backend
    if backend == "tilelang":
        config.precision = "fp16"
    if backend == "thunderkittens":
        config.precision = "bf16"

    config.prompt_option = str(config.prompt_option).lower()
    valid_prompt_options = {"zero_shot", "one_shot", "few_shot"}
    if not config.custom_prompt_key:
        if config.prompt_option not in valid_prompt_options:
            raise ValueError(
                f"Invalid prompt_option '{config.prompt_option}'. Must be one of {sorted(valid_prompt_options)}."
            )
        if include_hardware and not config.hardware_gpu_name:
            raise ValueError(
                "include_hardware_info is True but hardware_gpu_name is not provided."
            )

    print(f"Starting Batch Generation with config: {config}")

    # Dataset Configurations - Unified loading
    dataset = construct_kernelbench_dataset(
        level=config.level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )

    all_problem_ids = dataset.get_problem_ids()

    if config.subset == (None, None):
        problem_ids_to_run = all_problem_ids
    else:
        start, end = config.subset
        problem_ids_to_run = [pid for pid in all_problem_ids if start <= pid <= end]
        if not problem_ids_to_run:
            print(f"Warning: No problems found in subset range {config.subset}")

    print(
        f"Generating {config.num_samples} sample(s) each for level {config.level} problems: {problem_ids_to_run}"
    )

    # set up run directory
    run_dir = os.path.join(config.runs_dir, config.run_name)
    run_exists = os.path.exists(run_dir)
    if run_exists:
        print(f"\n⚠️  WARNING: Run directory already exists: {run_dir}")
        print(f"   Existing kernels will be skipped. Use a different run_name for a fresh run.\n")
    os.makedirs(run_dir, exist_ok=True)
    pydra.save_yaml(config.to_dict(), os.path.join(run_dir, "generation_config.yaml"))

    assert (
        config.store_type == "local"
    ), "supporting local file-system based storage for now"  # database integreation coming soon, need to migrate from CUDA Monkeys code

    problems_to_run = []
    total_problems = 0
    already_completed = 0
    for problem_id in problem_ids_to_run:
        for sample_id in range(config.num_samples):
            total_problems += 1
            if not check_kernel_exists(run_dir, config.level, problem_id, sample_id):
                problems_to_run.append(
                    WorkArgs(problem_id=int(problem_id), sample_id=sample_id)
                )
            else:
                already_completed += 1
    
    if already_completed > 0:
        print(f"📁 Found {already_completed}/{total_problems} kernels already generated. Generating remaining {len(problems_to_run)} kernels.")

    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        is_reasoning_model=config.is_reasoning_model,
        reasoning_effort=config.reasoning_effort,
        budget_tokens=config.budget_tokens,
    )

    # Launch workers
    generation_results = maybe_multithread(
        generate_sample_launcher,
        problems_to_run,
        config.num_workers,
        time_interval=config.api_query_interval,
        # extra args
        config=config,
        dataset=dataset,
        inference_server=inference_server,
        run_dir=run_dir,
    )

    num_generated_samples = len(generation_results)
    num_attempted = len(problems_to_run)
    num_failed_problems = num_attempted - num_generated_samples
    
    if num_attempted == 0:
        print(f"\n✅ All {total_problems} kernels already exist in {run_dir}")
        print(f"   Use a different run_name if you want to generate fresh samples.\n")
    else:
        print(
            f"\nGenerated {num_generated_samples} samples for total {num_attempted} problems, Please retry for the {num_failed_problems} failed problems."
        )


if __name__ == "__main__":
    main()
