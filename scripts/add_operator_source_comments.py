#!/usr/bin/env python3
import re
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVEL_DIRS = [ROOT / "KernelBench" / "level1", ROOT / "KernelBench" / "level2", ROOT / "KernelBench" / "level3"]
CACHE_DIR = ROOT / ".cache" / "pytorch_source_snippets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEGIN = "# === PyTorch Internal Source Excerpts (auto-generated) ==="
END = "# === End PyTorch Internal Source Excerpts ==="

NN_RE = re.compile(r"nn\.[A-Za-z_][A-Za-z0-9_]*")
TORCH_RE = re.compile(r"torch\.[A-Za-z_][A-Za-z0-9_]*")
F_RE = re.compile(r"torch\.nn\.functional\.[A-Za-z_][A-Za-z0-9_]*")

# For each op, provide a likely upstream C++ source file and a regex to focus nearby lines.
OP_SOURCE_MAP = {
    "torch.matmul": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/LinearAlgebra.cpp",
        [r"matmul", r"mm", r"bmm"],
    ),
    "torch.mm": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/LinearAlgebra.cpp",
        [r"\bmm\b", r"addmm"],
    ),
    "torch.bmm": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/LinearAlgebra.cpp",
        [r"\bbmm\b"],
    ),
    "torch.addmm": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/LinearAlgebra.cpp",
        [r"addmm"],
    ),
    "nn.Conv1d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Convolution.cpp",
        [r"conv1d", r"convolution"],
    ),
    "nn.Conv2d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Convolution.cpp",
        [r"conv2d", r"convolution"],
    ),
    "nn.Conv3d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Convolution.cpp",
        [r"conv3d", r"convolution"],
    ),
    "nn.ConvTranspose1d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/NaiveConvolutionTranspose1d.cpp",
        [r"conv_transpose1d", r"transpose"],
    ),
    "nn.ConvTranspose2d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/NaiveConvolutionTranspose2d.cpp",
        [r"conv_transpose2d", r"transpose"],
    ),
    "nn.ConvTranspose3d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/NaiveConvolutionTranspose3d.cpp",
        [r"conv_transpose3d", r"transpose"],
    ),
    "torch.clamp": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/TensorCompare.cpp",
        [r"clamp", r"clamp_min", r"clamp_max"],
    ),
    "torch.softmax": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/SoftMax.cpp",
        [r"softmax"],
    ),
    "torch.log_softmax": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/SoftMax.cpp",
        [r"log_softmax"],
    ),
    "nn.LayerNorm": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/layer_norm.cpp",
        [r"layer_norm"],
    ),
    "torch.layer_norm": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/layer_norm.cpp",
        [r"layer_norm"],
    ),
    "nn.ReLU": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Activation.cpp",
        [r"relu"],
    ),
    "nn.GELU": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Activation.cpp",
        [r"gelu"],
    ),
    "nn.Sigmoid": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Activation.cpp",
        [r"sigmoid"],
    ),
    "nn.Tanh": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Activation.cpp",
        [r"tanh"],
    ),
    "torch.nn.functional.mish": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Activation.cpp",
        [r"mish"],
    ),
    "torch.max_pool1d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Pooling.cpp",
        [r"max_pool1d", r"max_pool"],
    ),
    "torch.max_pool2d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Pooling.cpp",
        [r"max_pool2d", r"max_pool"],
    ),
    "torch.max_pool3d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Pooling.cpp",
        [r"max_pool3d", r"max_pool"],
    ),
    "nn.MaxPool1d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Pooling.cpp",
        [r"max_pool1d", r"max_pool"],
    ),
    "nn.MaxPool2d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Pooling.cpp",
        [r"max_pool2d", r"max_pool"],
    ),
    "nn.MaxPool3d": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/Pooling.cpp",
        [r"max_pool3d", r"max_pool"],
    ),
    "torch.mean": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/ReduceOps.cpp",
        [r"mean"],
    ),
    "torch.sum": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/ReduceOps.cpp",
        [r"sum"],
    ),
    "torch.argmax": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/ReduceOps.cpp",
        [r"argmax"],
    ),
    "torch.argmin": (
        "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/ReduceOps.cpp",
        [r"argmin"],
    ),
}

GENERIC_URL = "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/README.md"


def _cache_file_for(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.txt"


def fetch_text(url: str) -> str:
    cp = _cache_file_for(url)
    if cp.exists():
        return cp.read_text(encoding="utf-8", errors="ignore")

    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read().decode("utf-8", errors="ignore")
    cp.write_text(data, encoding="utf-8")
    return data


def strip_existing_block(text: str) -> str:
    patterns = [
        re.compile(rf"{re.escape(BEGIN)}.*?{re.escape(END)}\n?", re.S),
        re.compile(r"# === End Operator Source References ===\n?", re.S),
        re.compile(
            r"# === Operator Source References \(auto-generated\) ===.*?# === End Operator Source References ===\n?",
            re.S,
        ),
        re.compile(
            r"# === Operator Source Snippets \(auto-generated\) ===.*?# === End Operator Source Snippets ===\n?",
            re.S,
        ),
    ]
    cleaned = text
    for p in patterns:
        cleaned = p.sub("", cleaned)
    return cleaned


def detect_ops(text: str):
    ops = set(NN_RE.findall(text)) | set(TORCH_RE.findall(text)) | set(F_RE.findall(text))
    return sorted(ops)


def _is_preprocessor_or_include(line: str) -> bool:
    s = line.strip()
    return (
        s.startswith("#include")
        or s.startswith("#define")
        or s.startswith("#if")
        or s.startswith("#endif")
        or s.startswith("#else")
        or s.startswith("#elif")
        or s.startswith("#pragma")
    )


def extract_excerpt(src: str, focus_patterns: list[str], max_lines: int = 64) -> list[str]:
    """
    Extract a code-focused excerpt near the first meaningful match.
    Preference order:
    1) Non-preprocessor match lines.
    2) Fallback to any match line.
    """
    lines = src.splitlines()
    idx = None
    fallback_idx = None

    for i, ln in enumerate(lines):
        lower_ln = ln.lower()
        matched = any(re.search(p.lower(), lower_ln) for p in focus_patterns)
        if not matched:
            continue
        if fallback_idx is None:
            fallback_idx = i
        if not _is_preprocessor_or_include(ln):
            idx = i
            break

    if idx is None:
        idx = fallback_idx
    if idx is None:
        return lines[:max_lines]

    # Try to anchor around a nearby function-like start instead of random single line.
    start = max(0, idx - 20)
    for j in range(idx, max(-1, idx - 80), -1):
        s = lines[j].strip()
        if _is_preprocessor_or_include(s):
            continue
        if ("(" in s and (s.endswith("{") or s.endswith(")") or s.endswith(") {"))):
            start = max(0, j - 2)
            break

    end = min(len(lines), start + max_lines)
    excerpt = lines[start:end]

    # Final cleanup: trim leading run of preprocessor includes so snippet starts on logic.
    k = 0
    while k < len(excerpt) and _is_preprocessor_or_include(excerpt[k]):
        k += 1
    if k > 0 and k < len(excerpt):
        excerpt = excerpt[k:]

    return excerpt


def build_block(ops):
    lines = [BEGIN]
    lines.append("# PyTorch internal C/C++ source excerpts for detected target operators:")
    lines.append("# (auto-fetched from upstream pytorch/pytorch)")

    added = False
    used_urls = set()
    for op in ops:
        entry = OP_SOURCE_MAP.get(op)
        if not entry:
            continue
        url, focuses = entry
        if (op, url) in used_urls:
            continue
        used_urls.add((op, url))

        try:
            src = fetch_text(url)
            excerpt = extract_excerpt(src, focuses)
            lines.append(f"# - operator: {op}")
            lines.append(f"# - source: {url}")
            for ln in excerpt:
                lines.append(f"#   {ln}")
            lines.append("#")
            added = True
        except Exception as e:
            lines.append(f"# - operator: {op}")
            lines.append(f"# - source: {url}")
            lines.append(f"#   [fetch failed: {e}]")
            lines.append("#")

    if not added:
        lines.append(f"# - source: {GENERIC_URL}")
        lines.append("#   [no direct operator mapping found for this file]")

    lines.append("# Notes:")
    lines.append("# - Excerpts are for semantic reference and may be backend/dispatch-specific.")
    lines.append("# - Keep Model/ModelNew task semantics as the primary ground truth.")
    lines.append(END)
    return "\n".join(lines) + "\n\n"


def insert_after_imports(text: str, block: str) -> str:
    lines = text.splitlines(keepends=True)
    idx = 0
    while idx < len(lines):
        s = lines[idx].strip()
        if s.startswith("import ") or s.startswith("from ") or s == "":
            idx += 1
            continue
        break
    return "".join(lines[:idx]) + block + "".join(lines[idx:])


def annotate_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    cleaned = strip_existing_block(original)
    ops = detect_ops(cleaned)
    block = build_block(ops)
    updated = insert_after_imports(cleaned, block)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main():
    changed = 0
    total = 0
    for d in LEVEL_DIRS:
        for p in sorted(d.glob("*.py")):
            total += 1
            if annotate_file(p):
                changed += 1
    print(f"Annotated {changed}/{total} files")


if __name__ == "__main__":
    main()
