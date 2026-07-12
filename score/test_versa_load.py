from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL_ID = "UW-Madison-Lee-Lab/VersaPRM-Base-3B"
DEFAULT_BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_CANDIDATE_TOKEN_IDS = (12, 10)
DEFAULT_STEP_REWARD_TOKEN_ID = 23535
DEFAULT_STEP_SEPARATOR = " \n\n\n\n"


@dataclass
class VersaLoadTestResult:
    model_id: str
    base_model_id: str
    load_mode: str
    device: str
    dtype: str
    step_count: int
    reward_count: int
    reward_probabilities: list[float]
    mapped_scores: list[float]
    elapsed_seconds: float

    def to_lines(self) -> list[str]:
        return [
            "[OK] VersaPRM load test completed.",
            f"model_id={self.model_id}",
            f"base_model_id={self.base_model_id}",
            f"load_mode={self.load_mode}",
            f"device={self.device}",
            f"dtype={self.dtype}",
            f"step_count={self.step_count}",
            f"reward_count={self.reward_count}",
            f"reward_probabilities={self.reward_probabilities}",
            f"mapped_scores={self.mapped_scores}",
            f"elapsed_seconds={self.elapsed_seconds:.3f}",
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load VersaPRM-Base-3B and run a small step-reward smoke test. "
            "This script is intentionally standalone and does not touch SCP runtime."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument(
        "--load-mode",
        choices=("auto", "direct", "peft"),
        default="auto",
        help=(
            "auto tries direct loading first and falls back to PEFT adapter loading. "
            "direct follows the Hugging Face model-card example. "
            "peft loads the Llama base model first, then the Versa adapter."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--candidate-token-ids",
        default=",".join(str(value) for value in DEFAULT_CANDIDATE_TOKEN_IDS),
        help="Comma-separated candidate token ids. Official default: 12,10.",
    )
    parser.add_argument(
        "--step-reward-token-id",
        type=int,
        default=DEFAULT_STEP_REWARD_TOKEN_ID,
        help="Token id used by the official example to locate step rewards.",
    )
    parser.add_argument(
        "--step-separator",
        default=DEFAULT_STEP_SEPARATOR,
        help="Separator inserted between reasoning steps.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Only test tokenizer/model loading; skip the forward pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            "[ERROR] Missing dependency. Install with:\n"
            "C:\\SCP\\venv312\\Scripts\\python.exe -m pip install "
            "torch transformers accelerate peft safetensors sentencepiece "
            "huggingface_hub",
            file=sys.stderr,
        )
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    candidate_token_ids = parse_candidate_token_ids(args.candidate_token_ids)

    print(f"[INFO] Loading tokenizer: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        return handle_load_error(exc, base_model_id=args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(
        "[INFO] Loading model "
        f"mode={args.load_mode} device={device} dtype={dtype_name(dtype)}"
    )
    try:
        model, used_load_mode = load_model(
            AutoModelForCausalLM=AutoModelForCausalLM,
            model_id=args.model_id,
            base_model_id=args.base_model_id,
            load_mode=args.load_mode,
            device=device,
            dtype=dtype,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        return handle_load_error(exc, base_model_id=args.base_model_id)
    model.eval()

    if args.skip_forward:
        elapsed = time.perf_counter() - started_at
        print("[OK] VersaPRM loaded. Forward pass skipped.")
        print(f"load_mode={used_load_mode}")
        print(f"elapsed_seconds={elapsed:.3f}")
        return 0

    question, steps = sample_question_and_steps()
    input_text = build_prm_input(
        question=question,
        steps=steps,
        step_separator=args.step_separator,
    )
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(model.device)
    print(f"[INFO] Running forward pass with {len(steps)} steps.")

    with torch.no_grad():
        logits = model(input_ids).logits[:, :, candidate_token_ids]
        scores = logits.softmax(dim=-1)[:, :, 1]
        step_scores = scores[input_ids == args.step_reward_token_id]

    reward_probabilities = [round(float(value), 6) for value in step_scores.tolist()]
    mapped_scores = [round(2 * value - 1, 6) for value in reward_probabilities]
    result = VersaLoadTestResult(
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        load_mode=used_load_mode,
        device=str(model.device),
        dtype=dtype_name(dtype),
        step_count=len(steps),
        reward_count=len(reward_probabilities),
        reward_probabilities=reward_probabilities,
        mapped_scores=mapped_scores,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    for line in result.to_lines():
        print(line)

    if result.reward_count != result.step_count:
        print(
            "[ERROR] Reward count does not match reasoning step count. "
            "Check the step separator or reward token id.",
            file=sys.stderr,
        )
        return 1
    return 0


def resolve_device(torch_module: Any, requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested_device


def resolve_dtype(torch_module: Any, requested_dtype: str, device: str) -> Any:
    if requested_dtype == "float16":
        return torch_module.float16
    if requested_dtype == "bfloat16":
        return torch_module.bfloat16
    if requested_dtype == "float32":
        return torch_module.float32
    if device == "cuda":
        return torch_module.float16
    return torch_module.float32


def dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def handle_load_error(exc: Exception, *, base_model_id: str) -> int:
    if is_gated_model_error(exc):
        print(
            "[ERROR] Hugging Face gated model access is not available yet.",
            file=sys.stderr,
        )
        print(f"[INFO] base_model_id={base_model_id}", file=sys.stderr)
        print(
            "[HINT] Wait for Meta approval, then run `huggingface-cli login` "
            "with the approved account.",
            file=sys.stderr,
        )
        print(f"[DETAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1


def is_gated_model_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "gated",
        "unauthorized",
        "forbidden",
        "401",
        "403",
        "access to model",
        "restricted",
        "awaiting a review",
        "not authorized",
        "permission",
    )
    return any(marker in text for marker in markers)


def parse_candidate_token_ids(value: str) -> list[int]:
    token_ids: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        token_ids.append(int(item))
    if len(token_ids) != 2:
        raise ValueError("--candidate-token-ids must contain exactly two token ids.")
    return token_ids


def load_model(
    *,
    AutoModelForCausalLM: Any,
    model_id: str,
    base_model_id: str,
    load_mode: str,
    device: str,
    dtype: Any,
    local_files_only: bool,
) -> tuple[Any, str]:
    if load_mode in {"auto", "direct"}:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                local_files_only=local_files_only,
            )
            if device == "cpu":
                model.to(device)
            return model, "direct"
        except Exception as exc:
            if load_mode == "direct":
                raise
            print(f"[WARN] Direct load failed, falling back to PEFT: {exc}")

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "PEFT fallback requires `peft`. Install it with: "
            "C:\\SCP\\venv312\\Scripts\\python.exe -m pip install peft"
        ) from exc

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_id,
        local_files_only=local_files_only,
    )
    if device == "cpu":
        model.to(device)
    return model, "peft"


def sample_question_and_steps() -> tuple[str, list[str]]:
    question = (
        "Question: In Python 3, which of the following function converts a "
        "string to an int in python?\n"
        "A. short(x)\n"
        "B. float(x)\n"
        "C. integer(x [,base])\n"
        "D. double(x)\n"
        "E. int(x [,base])\n"
        "F. long(x [,base])\n"
        "G. num(x)\n"
        "H. str(x)\n"
        "I. char(x)\n"
        "J. digit(x [,base])"
    )
    steps = [
        "To convert a string to an integer in Python 3, we use the built-in function int().",
        "The int() function takes the string to be converted and an optional base.",
        "Looking at the options, the correct function is option E: int(x [,base]).",
        "The answer is (E).",
    ]
    return question, steps


def build_prm_input(
    *,
    question: str,
    steps: list[str],
    step_separator: str,
) -> str:
    return question + " \n\n" + step_separator.join(steps) + step_separator


if __name__ == "__main__":
    raise SystemExit(main())
