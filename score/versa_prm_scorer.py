from __future__ import annotations

import time
from dataclasses import dataclass, field
import gc
import os
from typing import Any

from parsers.reasoning_parser import extract_reasoning_steps


DEFAULT_VERSA_PRM_MODEL_ID = "UW-Madison-Lee-Lab/VersaPRM-Base-3B"
DEFAULT_VERSA_PRM_BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_CANDIDATE_TOKEN_IDS = (12, 10)
DEFAULT_STEP_REWARD_TOKEN_ID = 23535
DEFAULT_STEP_SEPARATOR = " \n\n\n\n"
HF_MODEL_REQUIRED_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
    "adapter_config.json",
)


def ensure_versa_prm_model_cache(
    *,
    model_id: str = DEFAULT_VERSA_PRM_MODEL_ID,
    base_model_id: str = DEFAULT_VERSA_PRM_BASE_MODEL_ID,
    allow_download: bool = True,
) -> dict[str, Any]:
    """
    在正式 Stage2 scoring 前確認 VersaPRM 需要的 HuggingFace cache 已存在。

    Args:
     - model_id: VersaPRM adapter repo id 或本地路徑。
     - base_model_id: VersaPRM base model repo id 或本地路徑。
     - allow_download: 本地 cache 缺檔時是否允許連 HuggingFace 下載。

    Returns:
     - dict[str, Any]: adapter 與 base model 的 cache 檢查結果。
    """
    return {
        "model": _ensure_hf_repo_cached(model_id, allow_download=allow_download),
        "base_model": _ensure_hf_repo_cached(base_model_id, allow_download=allow_download),
    }


def _ensure_hf_repo_cached(repo_id_or_path: str, *, allow_download: bool) -> dict[str, Any]:
    repo_id_or_path = str(repo_id_or_path or "").strip()
    if not repo_id_or_path:
        return {
            "ok": False,
            "repo_id": "",
            "source": "empty",
            "path": "",
            "error": "empty_repo_id",
        }

    local_path = os.path.expandvars(os.path.expanduser(repo_id_or_path))
    if os.path.exists(local_path):
        return {
            "ok": True,
            "repo_id": repo_id_or_path,
            "source": "local_path",
            "path": os.path.abspath(local_path),
            "downloaded": False,
        }

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return {
            "ok": False,
            "repo_id": repo_id_or_path,
            "source": "huggingface_cache",
            "path": "",
            "downloaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        cached_path = snapshot_download(
            repo_id=repo_id_or_path,
            local_files_only=True,
            allow_patterns=list(HF_MODEL_REQUIRED_PATTERNS),
        )
        return {
            "ok": True,
            "repo_id": repo_id_or_path,
            "source": "huggingface_cache",
            "path": cached_path,
            "downloaded": False,
        }
    except Exception as cache_exc:
        if not allow_download:
            return {
                "ok": False,
                "repo_id": repo_id_or_path,
                "source": "huggingface_cache",
                "path": "",
                "downloaded": False,
                "error": f"{type(cache_exc).__name__}: {cache_exc}",
            }

    cached_path = snapshot_download(
        repo_id=repo_id_or_path,
        local_files_only=False,
        allow_patterns=list(HF_MODEL_REQUIRED_PATTERNS),
    )
    return {
        "ok": True,
        "repo_id": repo_id_or_path,
        "source": "huggingface_download",
        "path": cached_path,
        "downloaded": True,
    }


@dataclass
class VersaPRMStepScore:
    """
    單一 reasoning step 的 VersaPRM reward probability。

    Args:
        - step_index: reasoning step 的順序編號。
        - step_text: 該 reasoning step 的文字內容。
        - reward_probability: VersaPRM 對該步驟給出的 reward probability。

    Returns:
        - VersaPRMStepScore: 可序列化為 dict 的單步分數資料。
    """

    step_index: int
    step_text: str
    reward_probability: float

    def to_dict(self) -> dict[str, Any]:
        """
        將 step reward 轉成 JSON 友善的 dict。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: 包含 step_index、step_text 與 reward_probability。
        """
        return {
            "step_index": self.step_index,
            "step_text": self.step_text,
            "reward_probability": self.reward_probability,
        }


@dataclass
class VersaPRMScoreResult:
    """
    VersaPRM 對一段 reasoning 的 step-level reward 結果。

    Args:
        - scorer_name: scorer 名稱。
        - model_id: VersaPRM 模型 id。
        - base_model_id: base model id。
        - step_scores: 每個 reasoning step 的 reward probability。
        - metadata: 載入模式、device、step 對齊狀態等診斷資訊。

    Returns:
        - VersaPRMScoreResult: Stage2 可使用的 step reward 結果。
    """

    scorer_name: str
    model_id: str
    base_model_id: str
    step_scores: list[VersaPRMStepScore]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reward_probabilities(self) -> list[float]:
        """
        取得所有 step 的 reward probability。

        Args:
            - 無。

        Returns:
            - list[float]: 依 step 順序排列的 reward probability。
        """
        return [item.reward_probability for item in self.step_scores]

    @property
    def avg_reward_probability(self) -> float:
        """
        計算所有 reasoning steps 的平均 reward probability。

        Args:
            - 無。

        Returns:
            - float: 平均 reward probability；若沒有 step 則為 0。
        """
        if not self.step_scores:
            return 0.0
        return sum(self.reward_probabilities) / len(self.step_scores)

    def to_dict(self) -> dict[str, Any]:
        """
        將 scorer 結果轉成 JSON 友善的 dict。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: 包含 step_scores、平均 reward 與 metadata。
        """
        return {
            "scorer_name": self.scorer_name,
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "avg_reward_probability": round(self.avg_reward_probability, 6),
            "step_scores": [item.to_dict() for item in self.step_scores],
            "metadata": dict(self.metadata),
        }


class VersaPRMScorer:
    """
    使用 VersaPRM-Base-3B 對 Agent reasoning steps 產生 reward probabilities。

    Args:
        - model_id: VersaPRM 模型 id。
        - base_model_id: PEFT fallback 使用的 base model id。
        - load_mode: direct、peft 或 auto 載入模式。
        - device: auto、cuda 或 cpu。
        - dtype: auto、float16、bfloat16 或 float32。
        - candidate_token_ids: 官方 PRM reward softmax 使用的兩個 token id。
        - step_reward_token_id: 用於定位每個 step reward 的 token id。
        - step_separator: reasoning steps 之間的 separator。
        - local_files_only: 是否只使用本地 Hugging Face cache。

    Returns:
        - VersaPRMScorer: 可 lazy-load 模型並計算 step reward 的 scorer。
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_VERSA_PRM_MODEL_ID,
        base_model_id: str = DEFAULT_VERSA_PRM_BASE_MODEL_ID,
        load_mode: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        candidate_token_ids: tuple[int, int] = DEFAULT_CANDIDATE_TOKEN_IDS,
        step_reward_token_id: int = DEFAULT_STEP_REWARD_TOKEN_ID,
        step_separator: str = DEFAULT_STEP_SEPARATOR,
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.load_mode = load_mode
        self.device = device
        self.dtype = dtype
        self.candidate_token_ids = list(candidate_token_ids)
        self.step_reward_token_id = step_reward_token_id
        self.step_separator = step_separator
        self.local_files_only = local_files_only

        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._used_load_mode = ""
        self._resolved_device = ""
        self._resolved_dtype_name = ""

    def score_reasoning(
        self,
        *,
        question: str,
        reasoning: str,
        evidence: str = "",
    ) -> VersaPRMScoreResult:
        """
        對 raw Agent reasoning 產生每個 step 的 reward probability。

        Args:
            - question: 原始任務問題。
            - reasoning: Agent 的 reasoning 文字；此 helper 會先用 reasoning parser 解析。
            - evidence: 可選的 evidence context，會放進 PRM input。

        Returns:
            - VersaPRMScoreResult: 每個 reasoning step 的 reward probability。
        """
        steps, fallback_single_step = self._extract_steps(reasoning)
        result = self.score_steps(
            question=question,
            reasoning_steps=steps,
            evidence=evidence,
        )
        result.metadata["fallback_single_step"] = fallback_single_step
        result.metadata["input_was_raw_reasoning"] = True
        return result

    def score_steps(
        self,
        *,
        question: str,
        reasoning_steps: list[tuple[int, str]],
        evidence: str = "",
    ) -> VersaPRMScoreResult:
        """
        對已解析好的 reasoning steps 產生每步 reward probability。

        Args:
            - question: 原始任務問題。
            - reasoning_steps: 已由上游 parser 解析完成的 (step_index, step_text) 清單。
            - evidence: 可選的 evidence context，會放進 PRM input。

        Returns:
            - VersaPRMScoreResult: 每個 reasoning step 的 reward probability。
        """
        started_at = time.perf_counter()
        steps = self._clean_steps(reasoning_steps)
        if not steps:
            return VersaPRMScoreResult(
                scorer_name="versa_prm",
                model_id=self.model_id,
                base_model_id=self.base_model_id,
                step_scores=[],
                metadata={
                    "error": "empty_reasoning",
                    "step_count": 0,
                    "reward_count": 0,
                    "input_was_raw_reasoning": False,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 6),
                },
            )

        self.load()
        input_text = self.build_input(
            question=question,
            steps=[text for _, text in steps],
            evidence=evidence,
        )
        input_ids = self._torch.tensor([self._tokenizer.encode(input_text)]).to(
            self._model.device
        )
        with self._torch.no_grad():
            logits = self._model(input_ids).logits[:, :, self.candidate_token_ids]
            scores = logits.softmax(dim=-1)[:, :, 1]
            raw_step_scores = scores[input_ids == self.step_reward_token_id]

        reward_probabilities = [
            round(float(value), 6) for value in raw_step_scores.tolist()
        ]
        step_scores = [
            VersaPRMStepScore(
                step_index=step_index,
                step_text=step_text,
                reward_probability=reward_probabilities[index],
            )
            for index, (step_index, step_text) in enumerate(steps)
            if index < len(reward_probabilities)
        ]
        metadata = {
            "load_mode": self._used_load_mode,
            "device": str(self._model.device),
            "dtype": self._resolved_dtype_name,
            "step_count": len(steps),
            "reward_count": len(reward_probabilities),
            "reward_count_matches_step_count": len(steps) == len(reward_probabilities),
            "fallback_single_step": False,
            "input_was_raw_reasoning": False,
            "candidate_token_ids": list(self.candidate_token_ids),
            "step_reward_token_id": self.step_reward_token_id,
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        }
        if len(steps) != len(reward_probabilities):
            metadata["warning"] = "reward_count_does_not_match_step_count"
            metadata["raw_reward_probabilities"] = reward_probabilities

        return VersaPRMScoreResult(
            scorer_name="versa_prm",
            model_id=self.model_id,
            base_model_id=self.base_model_id,
            step_scores=step_scores,
            metadata=metadata,
        )

    def load(self) -> None:
        """
        Lazy-load tokenizer 與 VersaPRM 模型。

        Args:
            - 無。

        Returns:
            - None。
        """
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._torch = torch
        resolved_device = self._resolve_device(torch)
        resolved_dtype = self._resolve_dtype(torch, resolved_device)
        self._resolved_device = resolved_device
        self._resolved_dtype_name = self._dtype_name(resolved_dtype)

        tokenizer = self._load_tokenizer(AutoTokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        model, used_load_mode = self._load_model(
            AutoModelForCausalLM=AutoModelForCausalLM,
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
        )
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._used_load_mode = used_load_mode

    def _load_tokenizer(self, AutoTokenizer: Any) -> Any:
        try:
            return AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
        except Exception:
            return AutoTokenizer.from_pretrained(
                self.base_model_id,
                local_files_only=self.local_files_only,
            )

    def unload(self) -> dict[str, Any]:
        """
        Release the loaded VersaPRM model and clear CUDA cache when possible.

        Args:
            - None.

        Returns:
            - dict[str, Any]: Unload status for runtime metadata.
        """
        was_loaded = self._model is not None or self._tokenizer is not None
        torch_module = self._torch
        device = self._resolved_device

        self._model = None
        self._tokenizer = None
        gc.collect()

        cuda_cache_cleared = False
        if torch_module is not None:
            cuda = getattr(torch_module, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)):
                try:
                    if cuda.is_available():
                        cuda.empty_cache()
                        cuda.ipc_collect()
                        cuda_cache_cleared = True
                except Exception:
                    cuda_cache_cleared = False

        return {
            "was_loaded": was_loaded,
            "device": device,
            "cuda_cache_cleared": cuda_cache_cleared,
        }

    def build_input(
        self,
        *,
        question: str,
        steps: list[str],
        evidence: str = "",
    ) -> str:
        """
        建立 VersaPRM 使用的 question/evidence/reasoning input。

        Args:
            - question: 原始任務問題。
            - steps: reasoning step 文字清單。
            - evidence: 可選 evidence context。

        Returns:
            - str: 送入 tokenizer 的 PRM input。
        """
        context_parts = [f"Question: {str(question or '').strip()}"]
        if str(evidence or "").strip():
            context_parts.append(f"Evidence:\n{str(evidence).strip()}")
        prefix = "\n\n".join(context_parts).strip()
        return prefix + " \n\n" + self.step_separator.join(steps) + self.step_separator

    def _extract_steps(self, reasoning: str) -> tuple[list[tuple[int, str]], bool]:
        parsed_steps = extract_reasoning_steps(reasoning)
        if parsed_steps:
            return self._clean_steps(parsed_steps), False
        text = " ".join(str(reasoning or "").strip().split())
        if not text:
            return [], False
        return [(1, text)], True

    def _clean_steps(
        self,
        reasoning_steps: list[tuple[int, str]],
    ) -> list[tuple[int, str]]:
        cleaned_steps: list[tuple[int, str]] = []
        for fallback_index, item in enumerate(reasoning_steps or [], start=1):
            try:
                step_index = int(item[0])
                step_text = " ".join(str(item[1] or "").strip().split())
            except (TypeError, ValueError, IndexError):
                continue
            if not step_text:
                continue
            cleaned_steps.append((step_index if step_index > 0 else fallback_index, step_text))
        return cleaned_steps

    def _load_model(
        self,
        *,
        AutoModelForCausalLM: Any,
        resolved_device: str,
        resolved_dtype: Any,
    ) -> tuple[Any, str]:
        if self.load_mode in {"auto", "direct"}:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=resolved_dtype,
                    local_files_only=self.local_files_only,
                )
                model.to(resolved_device)
                return model, "direct"
            except Exception:
                if self.load_mode == "direct":
                    raise

        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            dtype=resolved_dtype,
            local_files_only=self.local_files_only,
        )
        base_model.to(resolved_device)
        model = PeftModel.from_pretrained(
            base_model,
            self.model_id,
            local_files_only=self.local_files_only,
        )
        model.to(resolved_device)
        return model, "peft"

    def _resolve_device(self, torch_module: Any) -> str:
        if self.device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return self.device

    def _resolve_dtype(self, torch_module: Any, resolved_device: str) -> Any:
        if self.dtype == "float16":
            return torch_module.float16
        if self.dtype == "bfloat16":
            return torch_module.bfloat16
        if self.dtype == "float32":
            return torch_module.float32
        if resolved_device == "cuda":
            return torch_module.float16
        return torch_module.float32

    def _dtype_name(self, dtype: Any) -> str:
        return str(dtype).replace("torch.", "")


__all__ = [
    "DEFAULT_VERSA_PRM_BASE_MODEL_ID",
    "DEFAULT_VERSA_PRM_MODEL_ID",
    "ensure_versa_prm_model_cache",
    "VersaPRMScorer",
    "VersaPRMScoreResult",
    "VersaPRMStepScore",
]
