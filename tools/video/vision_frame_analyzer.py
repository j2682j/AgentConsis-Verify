from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .config import VideoEvidenceConfig
from .models import FrameAnalysisResult, FrameItem


class VisionFrameAnalyzer:
    """
    Analyze sampled frames with an Ollama vision model.

    Args:
        - config: Video evidence runtime configuration.

    Returns:
        - VisionFrameAnalyzer: Per-frame visual evidence analyzer.
    """

    def __init__(self, config: VideoEvidenceConfig) -> None:
        self.config = config

    def analyze(self, *, frame: FrameItem, question: str, answer_role: str = "") -> FrameAnalysisResult:
        """
        Ask the vision model for one frame-local observation.

        Args:
            - frame: Sampled frame metadata.
            - question: Original GAIA task.
            - answer_role: Optional answer role from routing or planner.

        Returns:
            - FrameAnalysisResult: Parsed visual observation.
        """
        try:
            image_b64 = base64.b64encode(frame.image_path.read_bytes()).decode("ascii")
            raw = self._call_ollama(
                image_b64=image_b64,
                question=question,
                answer_role=answer_role,
            )
            return self._parse_result(raw, frame=frame)
        except Exception as exc:
            return FrameAnalysisResult(
                frame_id=frame.frame_id,
                timestamp_sec=frame.timestamp_sec,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _call_ollama(self, *, image_b64: str, question: str, answer_role: str) -> str:
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt(question=question, answer_role=answer_role),
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": self.config.max_tokens,
            },
        }
        request = urllib.request.Request(
            self._ollama_chat_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama vision HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama vision request failed: {exc.reason}") from exc
        message = data.get("message") or {}
        return str(message.get("content") or message.get("thinking") or data.get("response") or "").strip()

    def _prompt(self, *, question: str, answer_role: str) -> str:
        role_hint = answer_role or "visual evidence"
        return (
            "Analyze only this video frame for the task.\n"
            "Do not answer from memory. Use visible content only.\n"
            "Do not include reasoning or <think> text.\n"
            f"Answer role: {role_hint}\n"
            f"Question: {question}\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "visible": true,\n'
            '  "answer_value": "",\n'
            '  "count": null,\n'
            '  "evidence": "short visual observation",\n'
            '  "confidence": 0.0\n'
            "}\n"
            "If the question asks for the highest simultaneous count, count only objects or species visible in this frame."
        )

    def _parse_result(self, raw: str, *, frame: FrameItem) -> FrameAnalysisResult:
        cleaned_raw = self._strip_thinking(raw)
        payload = self._extract_json(cleaned_raw)
        if payload is None:
            count = self._extract_count(cleaned_raw)
            return FrameAnalysisResult(
                frame_id=frame.frame_id,
                timestamp_sec=frame.timestamp_sec,
                ok=bool(cleaned_raw.strip()),
                answer_value=str(count) if count is not None else "",
                count=count,
                evidence=cleaned_raw.strip()[:600],
                confidence=0.0,
                raw_response=raw,
            )
        answer_value = str(payload.get("answer_value") or "").strip()
        count = payload.get("count")
        parsed_count = int(count) if isinstance(count, int) else self._extract_count(str(count or answer_value))
        confidence = payload.get("confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        return FrameAnalysisResult(
            frame_id=frame.frame_id,
            timestamp_sec=frame.timestamp_sec,
            ok=bool(payload.get("visible", True)),
            answer_value=answer_value or (str(parsed_count) if parsed_count is not None else ""),
            count=parsed_count,
            evidence=str(payload.get("evidence") or "").strip()[:600],
            confidence=max(0.0, min(confidence_value, 1.0)),
            raw_response=raw,
        )

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        cleaned = str(raw or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _strip_thinking(raw: str) -> str:
        cleaned = str(raw or "")
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^\s*<think>.*?(?=\{)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    @staticmethod
    def _extract_count(text: str) -> int | None:
        match = re.search(r"\b([0-9]{1,3})\b", str(text or ""))
        return int(match.group(1)) if match else None

    @staticmethod
    def _ollama_chat_endpoint() -> str:
        base_url = (
            os.getenv("OLLAMA_NATIVE_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).strip()
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url.rstrip("/") + "/api/chat"


__all__ = ["VisionFrameAnalyzer"]
