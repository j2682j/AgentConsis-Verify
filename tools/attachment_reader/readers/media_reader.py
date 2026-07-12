from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from ..models import AttachmentReaderConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}


class MediaAttachmentReader:
    """MediaAttachmentReader 類別，封裝此模組的資料結構與服務邏輯。"""
    def __init__(self, config: AttachmentReaderConfig) -> None:
        """初始化物件與必要狀態。"""
        self.config = config

    def read_image(self, question: str, file_path: Path) -> str:
        """read_image 的主要實作。"""
        image_bytes = file_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        endpoint = self._ollama_chat_endpoint()
        prompt = (
            "You are extracting evidence from an image attachment for a GAIA benchmark question.\n"
            "Read the image carefully. Return concise structured text only.\n"
            "Do not include reasoning, chain-of-thought, or <think> text.\n"
            "Include:\n"
            "- visible text/OCR, if any\n"
            "- important objects, layout, chart/table values, board positions, symbols, numbers, colors, and labels\n"
            "- facts directly relevant to the question\n"
            "- uncertainties if the image is ambiguous\n\n"
            f"Question:\n{question}"
        )
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 1024,
            },
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.vision_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama vision HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama vision request failed: {exc.reason}") from exc

        data = json.loads(raw)
        message = data.get("message") or {}
        content = str(message.get("content", "") or "").strip()
        if not content:
            content = str(data.get("response", "") or "").strip()
        if not content:
            content = str(message.get("thinking", "") or "").strip()
        content = self._strip_thinking(content)
        if not content:
            raise RuntimeError("Ollama vision response did not include text content")

        return f"Ollama vision model: {self.config.vision_model}\n{content}"

    @staticmethod
    def _strip_thinking(text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^\s*<think>.*?(?=\{|\w)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def _ollama_chat_endpoint(self) -> str:
        """_ollama_chat_endpoint 的內部輔助實作。"""
        base_url = (
            os.getenv("OLLAMA_NATIVE_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).strip()
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url.rstrip("/") + "/api/chat"

    def analyze_audio(self, question: str, file_path: Path) -> str:
        """analyze_audio 的主要實作。"""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed in the Python environment running this evaluation"
            ) from exc

        model = WhisperModel(
            self.config.audio_model_size,
            device=self.config.audio_device,
            compute_type=self.config.audio_compute_type,
        )
        segments, info = model.transcribe(
            str(file_path),
            beam_size=5,
            vad_filter=True,
        )

        lines: list[str] = []
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(segment, "start", 0.0) or 0.0)
            end = float(getattr(segment, "end", 0.0) or 0.0)
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")

        language = str(getattr(info, "language", "") or "unknown")
        probability = getattr(info, "language_probability", None)
        probability_text = ""
        if probability is not None:
            try:
                probability_text = f" confidence={float(probability):.2f}"
            except Exception:
                probability_text = f" confidence={probability}"

        transcript = "\n".join(lines).strip() or "(empty transcription)"
        return (
            "Audio transcription:\n"
            f"- faster_whisper_model: {self.config.audio_model_size}\n"
            f"- device: {self.config.audio_device}\n"
            f"- compute_type: {self.config.audio_compute_type}\n"
            f"- detected_language: {language}{probability_text}\n"
            f"- question_focus: {question}\n"
            "Transcript:\n"
            f"{transcript}"
        )
