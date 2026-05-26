from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..models import AttachmentReaderConfig


class CodeAttachmentReader:
    """CodeAttachmentReader 類別，封裝此模組的資料結構與服務邏輯。"""
    def __init__(self, config: AttachmentReaderConfig) -> None:
        """初始化物件與必要狀態。"""
        self.config = config

    def read_python(self, file_path: Path) -> str:
        """read_python 的主要實作。"""
        source = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            completed = subprocess.run(
                [sys.executable, str(file_path)],
                cwd=str(file_path.parent),
                capture_output=True,
                text=True,
                timeout=self.config.python_timeout,
            )
            execution = (
                f"exit_code={completed.returncode}\n"
                f"stdout:\n{completed.stdout.strip() or '(empty)'}\n"
                f"stderr:\n{completed.stderr.strip() or '(empty)'}"
            )
        except subprocess.TimeoutExpired:
            execution = f"execution timed out after {self.config.python_timeout} seconds"
        return f"Python source:\n{source}\n\nExecution result:\n{execution}"

