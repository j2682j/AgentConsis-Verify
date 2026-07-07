from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol


class JSONLRecord(Protocol):
    def to_dict(self) -> dict[str, str]: ...


class JSONLExporter:
    """
    將 corpus records 寫成 UTF-8 JSONL。

    Args:
        - ensure_ascii: 是否將非 ASCII 文字轉成 Unicode escape。

    Returns:
        - JSONLExporter: 可覆寫或附加 JSONL 的匯出器。
    """

    def __init__(self, *, ensure_ascii: bool = False) -> None:
        self.ensure_ascii = ensure_ascii

    def export(
        self,
        records: Iterable[JSONLRecord],
        output_path: str | Path,
        *,
        append: bool = False,
    ) -> int:
        """
        匯出 corpus JSONL。

        Args:
            - records: 具有 to_dict() 的 corpus records。
            - output_path: JSONL 輸出路徑。
            - append: 是否附加至既有檔案。

        Returns:
            - int: 寫入的 record 數量。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            for record in records:
                payload = record.to_dict()
                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=self.ensure_ascii,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
                count += 1
        return count


__all__ = ["JSONLExporter", "JSONLRecord"]
