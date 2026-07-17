from __future__ import annotations

from pathlib import Path

from ..base import HandlerInput
from .base import AdapterResult, payload_provenance


class AttachmentFileInputAdapter:
    """Bind specialized image handlers to the exact parsed attachment file."""

    handler_names = {"fraction_document", "chess_image_solver"}

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        del handler_name
        attachment = handler_input.attachment if isinstance(handler_input.attachment, dict) else {}
        file_path = str(attachment.get("file_path") or attachment.get("path") or "").strip()
        exists = bool(file_path and Path(file_path).is_file())
        return AdapterResult(
            status="ready" if exists else "missing_inputs",
            inputs={"file_path": file_path} if exists else {},
            missing_inputs=[] if exists else ["file_path"],
            input_provenance=(
                {
                    "source": "specialized_attachment_input",
                    "file_path": file_path,
                    "parse_status": "success",
                }
                if exists
                else payload_provenance(handler_input)
            ),
            reason="attachment_file_path" if exists else "attachment_file_path_missing",
        )


__all__ = ["AttachmentFileInputAdapter"]
