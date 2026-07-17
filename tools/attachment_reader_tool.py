from __future__ import annotations

from pathlib import Path
from typing import Any

from .attachment_reader import AttachmentEvidenceBuilder
from .attachment_tool_coordinator import AttachmentToolCoordinator
from .attachment_workspace import AttachmentWorkspace
from .base import Tool, ToolParameter


class AttachmentReaderTool(Tool):
    """
    Wrap AttachmentEvidenceBuilder as a ToolManager tool.

    Args:
        - builder: Optional attachment evidence builder.

    Returns:
        - AttachmentReaderTool: Tool wrapper for reading task attachments.
    """

    def __init__(
        self,
        builder: AttachmentEvidenceBuilder | None = None,
        coordinator: AttachmentToolCoordinator | None = None,
    ) -> None:
        super().__init__(
            name="attachment_reader",
            description="Read a task attachment and return extracted evidence context.",
            capabilities={
                "attachment.read",
                "attachment.archive",
                "attachment.table",
                "attachment.document",
                "attachment.media",
            },
            attachment_types={
                "csv",
                "docx",
                "jpg",
                "jpeg",
                "json",
                "mp3",
                "mp4",
                "pdf",
                "png",
                "pptx",
                "tsv",
                "txt",
                "wav",
                "xls",
                "xlsx",
                "xml",
                "zip",
            },
            deterministic=False,
        )
        self.builder = builder or AttachmentEvidenceBuilder()
        self.coordinator = coordinator or AttachmentToolCoordinator()

    def run(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Read an attachment through AttachmentEvidenceBuilder.

        Args:
            - parameters: Tool arguments containing question/input and attachment metadata.

        Returns:
            - dict[str, Any]: Attachment context, usage metadata, and read status.
        """
        question = str(
            parameters.get("question")
            or parameters.get("input")
            or parameters.get("query")
            or ""
        )
        attachment = self._attachment_from_parameters(parameters)
        result = self.builder.build(question, attachment)
        metadata = dict(result.get("metadata") or {})
        return {
            "used": bool(result.get("used")),
            "context": str(result.get("context") or ""),
            "profile": dict(result.get("profile") or {}),
            "parsed_payload": dict(result.get("parsed_payload") or {}),
            "metadata": metadata,
            "tool_usage": result.get("tool_usage", []),
        }

    def run_with_context(
        self,
        parameters: dict[str, Any],
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Reuse a task attachment workspace when Stage1 requests attachment evidence."""

        workspace = runtime_context.get("attachment_workspace")
        if not isinstance(workspace, AttachmentWorkspace):
            return self.run(parameters)
        question = str(
            parameters.get("question")
            or runtime_context.get("question")
            or parameters.get("input")
            or ""
        ).strip()
        information_need = str(
            parameters.get("information_need")
            or parameters.get("input")
            or question
        ).strip()
        attachment = self._attachment_from_parameters(parameters) or dict(workspace.attachment)
        return self.coordinator.run(
            question=question,
            information_need=information_need,
            attachment=attachment,
            workspace=workspace,
            search_context=str(runtime_context.get("search_context") or ""),
        )

    def get_parameters(self) -> list[ToolParameter]:
        """
        Return the attachment reader parameter schema.

        Args:
            - None.

        Returns:
            - list[ToolParameter]: Parameters accepted by attachment_reader.
        """
        return [
            ToolParameter(
                name="question",
                type="string",
                description="Question used to guide attachment extraction.",
                required=True,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="Absolute or relative path to the attachment file.",
                required=True,
            ),
            ToolParameter(
                name="information_need",
                type="string",
                description="One specific fact or operation needed from the attachment.",
                required=False,
            ),
        ]

    def _attachment_from_parameters(self, parameters: dict[str, Any]) -> dict[str, Any] | None:
        attachment = parameters.get("attachment")
        if isinstance(attachment, dict) and attachment:
            return dict(attachment)

        file_path = str(
            parameters.get("file_path")
            or parameters.get("path")
            or parameters.get("attachment_path")
            or ""
        ).strip()
        if not file_path:
            return None

        path = Path(file_path)
        return {
            "file_path": file_path,
            "file_name": str(parameters.get("file_name") or path.name),
            "extension": str(parameters.get("extension") or path.suffix).lower(),
        }


__all__ = ["AttachmentReaderTool"]
