"""A second blind set showing what the agent is actually handed.

The raw set asks what a block carried at its source. It cannot answer whether
any of that reached the model, because `_compress_multiline_text` strips every
line and drops the blank ones before the budget is even consulted -- so a block
annotated `STRUCTURAL_INPUT` on the strength of its indentation may be
describing something the agent never sees. Task 044 asks which stanza of a poem
is indented, and the indentation is gone by then.

Showing only the compressed text would fail the other way: it measures what
arrived and cannot say what was lost on the way, because the loss is invisible
once it has happened.

So there are two sets and they are annotated apart. Same blocks, different
question, different shuffled ids, no pairing shown. Anchoring is the risk being
designed against: an annotator who has just called a block structurally
essential will not then judge its flattened form freshly.

The mechanical fields alongside record what the compressor did, and stop there.
`structure_delta_present` says the formatting changed, not that anything was
lost -- whether it mattered is what the annotation is for, and inferring harm
from a format diff would be the measurement answering its own question.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, r"c:/SCP")
sys.path.insert(0, r"c:/SCP/scripts/replay")

from build_block_role_annotation import neutralise
from context.compressor_alignment import align
from context.evidence_block_lineage import digest, parse_blocks

RUN = "c:/SCP/outputs/level1_final_23/tasks"
OUT = "c:/SCP/outputs/block_role_annotation"

ANNOTATION_PROTOCOL_VERSION = 2
PERSPECTIVE = "agent_visible"


def structure_telemetry(raw: str, compressed: str) -> dict:
    """What the compressor changed about the shape, described and not judged.

    Newlines and runs of spaces inside a line survive; only the edges and the
    blank lines go. So indentation-based structure dies and column alignment may
    not, which is why these are separate flags rather than one verdict.
    """

    raw_lines = raw.replace("\r\n", "\n").split("\n")
    return {
        "leading_indent_removed": any(
            line[:1].isspace() for line in raw_lines if line.strip()
        ),
        "trailing_whitespace_removed": any(
            line != line.rstrip() for line in raw_lines if line.strip()
        ),
        "blank_lines_removed": sum(1 for line in raw_lines if not line.strip()),
        "internal_multi_space_preserved": "  " in compressed,
        "newlines_preserved": "\n" in compressed if "\n" in raw else None,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    from context.stage1_context import Stage1ContextBuilder

    builder = Stage1ContextBuilder()
    max_lines = builder.config.max_context_lines
    max_chars = builder.config.max_context_chars

    units: list[dict] = []
    for path in sorted(glob.glob(f"{RUN}/*.json")):
        record = json.load(open(path, encoding="utf-8"))
        meta = (record.get("network_summary") or {}).get("metadata") or {}
        raw_result = {}
        for item in meta.get("tool_usage") or []:
            if item.get("tool_name") == "search" and isinstance(item.get("raw_result"), dict):
                raw_result = item["raw_result"]
                break
        summary = str(raw_result.get("summary") or "")
        if not summary:
            continue
        task = os.path.basename(path)[:3]
        question = str(record.get("question") or "")
        trimmed = builder._compress_multiline_text(
            summary, max_lines=max_lines, max_chars=max_chars
        ) or ""
        raw_blocks = {b.block_id: b.text for b in parse_blocks(summary)}
        seen_blocks = {b.block_id: b.text for b in parse_blocks(trimmed)}

        for block_id, raw_text in raw_blocks.items():
            visible = seen_blocks.get(block_id)
            if visible is None:
                # Never reached the agent, so there is nothing to judge as
                # delivered. It stays in the raw set and is counted as an
                # availability loss, not a fidelity one.
                continue
            alignment = align(raw_text, visible, max_lines=max_lines, max_chars=max_chars)
            units.append({
                "task_id": task,
                "block_id": block_id,
                "parent_raw_hash": digest(raw_text),
                "question": question,
                "block_text": neutralise(visible),
                "provenance": "web_retrieval",
                "alignment_shape": alignment["shape"],
                "content_survival": (
                    "kept" if alignment["shape"] == "exact_after_line_normalization"
                    else "partial" if alignment["shape"] == "prefix_after_line_normalization"
                    else "unsupported"
                ),
                "line_limit_applied": alignment["line_limited_chars"] < alignment["normalised_chars"],
                "char_limit_applied": alignment["truncation_marker_added"],
                **structure_telemetry(raw_text, visible),
            })

    for unit in units:
        unit["structure_delta_present"] = bool(
            unit["leading_indent_removed"]
            or unit["trailing_whitespace_removed"]
            or unit["blank_lines_removed"]
        )

    # Shuffled on a different salt from the raw set, so the two orders share no
    # structure and a reader of both cannot pair them by position.
    units.sort(key=lambda u: hashlib.sha256(
        f"agentvisible:{u['task_id']}:{u['block_id']}".encode()
    ).hexdigest())
    for index, unit in enumerate(units, 1):
        unit["annotation_id"] = f"V{index:03d}"

    blind_path = f"{OUT}/agent_visible_blind.jsonl"
    with open(blind_path, "w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps({
                "annotation_id": unit["annotation_id"],
                "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
                "perspective": PERSPECTIVE,
                "conditioning": "full_question",
                "question": unit["question"],
                "provenance": unit["provenance"],
                "block_text": unit["block_text"],
                "roles": [],
                "agent_visible_usability": "",
                "structure_available": "",
                "standalone_answerable": "",
                "requires_other_blocks": "",
                "polarity": "",
                "annotation_confidence": "",
                "notes": "",
            }, ensure_ascii=False) + "\n")

    key_path = f"{OUT}/_agent_visible_key.jsonl"
    with open(key_path, "w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps({
                k: unit[k] for k in (
                    "annotation_id", "task_id", "block_id", "parent_raw_hash",
                    "alignment_shape", "content_survival", "line_limit_applied",
                    "char_limit_applied", "leading_indent_removed",
                    "trailing_whitespace_removed", "blank_lines_removed",
                    "internal_multi_space_preserved", "newlines_preserved",
                    "structure_delta_present",
                )
            }, ensure_ascii=False) + "\n")

    from collections import Counter

    survival = Counter(u["content_survival"] for u in units)
    delta = sum(1 for u in units if u["structure_delta_present"])
    indent = sum(1 for u in units if u["leading_indent_removed"])
    spaces = sum(1 for u in units if u["internal_multi_space_preserved"])
    print(f"agent-visible blind set: {len(units)} units -> {blind_path}")
    print(f"   sha256 {hashlib.sha256(open(blind_path,'rb').read()).hexdigest()[:16]}")
    print(f"   content_survival: {dict(survival)}")
    print(f"   structure_delta_present {delta}/{len(units)}"
          f"（行首縮排被移除 {indent}）")
    print(f"   行內多重空格仍存在 {spaces}/{len(units)}"
          f" —— 縮排型結構會死，欄位對齊未必")
    print(f"   標籤: source_role 在 raw set；此處為 evidence role +"
          f" agent_visible_usability + structure_available")
    print(f"   structure_delta_present 只表示格式被改變，是否有害由盲標決定")


if __name__ == "__main__":
    sys.exit(main())
