"""How the line compressor removes text, decided before any segment is drawn.

The segmenter is frozen first and its source hashed, because the question this
probe answers -- prefix cut, ordered subsequence, or scattered selection -- is
exactly the kind of answer that invites redrawing the cuts to suit it. Freezing
first makes that impossible rather than merely discouraged.

Alignment is computed on line identity alone. No answer, no question terms, no
relevance ordering: the probe reports the shape of the removal, and any judgement
about what the removal cost belongs to the blind annotation, not here.

Four shapes are distinguished, because they imply different things and the first
is easy to assume without checking. A prefix cut means survival is decided by
position and nothing else. An ordered subsequence means whole passages are being
skipped while order holds. A discontinuous selection means order is not even
preserved. And an unalignable result means the text was rewritten rather than
reduced, which would make offset-based lineage meaningless.
"""

from __future__ import annotations

import difflib
import glob
import hashlib
import inspect
import json
import os
import sys

sys.path.insert(0, r"c:/SCP")
sys.path.insert(0, r"c:/SCP/scripts/replay")

from build_block_role_annotation import CHILD_TARGET_CHARS, segment
from context.evidence_block_lineage import parse_blocks

RUN = "c:/SCP/outputs/level1_final_23/tasks"
OUT = "c:/SCP/outputs/block_role_annotation"

#: Coverage below this means the compressed text is not simply a reduction of
#: the raw text, and offset-based child lineage would be reporting fiction.
ALIGNABLE_COVERAGE = 0.90


def freeze_segmenter() -> dict:
    source = inspect.getsource(segment)
    return {
        "segment_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "child_target_chars": CHILD_TARGET_CHARS,
        "frozen_before": "alignment probe",
    }


def classify(raw: str, compressed: str) -> dict:
    """Name the shape of the removal, from line matches only."""

    raw_lines = raw.split("\n")
    compressed_lines = compressed.split("\n")
    matcher = difflib.SequenceMatcher(None, raw_lines, compressed_lines, autojunk=False)
    runs = [b for b in matcher.get_matching_blocks() if b.size]
    matched_lines = sum(b.size for b in runs)
    coverage = matched_lines / max(len(compressed_lines), 1)

    if coverage < ALIGNABLE_COVERAGE:
        shape = "transformed_or_unalignable"
    elif len(runs) == 1 and runs[0].a == 0 and runs[0].b == 0:
        shape = "prefix_truncation"
    elif all(runs[i].a < runs[i + 1].a for i in range(len(runs) - 1)):
        shape = "ordered_subsequence"
    else:
        shape = "discontinuous_selection"

    return {
        "shape": shape,
        "raw_lines": len(raw_lines),
        "compressed_lines": len(compressed_lines),
        "matching_runs": len(runs),
        "matched_line_coverage": round(coverage, 4),
        "first_run_starts_at_line": runs[0].a if runs else None,
        "last_matched_raw_line": max((b.a + b.size for b in runs), default=None),
        "runs": [
            {"raw_line": b.a, "compressed_line": b.b, "lines": b.size} for b in runs[:12]
        ],
    }


def main() -> None:
    frozen = freeze_segmenter()
    with open(f"{OUT}/segmenter_freeze.json", "w", encoding="utf-8") as handle:
        json.dump(frozen, handle, ensure_ascii=False, indent=1)
    print(f"segmenter 已凍結：sha256 {frozen['segment_source_sha256'][:16]}"
          f"、child_target_chars {frozen['child_target_chars']}")
    print(f"   凍結時機：{frozen['frozen_before']}\n")

    from context.stage1_context import Stage1ContextBuilder

    builder = Stage1ContextBuilder()
    reports = []
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
        trimmed = builder._compress_multiline_text(
            summary,
            max_lines=builder.config.max_context_lines,
            max_chars=builder.config.max_context_chars,
        ) or ""
        raw_blocks = {b.block_id: b.text for b in parse_blocks(summary)}
        out_blocks = {b.block_id: b.text for b in parse_blocks(trimmed)}
        for block_id, text in raw_blocks.items():
            after = out_blocks.get(block_id)
            if after is None or after == text:
                continue
            report = classify(text, after)
            report.update(
                task_id=os.path.basename(path)[:3],
                block_id=block_id,
                raw_chars=len(text),
                compressed_chars=len(after),
                segments_if_split=len(segment(text)),
            )
            reports.append(report)

    print(f"=== 被壓縮器改動的 block {len(reports)} 個")
    for report in sorted(reports, key=lambda r: -r["raw_chars"]):
        print(f"   {report['task_id']} {report['block_id']}: "
              f"{report['raw_chars']} → {report['compressed_chars']} 字"
              f"　shape={report['shape']}")
        print(f"        比對行覆蓋 {report['matched_line_coverage']}"
              f"、連續區段 {report['matching_runs']}"
              f"、首段起於 raw 第 {report['first_run_starts_at_line']} 行"
              f"、最後對上 raw 第 {report['last_matched_raw_line']} 行")
        print(f"        依凍結切分器會切成 {report['segments_if_split']} 個 child")

    with open(f"{OUT}/compression_alignment.json", "w", encoding="utf-8") as handle:
        json.dump({"segmenter": frozen, "blocks": reports}, handle,
                  ensure_ascii=False, indent=1)
    print(f"\n   -> {OUT}/compression_alignment.json")
    print("   shape 只描述移除的形狀，不判斷移除的內容是否重要；")
    print("   `cross_segment_structure` 不由 shape 決定，仍由標註者判斷。")


if __name__ == "__main__":
    sys.exit(main())
