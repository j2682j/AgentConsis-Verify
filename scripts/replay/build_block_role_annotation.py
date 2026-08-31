"""A blind annotation set for evidence block roles, at protocol version 2.

Version 1 had three defects and none of them were visible from its summary.

It reported six blocks lost before the budget and missed five more that survived
with their bodies cut short -- a block that keeps its marker and loses the
sentence carrying the relation is not a block that was kept, and calling it one
would hide the compressor's most interesting failure. Both versions of those
five are annotated separately, because whether the trim changed the role is the
question.

It advertised a question contract it did not have: `answer_requirement` and
`answer_target` were empty on all 231 rows, because the historical records never
stored them. Rather than fill them from the automatic classifier -- which puts a
surname requirement under `answer_role=title` on task 029, and would feed its own
error into the labels meant to check it -- the fields are gone and the
conditioning is named for what it is: the full question.

And it leaked. The renderer's own wrappers rode along inside the block text --
`Unverified References: ... NOT verified answer support`, `Candidate answer` --
which tell an annotator what the system already concluded about trust, on the
very judgement being asked for. Those are stripped; the source title, the
content, the line breaks and the indentation are not, because `STRUCTURAL_INPUT`
is a judgement about exactly those.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, r"c:/SCP")

from context.evidence_block_lineage import BLOCK_MARKER, digest, parse_blocks

RUN = "c:/SCP/outputs/level1_final_23/tasks"
AUDIT = "c:/SCP/outputs/prospective_render_audit/render_audit.jsonl"
OUT = "c:/SCP/outputs/block_role_annotation"

ANNOTATION_PROTOCOL_VERSION = 2

#: Named honestly. There is no contract to condition on, so the annotator reads
#: the whole question, which is enough to judge a relation and is what version 1
#: was actually doing while claiming otherwise.
CONDITIONING = "full_question"

#: Decided now, not after the labels arrive. A parent that was split has no text
#: of its own, so its role is the union of its children's -- any child carrying
#: direct support makes the parent a carrier of direct support. Choosing this
#: afterwards would mean choosing whichever rule produced the tidier result.
ROLE_AGGREGATION = {
    "container": "union_children",
    "leaf": "self",
}

ROLES = (
    "DIRECT_SUPPORT", "BRIDGE_SUPPORT", "DERIVATION_INPUT", "STRUCTURAL_INPUT",
    "MENTION_ONLY", "IRRELEVANT", "UNCLEAR",
)

#: Wrappers the renderer adds around retrieved text. They state the system's own
#: trust verdict, which is the thing the annotation is supposed to check.
WRAPPERS = (
    re.compile(r"^\s*Grounded Evidence:\s*$", re.MULTILINE),
    re.compile(r"^\s*Unverified References:\s*$", re.MULTILINE),
    re.compile(r"^.*NOT verified answer support.*$", re.MULTILINE),
    # Not anchored to a line start: it turns up mid-paragraph, inside an earlier
    # agent's reasoning that found its way into the retrieved text.
    re.compile(r"Candidate answer[^\n]*", re.IGNORECASE),
    re.compile(r"^\s*Answer Requirement:[^\n]*$", re.MULTILINE),
    re.compile(r"^\s*Answer Target:[^\n]*$", re.MULTILINE),
)

#: The pipeline's own `Evidence:` label. Only the label goes; whatever follows it
#: on the same line is retrieved content and stays, so the whole line is never
#: deleted.
EVIDENCE_LABEL = re.compile(r"^[ \t]*Evidence:[ \t]*", re.MULTILINE)


def neutralise(text: str) -> str:
    """Strip the renderer's trust labels, keep the source text byte for byte."""

    body = BLOCK_MARKER.sub("", text, count=1)
    for pattern in WRAPPERS:
        body = pattern.sub("", body)
    body = EVIDENCE_LABEL.sub("", body)
    return body.strip("\n")


def normalise_format(text: str) -> str:
    """Line endings unified and edge blank lines removed, nothing else.

    Internal blank lines, indentation and alignment stay exactly as they are:
    they are what `STRUCTURAL_INPUT` is a judgement about, and collapsing them
    to compare two versions would erase the difference being looked for. Three
    of the five blocks the compressor "truncated" differ by one edge newline and
    one character; treating those as content change would put identical text in
    front of an annotator twice and inflate agreement for free.
    """

    unified = str(text or "").replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
    rows = unified.split(chr(10))
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    return chr(10).join(rows)


#: Long enough to carry a paragraph's argument, short enough to read. Chosen for
#: legibility, not from the content: nothing about the question, no embedding and
#: no relevance ordering decides where a cut lands.
CHILD_TARGET_CHARS = 1800


def segment(text: str) -> list[tuple[int, int]]:
    """Split a long block on its own boundaries, keeping offsets.

    A 68,619-character block cannot be annotated whole, and showing its first few
    thousand characters would hide whatever support sits later while also
    destroying the structure judgement. So it is cut on the boundaries the text
    already has -- blank lines, then single line breaks, then sentences only for
    a paragraph too long to fit -- and each piece keeps its offset into the
    original, so what the compressor removed can be located afterwards.
    """

    body = str(text or "")
    if len(body) <= CHILD_TARGET_CHARS:
        return [(0, len(body))]

    pieces: list[tuple[int, int]] = []
    blank_line = re.compile(chr(10) + r"\s*" + chr(10))
    boundaries = [m.end() for m in blank_line.finditer(body)]
    if not boundaries:
        boundaries = [m.end() for m in re.finditer(chr(10), body)]
    boundaries = [b for b in boundaries if b > 0] + [len(body)]
    start = 0
    for boundary in boundaries:
        if boundary - start >= CHILD_TARGET_CHARS:
            pieces.append((start, boundary))
            start = boundary
    if start < len(body):
        pieces.append((start, len(body)))

    # A single paragraph longer than the target falls back to sentence ends.
    refined: list[tuple[int, int]] = []
    for begin, end in pieces:
        if end - begin <= CHILD_TARGET_CHARS * 2:
            refined.append((begin, end))
            continue
        cut = begin
        for match in re.finditer(r"(?<=[.!?])\s+", body[begin:end]):
            position = begin + match.end()
            if position - cut >= CHILD_TARGET_CHARS:
                refined.append((cut, position))
                cut = position
        if cut < end:
            refined.append((cut, end))

    # No line boundaries at all: one block is 8,868 characters on a single line,
    # so the paragraph and line rules both find nothing. Sentence ends first, and
    # a fixed-width cut only when the text has none of those either -- a bad cut
    # still beats handing over something nobody reads to the end of.
    if len(refined) <= 1 and len(body) > CHILD_TARGET_CHARS:
        cuts = [m.end() for m in re.finditer(r"(?<=[.!?])\s+", body)]
        rebuilt, cursor = [], 0
        for cut in cuts + [len(body)]:
            if cut - cursor >= CHILD_TARGET_CHARS:
                rebuilt.append((cursor, cut))
                cursor = cut
        if cursor < len(body):
            rebuilt.append((cursor, len(body)))
        refined = rebuilt or [
            (start, min(start + CHILD_TARGET_CHARS, len(body)))
            for start in range(0, len(body), CHILD_TARGET_CHARS)
        ]
    return refined or [(0, len(body))]


def compressed_blocks(builder, search_result: str) -> dict[str, str]:
    """The blocks as the line compressor leaves them, recomputed.

    The audit stored each block's post-compression hash and not its text, so
    version 1 wrote the raw text under both variants: five pairs that were meant
    to show what the trim removed were byte-identical. The compressor is
    deterministic, so the text is recoverable by running it again.
    """

    trimmed = builder._compress_multiline_text(
        search_result,
        max_lines=builder.config.max_context_lines,
        max_chars=builder.config.max_context_chars,
    )
    return {b.block_id: b.text for b in parse_blocks(trimmed or "")}


def search_raw(record: dict) -> dict:
    meta = (record.get("network_summary") or {}).get("metadata") or {}
    for item in meta.get("tool_usage") or []:
        if item.get("tool_name") == "search" and isinstance(item.get("raw_result"), dict):
            return item["raw_result"]
    return {}


def audit_stages() -> dict[str, dict]:
    """One representative render per task; all nine agree by measurement."""

    out: dict[str, dict] = {}
    for line in open(AUDIT, encoding="utf-8"):
        row = json.loads(line)
        out.setdefault(row["task_id"], row)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    from context.stage1_context import Stage1ContextBuilder

    stages = audit_stages()
    builder = Stage1ContextBuilder()
    units: list[dict] = []

    for path in sorted(glob.glob(f"{RUN}/*.json")):
        record = json.load(open(path, encoding="utf-8"))
        raw = search_raw(record)
        if not raw:
            continue
        task = os.path.basename(path)[:3]
        question = str(record.get("question") or "")
        stage = stages.get(task) or {}
        first = {
            b["block_id"]: b for b in (stage.get("raw_to_compressed", {}).get("blocks") or [])
        }
        second = {
            b["block_id"]: b for b in (stage.get("compressed_to_rendered", {}).get("blocks") or [])
        }
        summary = str(raw.get("summary") or "")
        raw_blocks = {b.block_id: b for b in parse_blocks(summary)}
        after_compressor = compressed_blocks(builder, summary)

        for block_id, block in raw_blocks.items():
            first_stage = (first.get(block_id) or {}).get("disposition", "unknown")
            # A block the compressor removed never met the budget. Saying
            # `unknown` implied the record was incomplete; `not_reached` says
            # what happened.
            second_stage = (
                (second.get(block_id) or {}).get("disposition", "unknown")
                if first_stage != "dropped" else "not_reached"
            )
            final = (
                "dropped_by_compressor" if first_stage == "dropped"
                else "dropped_by_budget" if second_stage == "dropped"
                else "kept"
            )
            body = neutralise(block.text)
            # Long blocks become children on the frozen segmenter's boundaries.
            # A 68,000-character unit cannot be read carefully, and showing only
            # its opening would hide whatever sits later while destroying the
            # structure judgement. Children are annotation units only: the drop
            # rate is still computed over the 231 parents.
            pieces = segment(body)
            if len(pieces) > 1:
                for index, (begin, end) in enumerate(pieces, 1):
                    units.append({
                        "task_id": task, "block_id": block_id, "variant": "child",
                        "question": question, "block_text": body[begin:end],
                        "raw_block_hash": block.text_hash,
                        "compressed_block_hash": "",
                        "child_index": index, "child_of": block.text_hash,
                        "child_start": begin, "child_end": end,
                        "raw_to_compressed_disposition": first_stage,
                        "compressed_to_rendered_disposition": second_stage,
                        "final_disposition": final,
                        "provenance": "web_retrieval",
                    })
            units.append({
                "task_id": task, "block_id": block_id, "variant": "raw",
                "question": question,
                "block_text": "" if len(pieces) > 1 else body,
                "container_only": len(pieces) > 1,
                "raw_block_hash": block.text_hash, "compressed_block_hash": "",
                "raw_to_compressed_disposition": first_stage,
                "compressed_to_rendered_disposition": second_stage,
                "final_disposition": final,
                "provenance": "web_retrieval",
            })
            # A compressor "truncation" that only moved an edge newline is not
            # content change. Three of the five differ by a single blank line,
            # and annotating both copies would put identical text in front of a
            # reader twice while inflating agreement for nothing.
            material = first_stage == "truncated" and normalise_format(
                neutralise(block.text)
            ) != normalise_format(neutralise(after_compressor.get(block_id, "")))
            if first_stage == "truncated" and not material:
                units[-1]["format_only_change"] = True
            if material:
                trimmed_body = neutralise(after_compressor.get(block_id, ""))
                trimmed_pieces = segment(trimmed_body)
                # The compressed form of a long block is still long: the one at
                # 68,624 characters comes out at 8,869, which is no more
                # readable in one sitting than the original was.
                for index, (begin, end) in enumerate(trimmed_pieces, 1):
                    if len(trimmed_pieces) == 1:
                        break
                    units.append({
                        "task_id": task, "block_id": block_id,
                        "variant": "compressed_child",
                        "question": question,
                        "block_text": trimmed_body[begin:end],
                        "raw_block_hash": block.text_hash,
                        "compressed_block_hash": (first.get(block_id) or {}).get(
                            "rendered_text_hash", ""
                        ),
                        "child_index": index,
                        "raw_to_compressed_disposition": first_stage,
                        "compressed_to_rendered_disposition": second_stage,
                        "final_disposition": final,
                        "provenance": "web_retrieval",
                    })
                units.append({
                    "task_id": task, "block_id": block_id, "variant": "compressed",
                    "question": question,
                    "block_text": "" if len(trimmed_pieces) > 1 else trimmed_body,
                    "container_only": len(trimmed_pieces) > 1,
                    "raw_block_hash": block.text_hash,
                    "compressed_block_hash": (first.get(block_id) or {}).get(
                        "rendered_text_hash", ""
                    ),
                    "raw_to_compressed_disposition": first_stage,
                    "compressed_to_rendered_disposition": second_stage,
                    "final_disposition": final,
                    "provenance": "web_retrieval",
                })

    # Deterministic shuffle on a salted identity hash, so the two variants of one
    # block land far apart and nothing in the order encodes task or disposition.
    units.sort(key=lambda u: hashlib.sha256(
        f"v2:{u['task_id']}:{u['block_id']}:{u['variant']}".encode()
    ).hexdigest())
    for index, unit in enumerate(units, 1):
        unit["annotation_id"] = f"B{index:03d}"

    blind_path = f"{OUT}/block_role_blind.jsonl"
    with open(blind_path, "w", encoding="utf-8") as handle:
        for unit in units:
            # A split parent is a container: no text, nothing to judge. Leaving
            # it in the file put four blank rows in front of an annotator with
            # no way to know they should be skipped, and a blank row invites an
            # `IRRELEVANT` or `UNCLEAR` label that would then be counted.
            if unit.get("container_only"):
                continue
            handle.write(json.dumps({
                # Identity lives in the key file. The blind file carries an id,
                # the question, the text and the empty label fields.
                "annotation_id": unit["annotation_id"],
                "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
                "conditioning": CONDITIONING,
                "question": unit["question"],
                "provenance": unit["provenance"],
                "block_text": unit["block_text"],
                # One label set, read differently per population by the key.
                # Annotating raw and post-compressor separately would have meant
                # 225 identical rows: the compressor leaves all but two blocks
                # byte-identical, so a second pass adds fatigue and a free
                # agreement score, not information.
                "representation_role": [], "standalone_answerable": "",
                "requires_other_blocks": "", "polarity": "",
                # For segmented blocks: whether the piece depends on structure
                # that spans the cut, and whether judging it needs the whole
                # document rather than this piece.
                "cross_segment_structure": "", "requires_full_document_scope": "",
                "structure_available": "", "usability": "",
                "annotation_confidence": "", "notes": "",
            }, ensure_ascii=False) + "\n")

    key_path = f"{OUT}/_disposition_key.jsonl"
    with open(key_path, "w", encoding="utf-8") as handle:
        for unit in units:
            parent = unit["variant"] == "raw"
            handle.write(json.dumps({
                "annotation_id": unit["annotation_id"],
                "task_id": unit["task_id"],
                "block_id": unit["block_id"],
                "variant": unit["variant"],
                "raw_block_hash": unit.get("raw_block_hash", ""),
                "compressed_block_hash": unit.get("compressed_block_hash", ""),
                "raw_to_compressed_disposition": unit["raw_to_compressed_disposition"],
                "compressed_to_rendered_disposition": unit["compressed_to_rendered_disposition"],
                "final_disposition": unit["final_disposition"],
                # Which population a row belongs to. Every rate is computed over
                # one of these and never over a mixture: the compressed variants
                # and the children are units this script created, not blocks the
                # system produced, and counting them would let the instrument
                # inflate the thing it measures.
                "system_parent": parent,
                "representation": "raw" if parent else unit["variant"],
                "source_role_eligible": parent,
                "post_compressor_eligible":
                    parent and unit["raw_to_compressed_disposition"] != "dropped",
                "rendered_to_agent": parent and unit["final_disposition"] == "kept",
                "statistical_cluster_id": f"{unit['task_id']}:{unit['block_id']}",
                "include_in_system_drop_rate": parent,
                "format_only_change": bool(unit.get("format_only_change")),
                "container_only": bool(unit.get("container_only")),
                "child_index": unit.get("child_index"),
                "annotation_required": not bool(unit.get("container_only")),
                "role_aggregation": ROLE_AGGREGATION[
                    "container" if unit.get("container_only") else "leaf"
                ],
                "label_source": (
                    "derived_from_children" if unit.get("container_only") else "direct"
                ),
            }, ensure_ascii=False) + "\n")

    with open(f"{OUT}/annotation_schema.json", "w", encoding="utf-8") as handle:
        json.dump({
            "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
            "conditioning": CONDITIONING,
            "conditioning_note": "沒有 answer_requirement／answer_target；歷史紀錄未保存，"
                                 "且不以自動 QuestionRole 填充（其本身可能錯）",
            "roles": list(ROLES), "multi_label": True, "unclear_is_exclusive": True,
            "standalone_answerable": ["yes", "no"],
            "requires_other_blocks": ["yes", "no"],
            "polarity": ["supports", "contradicts", "neutral"],
            "annotation_confidence": ["high", "medium", "low"],
            "structure_available": {
                "values": ["yes", "partial", "no", "not_required", "unclear"],
                "criterion": "回答本題所需的排版關係（縮排、欄位對齊、順序、"
                             "表格結構）在這段文字中是否仍可判讀",
            },
            "usability": {
                "values": ["full", "partial", "unusable", "unclear"],
                "criterion": "只看這段文字，能否支持它所承載的角色；"
                             "被截斷而失去關鍵子句者為 partial",
            },
            "cross_segment_structure": {
                "values": ["yes", "no", "not_applicable"],
                "criterion": "理解這一段是否需要相鄰段落的結構或上下文；"
                             "未切分的 block 填 not_applicable",
            },
            "requires_full_document_scope": {
                "values": ["yes", "no", "unclear"],
                "criterion": "判斷是否需要整份文件範圍，例如否定式驗證"
                             "（『全文中沒有出現 X』）",
            },
            "hidden_from_annotator": [
                "task_id", "block_id", "block hash", "variant (raw/compressed)",
                "both stage dispositions", "final_22/final_23 outcome",
                "agent answers", "automatic role", "retrieval rank",
                "renderer trust wrappers",
            ],
        }, handle, ensure_ascii=False, indent=1)

    # Two populations, never added together. The parent blocks are what the
    # system produced and what a drop rate is about; the variants are annotation
    # units this script created, and counting them into a system rate would let
    # a measurement instrument inflate the thing it measures.
    parents = [u for u in units if u["variant"] == "raw"]
    final = Counter(u["final_disposition"] for u in parents)
    stage_one = Counter(u["raw_to_compressed_disposition"] for u in parents)
    print(f"protocol v{ANNOTATION_PROTOCOL_VERSION}　conditioning={CONDITIONING}")
    labelable = [u for u in units if not u.get("container_only")]
    print(f"   可標註列 {len(labelable)}"
          f"（raw 未切 {sum(1 for u in labelable if u['variant']=='raw')}"
          f"、compressed 未切 {sum(1 for u in labelable if u['variant']=='compressed')}"
          f"、raw children {sum(1 for u in labelable if u['variant']=='child')}"
          f"、compressed children {sum(1 for u in labelable if u['variant']=='compressed_child')}）")
    print(f"   container（不標，由 children 聯集推導）"
          f" {sum(1 for u in units if u.get('container_only'))}")
    print(f"   parent system blocks {len(parents)}"
          f"　final: {dict(final)}")
    print(f"   annotation units {len(units)}"
          f"（{len(parents)} parent + "
          f"{sum(1 for u in units if u['variant']=='compressed')} material variant"
          f" + {sum(1 for u in units if u['variant']=='child')} child）")
    print(f"   format_only_change（未建 variant）: "
          f"{sum(1 for u in parents if u.get('format_only_change'))}")
    print(f"   sha256 {hashlib.sha256(open(blind_path,'rb').read()).hexdigest()[:16]}")
    print(f"   raw→compressed: {dict(stage_one)}")
    print(f"   final_disposition: {dict(final)}")

    blind = [json.loads(l) for l in open(blind_path, encoding="utf-8")]
    leaks = {
        term: sum(1 for b in blind if term.lower() in b["block_text"].lower())
        for term in ("NOT verified", "Candidate answer", "Unverified References",
                     "Grounded Evidence")
    }
    print(f"   wrapper 洩漏檢查: {leaks}")
    print(f"   blind 欄位: {sorted(blind[0])}")


if __name__ == "__main__":
    sys.exit(main())
