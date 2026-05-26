from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def clean_answer(value: Any) -> str:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? clean_answer 瘚?嚗??怎垢?喳????皜??澆?銝西??敺?瘚??臭蝙?函??批捆??
    
    Args:
        value: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def attachment_type(sample: dict[str, Any]) -> str:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? attachment_type 瘚?嚗???evaluation.benchmarks.gaia.gaia_analysis_report ??蝔?瘙???attachment_type 撠?????????雿?蝯??Ｙ???
    
    Args:
        sample: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    file_path = str(sample.get("file_name") or sample.get("file_path") or "").strip()
    if not file_path:
        return "-"
    return Path(file_path).suffix.lower() or "(unknown)"


def stage2_tool_names(stage2_outputs: list[dict[str, Any]]) -> list[str]:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? stage2_tool_names 瘚?嚗???evaluation.benchmarks.gaia.gaia_analysis_report ??蝔?瘙???stage2_tool_names 撠?????????雿?蝯??Ｙ???
    
    Args:
        stage2_outputs: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[str]??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    names = set()
    for output in stage2_outputs or []:
        for tool in output.get("tool_usage", []) or []:
            if isinstance(tool, dict) and tool.get("tool_name"):
                names.add(str(tool["tool_name"]))
    return sorted(names)


def graph_memory_reads_for_task(runtime: Any, task_id: str) -> list[dict[str, Any]]:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? graph_memory_reads_for_task 瘚?嚗???evaluation.benchmarks.gaia.gaia_analysis_report ??蝔?瘙???graph_memory_reads_for_task 撠?????????雿?蝯??Ｙ???
    
    Args:
        runtime: ?桀?瘚?????銝???????鞈???
        task_id: 閮蝟餌絞???炎蝝Ｙ??神?亥?????隞??
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[dict[str, Any]]??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    reads = []
    for item in list(getattr(runtime, "shared_memory_reads", []) or []):
        if not isinstance(item, dict) or item.get("source") != "graph_memory":
            continue
        if task_id and str(item.get("task_id", "")) != str(task_id):
            continue
        seed_hits = list(item.get("seed_task_hits", []) or [])
        expanded_hits = list(item.get("expanded_task_hits", []) or [])
        related_task_ids = [str(value) for value in item.get("related_task_ids", []) or [] if str(value)]
        insight_ids = [str(value) for value in item.get("insight_ids", []) or [] if str(value)]
        suppressed_task_ids = [str(value) for value in item.get("suppressed_task_ids", []) or [] if str(value)]
        reads.append(
            {
                "stage": item.get("stage", ""),
                "task_type": item.get("task_type", ""),
                "trigger_terms": item.get("trigger_terms", []),
                "related_task_ids": related_task_ids,
                "suppressed_task_ids": suppressed_task_ids,
                "insight_ids": insight_ids,
                "seed_task_hits": seed_hits,
                "expanded_task_hits": expanded_hits,
                "qdrant_hit_count": len(seed_hits),
                "expanded_hit_count": len(expanded_hits),
                "has_retrieval_hit": bool(related_task_ids or insight_ids or seed_hits or expanded_hits),
                "insight_hit": bool(insight_ids),
                "task_memory_hit": bool(related_task_ids or seed_hits or expanded_hits),
                "task_memory_suppressed": bool(suppressed_task_ids),
            }
        )
    return reads


def build_sample_analysis_record(
    *,
    sample_index: int,
    sample: dict[str, Any],
    network: Any,
    sample_result: dict[str, Any],
    evaluator: Any,
) -> dict[str, Any]:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? build_sample_analysis_record 瘚?嚗遣蝡?蝥?蝔?閬??拐辣????瑽?頛詨?憛?
    
    Args:
        sample_index: 甇斗?蝔?閬蝙?函?頛詨鞈???
        sample: 甇斗?蝔?閬蝙?函?頛詨鞈???
        network: 甇斗?蝔?閬蝙?函?頛詨鞈???
        sample_result: 閰摯???撌亙?瑁?敺??蝯????貉???
        evaluator: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 dict[str, Any]??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    expected = clean_answer(sample.get("final_answer", ""))
    stage1_result = clean_answer(getattr(network, "last_stage1_result", ""))
    final_decision = getattr(network, "last_final_decision", None) or {}
    final_result = clean_answer(final_decision.get("final_result", stage1_result))
    predicted = clean_answer(sample_result.get("predicted", ""))
    stage2_outputs = getattr(network, "last_stage2_outputs", []) or []
    runtime = getattr(network, "runtime", None)
    task_id = str(sample.get("task_id", "") or "")
    token_usage = runtime.token_usage_summary() if runtime is not None else {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "by_stage": {},
        "by_model": {},
        "records": [],
    }
    graph_reads = graph_memory_reads_for_task(runtime, task_id)
    related_task_ids = sorted(
        {
            task_id
            for read in graph_reads
            for task_id in read.get("related_task_ids", [])
            if task_id
        }
    )
    insight_ids = sorted(
        {
            insight_id
            for read in graph_reads
            for insight_id in read.get("insight_ids", [])
            if insight_id
        }
    )
    qdrant_hit_count = sum(int(read.get("qdrant_hit_count", 0) or 0) for read in graph_reads)
    expanded_hit_count = sum(int(read.get("expanded_hit_count", 0) or 0) for read in graph_reads)
    suppressed_task_ids = sorted(
        {
            task_id
            for read in graph_reads
            for task_id in read.get("suppressed_task_ids", [])
            if task_id
        }
    )
    graph_memory_insight_hit = bool(insight_ids)
    graph_memory_task_memory_hit = bool(related_task_ids or qdrant_hit_count or expanded_hit_count)
    graph_memory_task_memory_suppressed = bool(suppressed_task_ids)
    graph_memory_retrieval_hit = bool(graph_memory_insight_hit or graph_memory_task_memory_hit)
    shared_attachment_bundle = getattr(runtime, "shared_attachment_bundle", None) if runtime is not None else None
    shared_search_bundle = getattr(runtime, "shared_stage2_search_bundle", None) if runtime is not None else None

    stage2_answers = []
    for item in stage2_outputs:
        stage2_answers.append(
            {
                "agent_idx": item.get("agent_idx"),
                "model_name": item.get("model_name"),
                "answer": clean_answer(item.get("answer", "")),
                "success": bool(item.get("success", False)),
                "tools": [
                    tool.get("tool_name")
                    for tool in item.get("tool_usage", []) or []
                    if isinstance(tool, dict) and tool.get("tool_name")
                ],
                "routing": item.get("routing", {}),
                "error": item.get("error"),
            }
        )

    return {
        "sample_index": sample_index,
        "task_id": task_id,
        "attachment_type": attachment_type(sample),
        "expected": expected,
        "stage1_result": stage1_result,
        "stage1_exact": evaluator._check_exact_match(stage1_result, expected),
        "final_result": final_result,
        "predicted": predicted,
        "final_exact": bool(sample_result.get("exact_match", False)),
        "final_partial": bool(sample_result.get("partial_match", False)),
        "score": sample_result.get("score"),
        "selected_agent_idx": final_decision.get("selected_agent_idx"),
        "stage2_answers": stage2_answers,
        "stage2_tool_names": stage2_tool_names(stage2_outputs),
        "token_usage": token_usage,
        "llm_call_count": token_usage.get("calls", 0),
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "graph_memory_read_count": len(graph_reads),
        "graph_memory_retrieval_hit": graph_memory_retrieval_hit,
        "graph_memory_insight_hit": graph_memory_insight_hit,
        "graph_memory_task_memory_hit": graph_memory_task_memory_hit,
        "graph_memory_task_memory_suppressed": graph_memory_task_memory_suppressed,
        "graph_memory_related_task_ids": related_task_ids,
        "graph_memory_suppressed_task_ids": suppressed_task_ids,
        "graph_memory_insight_ids": insight_ids,
        "graph_memory_qdrant_hit_count": qdrant_hit_count,
        "graph_memory_expanded_hit_count": expanded_hit_count,
        "graph_memory_reads": graph_reads,
        "shared_attachment_used": bool(
            isinstance(shared_attachment_bundle, dict) and shared_attachment_bundle.get("used")
        ),
        "shared_attachment_reader": (
            (shared_attachment_bundle.get("metadata") or {}).get("reader", "")
            if isinstance(shared_attachment_bundle, dict)
            else ""
        ),
        "shared_search_enabled": bool(
            isinstance(shared_search_bundle, dict) and shared_search_bundle.get("enabled")
        ),
        "question_excerpt": " ".join(str(sample.get("question", "") or "").split())[:240],
    }


def status_text(record: dict[str, Any]) -> str:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? status_text 瘚?嚗???evaluation.benchmarks.gaia.gaia_analysis_report ??蝔?瘙???status_text 撠?????????雿?蝯??Ｙ???
    
    Args:
        record: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    if record["final_exact"]:
        return "exact"
    if record["final_partial"]:
        return "partial"
    return "wrong"


def write_markdown_table(handle, headers: list[str], rows: list[list[Any]]) -> None:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? write_markdown_table 瘚?嚗??桀???蝯??身摰???神?交?摰摮?蝵柴?
    
    Args:
        handle: 甇斗?蝔?閬蝙?函?頛詨鞈???
        headers: 甇斗?蝔?閬蝙?函?頛詨鞈???
        rows: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 None??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in rows:
        clean_row = [str(cell).replace("\n", " ").replace("|", "\\|") for cell in row]
        handle.write("| " + " | ".join(clean_row) + " |\n")
    handle.write("\n")


def infer_wrong_common_features(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? infer_wrong_common_features 瘚?嚗???evaluation.benchmarks.gaia.gaia_analysis_report ??蝔?瘙???infer_wrong_common_features 撠?????????雿?蝯??Ｙ???
    
    Args:
        records: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[dict[str, Any]]??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    wrong = [record for record in records if not record["final_exact"]]
    total_wrong = len(wrong)
    if total_wrong == 0:
        return []

    def count_if(predicate):
        """
        鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? count_if 瘚?嚗???evaluation.benchmarks.gaia.gaia_analysis_report ??蝔?瘙???count_if 撠?????????雿?蝯??Ｙ???
        
        Args:
            predicate: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 ?芣?閮颯?
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return sum(1 for record in wrong if predicate(record))

    features = [
        ("stage1_wrong", count_if(lambda r: not r["stage1_exact"]), "Stage1 already produced a wrong answer."),
        (
            "graph_memory_task_memory_hit",
            count_if(lambda r: r["graph_memory_task_memory_hit"]),
            "GraphMemory retrieved a prior task, Qdrant seed, or graph-expanded task.",
        ),
        (
            "graph_memory_insight_hit",
            count_if(lambda r: r["graph_memory_insight_hit"]),
            "GraphMemory returned a global insight or strategy rule.",
        ),
        (
            "graph_memory_qdrant_hit",
            count_if(lambda r: int(r.get("graph_memory_qdrant_hit_count", 0) or 0) > 0),
            "GraphMemory seed retrieval returned Qdrant hits.",
        ),
        (
            "search_used",
            count_if(lambda r: r["shared_search_enabled"] or "search" in r["stage2_tool_names"]),
            "Search evidence was used or shared search was enabled.",
        ),
        ("attachment_task", count_if(lambda r: r["attachment_type"] != "-"), "The task had an attachment."),
        ("attachment_used", count_if(lambda r: r["shared_attachment_used"]), "Shared attachment evidence was available."),
        ("final_changed_stage1", count_if(lambda r: r["final_result"] != r["stage1_result"]), "Final decision changed the stage1 answer."),
        (
            "stage1_correct_but_final_wrong",
            count_if(lambda r: r["stage1_exact"] and not r["final_exact"]),
            "Stage1 was correct, but final answer became wrong.",
        ),
    ]
    return [
        {"feature": name, "count": count, "rate": count / total_wrong, "note": note}
        for name, count, note in features
    ]


def write_gaia_analysis_report(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    run_metadata: dict[str, Any],
) -> Path:
    """
    鞎痊?瑁? evaluation.benchmarks.gaia.gaia_analysis_report 銝剔? write_gaia_analysis_report 瘚?嚗??桀???蝯??身摰???神?交?摰摮?蝵柴?
    
    Args:
        path: 閬???撖怠??獢??桅?頝臬???
        records: 甇斗?蝔?閬蝙?函?頛詨鞈???
        run_metadata: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Path??
    
    ??雿:
        ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
    """
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage2_pulled_back = [record for record in records if not record["stage1_exact"] and record["final_exact"]]
    stage2_broke_correct = [record for record in records if record["stage1_exact"] and not record["final_exact"]]
    graph_memory_wrong = [record for record in records if record["graph_memory_task_memory_hit"] and not record["final_exact"]]
    insight_only_wrong = [
        record
        for record in records
        if record["graph_memory_insight_hit"] and not record["graph_memory_task_memory_hit"] and not record["final_exact"]
    ]
    exact_count = sum(1 for record in records if record["final_exact"])
    partial_count = sum(1 for record in records if record["final_partial"])
    prompt_tokens = sum(int(record.get("prompt_tokens", 0) or 0) for record in records)
    completion_tokens = sum(int(record.get("completion_tokens", 0) or 0) for record in records)
    total_tokens = sum(int(record.get("total_tokens", 0) or 0) for record in records)
    llm_call_count = sum(int(record.get("llm_call_count", 0) or 0) for record in records)
    wrong_features = infer_wrong_common_features(records)

    with output_path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write("# GAIA Run Analysis\n\n")
        handle.write("## Summary\n\n")
        write_markdown_table(
            handle,
            ["metric", "value"],
            [
                ["run_started_at", run_metadata.get("run_started_at", "")],
                ["level", run_metadata.get("level", "")],
                ["total_samples", len(records)],
                ["exact_matches", exact_count],
                ["partial_matches", partial_count],
                ["llm_call_count", llm_call_count],
                ["prompt_tokens", prompt_tokens],
                ["completion_tokens", completion_tokens],
                ["total_tokens", total_tokens],
                ["stage2_pulled_back", len(stage2_pulled_back)],
                ["stage1_correct_but_stage2_wrong", len(stage2_broke_correct)],
                ["graph_memory_task_memory_hit_and_final_wrong", len(graph_memory_wrong)],
                ["graph_memory_insight_only_and_final_wrong", len(insight_only_wrong)],
                ["memory_mode", run_metadata.get("memory_mode", "")],
                ["graph_memory_enabled", run_metadata.get("graph_memory_enabled", "")],
            ],
        )

        handle.write("## 1. Stage2 Pulled Back Wrong Stage1\n\n")
        write_markdown_table(
            handle,
            ["#", "task_id", "attachment", "expected", "stage1_result", "final_result", "selected_agent"],
            [
                [
                    record["sample_index"],
                    record["task_id"],
                    record["attachment_type"],
                    record["expected"],
                    record["stage1_result"],
                    record["final_result"],
                    record["selected_agent_idx"],
                ]
                for record in stage2_pulled_back
            ] or [["(none)", "", "", "", "", "", ""]],
        )

        handle.write("## 2. Stage2 Broke Correct Stage1\n\n")
        write_markdown_table(
            handle,
            ["#", "task_id", "attachment", "expected", "stage1_result", "final_result", "selected_agent"],
            [
                [
                    record["sample_index"],
                    record["task_id"],
                    record["attachment_type"],
                    record["expected"],
                    record["stage1_result"],
                    record["final_result"],
                    record["selected_agent_idx"],
                ]
                for record in stage2_broke_correct
            ] or [["(none)", "", "", "", "", "", ""]],
        )

        handle.write("## 3. GraphMemory Task Memory Hit But Final Wrong\n\n")
        handle.write(
            "This section lists samples where GraphMemory returned prior task IDs, Qdrant seed hits, "
            "or graph-expanded task hits, but the final answer was not exact. This excludes global insight-only hits. "
            "This is a signal for manual review; "
            "it does not prove GraphMemory caused the mistake.\n\n"
        )
        write_markdown_table(
            handle,
            ["#", "task_id", "status", "expected", "predicted", "read_count", "task_memory_hit", "insight_hit", "related_task_ids", "insight_ids", "qdrant_hits"],
            [
                [
                    record["sample_index"],
                    record["task_id"],
                    status_text(record),
                    record["expected"],
                    record["predicted"],
                    record["graph_memory_read_count"],
                    record["graph_memory_task_memory_hit"],
                    record["graph_memory_insight_hit"],
                    ", ".join(record["graph_memory_related_task_ids"][:5]),
                    ", ".join(record["graph_memory_insight_ids"][:5]),
                    record["graph_memory_qdrant_hit_count"],
                ]
                for record in graph_memory_wrong
            ] or [["(none)", "", "", "", "", "", "", "", "", "", ""]],
        )

        if graph_memory_wrong:
            handle.write("### GraphMemory Read Details\n\n")
            for record in graph_memory_wrong:
                handle.write(f"#### Sample {record['sample_index']} - {record['task_id']}\n\n")
                for read in record["graph_memory_reads"][:8]:
                    handle.write(
                        "- "
                        f"stage={read['stage']} | "
                        f"task_type={read['task_type']} | "
                        f"related={read['related_task_ids']} | "
                        f"insights={read['insight_ids']} | "
                        f"qdrant_hits={read['qdrant_hit_count']} | "
                        f"expanded_hits={read['expanded_hit_count']}\n"
                    )
                    for hit in read.get("seed_task_hits", [])[:3]:
                        question = str(hit.get("question", "") or "").replace("\n", " ")
                        handle.write(
                            f"  - qdrant_hit task_id={hit.get('task_id')} "
                            f"weight={hit.get('weight')} source={hit.get('source')} "
                            f"question={question[:220]}\n"
                        )
                handle.write("\n")

        handle.write("## 4. Common Wrong-Answer Features\n\n")
        write_markdown_table(
            handle,
            ["feature", "wrong_count", "wrong_rate", "note"],
            [[item["feature"], item["count"], f"{item['rate']:.2%}", item["note"]] for item in wrong_features],
        )

        handle.write("## Per-Sample Overview\n\n")
        write_markdown_table(
            handle,
            [
                "#",
                "status",
                "attachment",
                "graph_reads",
                "task_memory_hit",
                "insight_hit",
                "retrieval_hit",
                "qdrant_hits",
                "tokens",
                "related_tasks",
                "insights",
                "expected",
                "stage1_result",
                "final_result",
                "predicted",
            ],
            [
                [
                    record["sample_index"],
                    status_text(record),
                    record["attachment_type"],
                    record["graph_memory_read_count"],
                    record["graph_memory_task_memory_hit"],
                    record["graph_memory_insight_hit"],
                    record["graph_memory_retrieval_hit"],
                    record["graph_memory_qdrant_hit_count"],
                    record["total_tokens"],
                    ", ".join(record["graph_memory_related_task_ids"][:3]),
                    ", ".join(record["graph_memory_insight_ids"][:3]),
                    record["expected"],
                    record["stage1_result"],
                    record["final_result"],
                    record["predicted"],
                ]
                for record in records
            ],
        )

        handle.write("## Raw JSON\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(records, ensure_ascii=False, indent=2))
        handle.write("\n```\n")

    return output_path

