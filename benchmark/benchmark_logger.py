"""
Benchmark logging ?梁?箏???
甇斗芋蝯蝵?GAIA?FCL 蝑?benchmark ?賣??典??logging ?賢?嚗?UTF-8 full log tee?ompact log ???tdout/stderr ???SON 摰撖怠??token usage ????GraphMemory retrieval ????"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO


class TeeStream:
    """
    鞎痊??stdout ??stderr ??撖怠???stream ??log 瑼???
    Args:
        primary: ???stdout ??stderr stream??        mirror: 閬?甇亙神?亦? log file handle??
    Returns:
        TeeStream 撖虫?嚗?晷蝯?sys.stdout ??sys.stderr??
    ??雿:
        write() ???神?亙??stream嚗?怎垢?閬雿輻摰敺???憪?stdio??    """

    def __init__(self, primary: TextIO, mirror: TextIO):
        """
        鞎痊????TeeStream ?蜓閬撓?箄??∪?頛詨??
        Args:
            primary: ??撓??stream??            mirror: ?∪?撖怠??log stream??
        Returns:
            ?～?
        ??雿:
            銝??亦恣 stdio嚗蝞∪?雿 setup_utf8_log() 鞎痊??        """
        self.primary = primary
        self.mirror = mirror
        self.mirror_enabled = True

    @property
    def encoding(self) -> str:
        """
        鞎痊? primary stream ??encoding??
        Args:
            ?～?
        Returns:
            primary stream ??encoding嚗銝??典?? utf-8??
        ??雿:
            ?芾??惇?改?銝?靽格 stream??        """
        return getattr(self.primary, "encoding", "utf-8")

    def write(self, data: Any) -> int:
        """
        鞎痊????甇亙神??primary ??mirror??
        Args:
            data: 閬神?亦????頧?銝脰???
        Returns:
            撖怠???瑕漲??
        ??雿:
            ???喳神?亙??stream嚗隞颱? stream 撖怠憭望?嚗?憭?敺憭???        """
        if not isinstance(data, str):
            data = str(data)
        self.primary.write(data)
        if self.mirror_enabled and not getattr(self.mirror, "closed", False):
            try:
                self.mirror.write(data)
            except ValueError:
                self.mirror_enabled = False
        return len(data)

    def flush(self) -> None:
        """
        鞎痊 flush primary ??mirror??
        Args:
            ?～?
        Returns:
            ?～?
        ??雿:
            ?孛?澆??stream ??flush??        """
        self.primary.flush()
        if self.mirror_enabled and not getattr(self.mirror, "closed", False):
            try:
                self.mirror.flush()
            except ValueError:
                self.mirror_enabled = False

    def disable_mirror(self) -> None:
        self.mirror_enabled = False

    def isatty(self) -> bool:
        """
        鞎痊? primary ?臬?箔???蝯垢??
        Args:
            ?～?
        Returns:
            ??primary ??TTY ? True嚗????False??
        ??雿:
            ?芾???primary ???        """
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        """
        鞎痊? primary stream ??獢?餈啁泵??
        Args:
            ?～?
        Returns:
            primary.fileno() ????
        ??雿:
            ??primary 銝??fileno嚗????靘???        """
        return self.primary.fileno()

    def __getattr__(self, name: str) -> Any:
        """
        鞎痊??亙惇?批?瘣曄策 primary stream??
        Args:
            name: 撅祆批?蝔晞?
        Returns:
            primary 銝???撅祆扳??寞???
        ??雿:
            ??primary 瘝?閰脣惇?改?????AttributeError??        """
        return getattr(self.primary, name)


class BenchmarkLogger:
    """
    鞎痊?? benchmark logging ??典極?瑟瘜?
    Args:
        ?～?
    Returns:
        BenchmarkLogger 撖虫?嚗??GAIA/BFCL logger 蝜潭??乩誑 static method 雿輻??
    ??雿:
        setup_utf8_log() ????sys.stdout ??sys.stderr??    """

    @staticmethod
    def setup_utf8_log(log_file_path: Path):
        """
        鞎痊撱箇? UTF-8 full log嚗蒂??stdout/stderr tee ??log 瑼?
        Args:
            log_file_path: full log 頛詨頝臬???
        Returns:
            銝?蝯?(log_handle, original_stdout, original_stderr)??
        ??雿:
            ?遣蝡鞈?憭橘??? log 瑼?銝虫耨??sys.stdout/sys.stderr??        """
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file_path.open("w", encoding="utf-8-sig", buffering=1, newline="\n")

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_handle)
        sys.stderr = TeeStream(original_stderr, log_handle)

        return log_handle, original_stdout, original_stderr

    @staticmethod
    def restore_stdio(*, original_stdout, original_stderr) -> None:
        """
        鞎痊?? setup_utf8_log() ?踵??? stdout/stderr??
        Args:
            original_stdout: setup ?? stdout??            original_stderr: setup ?? stderr??
        Returns:
            ?～?
        ??雿:
            ??flush ?桀? stdout/stderr嚗蒂?孵??? stream??        """
        current_stdout = sys.stdout
        current_stderr = sys.stderr
        sys.stdout.flush()
        sys.stderr.flush()
        if isinstance(current_stdout, TeeStream):
            current_stdout.disable_mirror()
        if isinstance(current_stderr, TeeStream):
            current_stderr.disable_mirror()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    @staticmethod
    def open_utf8_text_log(log_file_path: Path):
        """
        鞎痊?? UTF-8-SIG ?? log 瑼?
        Args:
            log_file_path: ?? log 頛詨頝臬???
        Returns:
            撌脤?????瑼?handle??
        ??雿:
            ?遣蝡鞈?憭橘?銝西?撖怠???獢?        """
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        return log_file_path.open("w", encoding="utf-8-sig", buffering=1, newline="\n")

    @staticmethod
    def write_indented_block(handle: Any, text: Any, indent: str = "    ") -> None:
        """
        鞎痊??銵?摮誑?箏?蝮格?撖怠 log??
        Args:
            handle: 撌脤?????瑼?handle??            text: 閬神?亦?????            indent: 瘥?銵??寡????葬??
        Returns:
            ?～?
        ??雿:
            蝛箏摰寞?撖怠 `(none)`??        """
        content = str(text or "").strip()
        if not content:
            handle.write(f"{indent}(none)\n")
            return

        for line in content.splitlines():
            stripped = line.rstrip()
            if stripped:
                handle.write(f"{indent}{stripped}\n")
            else:
                handle.write(f"{indent}\n")

    @staticmethod
    def write_json_line(handle: Any, label: str, value: Any, *, indent: str = "") -> None:
        """
        鞎痊???誑?株? JSON ?澆?撖怠 log??
        Args:
            handle: 撌脤?????瑼?handle??            label: 甈??迂??            value: 閬撓?箇?鞈???            indent: 銵?蝮格???
        Returns:
            ?～?
        ??雿:
            ?⊥? JSON 摨?????fallback ??repr 摮葡??        """
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            encoded = repr(value)
        handle.write(f"{indent}- {label}: {encoded}\n")

    @staticmethod
    def token_usage_summary(network_or_runtime: Any) -> dict[str, Any]:
        """
        鞎痊敺?network ??runtime ?? token usage summary??
        Args:
            network_or_runtime: AgentNetwork?etworkRuntime嚗??瑕? token_usage_summary() ?隞嗚?
        Returns:
            token usage 蝯梯?摮??
        ??雿:
            ?交銝 runtime ??token_usage_summary()嚗???嗅潮?閮剔?瑽?        """
        runtime = getattr(network_or_runtime, "runtime", network_or_runtime)
        if runtime is None or not hasattr(runtime, "token_usage_summary"):
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "by_stage": {},
                "by_model": {},
                "records": [],
            }
        return runtime.token_usage_summary()

    @staticmethod
    def graph_memory_trace_summary(network_or_runtime: Any, task_id: str | None = None) -> dict[str, Any]:
        """
        鞎痊?渡? GraphMemory retrieval ?神??trace ??冽?閬?
        Args:
            network_or_runtime: AgentNetwork ??NetworkRuntime??            task_id: ?交?靘????府 task_id ??撖怎???
        Returns:
            ? reads?rites?elated_task_ids?nsight_ids?drant_hits?xpanded_hits ??hit count ???詻?
        ??雿:
            ?芾???runtime.shared_memory_reads/shared_memory_writes嚗??孛?潭??retrieval ?神?乓?        """
        runtime = getattr(network_or_runtime, "runtime", network_or_runtime)
        reads_source = list(getattr(runtime, "shared_memory_reads", []) or []) if runtime is not None else []
        writes_source = list(getattr(runtime, "shared_memory_writes", []) or []) if runtime is not None else []

        def same_task(item: dict[str, Any]) -> bool:
            return not task_id or str(item.get("task_id", "")) == str(task_id)

        reads = [
            item
            for item in reads_source
            if isinstance(item, dict) and item.get("source") == "graph_memory" and same_task(item)
        ]
        writes = [
            item
            for item in writes_source
            if isinstance(item, dict)
            and item.get("memory_type") in {"graph_memory", "interaction_graph"}
            and same_task(item)
        ]

        related_task_ids: list[str] = []
        insight_ids: list[str] = []
        qdrant_hits: list[dict[str, Any]] = []
        expanded_hits: list[dict[str, Any]] = []
        suppressed_task_ids: list[str] = []

        def append_unique(values: list[str], value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)

        for read in reads:
            for key in ("related_task_ids", "seed_task_ids", "expanded_task_ids"):
                for value in read.get(key, []) or []:
                    append_unique(related_task_ids, value)

            for value in read.get("insight_ids", []) or []:
                append_unique(insight_ids, value)

            for value in read.get("suppressed_task_ids", []) or []:
                append_unique(suppressed_task_ids, value)

            for key in ("qdrant_hits", "seed_task_hits"):
                for hit in read.get(key, []) or []:
                    if isinstance(hit, dict):
                        qdrant_hits.append(hit)

            for key in ("expanded_hits", "expanded_task_hits"):
                for hit in read.get(key, []) or []:
                    if isinstance(hit, dict):
                        expanded_hits.append(hit)

        insight_hit = bool(insight_ids)
        task_memory_hit = bool(related_task_ids or qdrant_hits or expanded_hits)
        task_memory_suppressed = bool(suppressed_task_ids)
        retrieval_hit = bool(task_memory_hit or insight_hit)

        return {
            "reads": reads,
            "writes": writes,
            "related_task_ids": related_task_ids,
            "insight_ids": insight_ids,
            "qdrant_hits": qdrant_hits,
            "expanded_hits": expanded_hits,
            "qdrant_hit_count": len(qdrant_hits),
            "expanded_hit_count": len(expanded_hits),
            "suppressed_task_ids": suppressed_task_ids,
            "suppressed_task_count": len(suppressed_task_ids),
            "task_memory_injected": task_memory_hit,
            "task_memory_suppressed": task_memory_suppressed,
            "retrieval_hit": retrieval_hit,
            "insight_hit": insight_hit,
            "task_memory_hit": task_memory_hit,
        }

    @staticmethod
    def print_memory_usage_summary(
        network_or_runtime: Any,
        *,
        task_id: str | None = None,
        prefix: str = "   [MEMORY]",
    ) -> None:
        runtime = getattr(network_or_runtime, "runtime", network_or_runtime)
        summary = BenchmarkLogger.graph_memory_trace_summary(runtime, task_id=task_id)
        reads = summary.get("reads", []) or []
        writes = summary.get("writes", []) or []
        cache_hits = sum(1 for item in reads if isinstance(item, dict) and item.get("cache_hit"))
        stages = sorted(
            {
                str(item.get("stage", "") or "")
                for item in reads
                if isinstance(item, dict) and str(item.get("stage", "") or "").strip()
            }
        )
        targets = sorted(
            {
                str(item.get("injection_target", "") or "")
                for item in reads
                if isinstance(item, dict) and str(item.get("injection_target", "") or "").strip()
            }
        )
        agents = sorted(
            {
                str(item.get("agent_id", "") or "")
                for item in reads
                if isinstance(item, dict) and str(item.get("agent_id", "") or "").strip()
            }
        )
        print(
            f"{prefix} 記憶讀取次數={len(reads)} "
            f"任務記憶命中={bool(summary.get('task_memory_hit'))} "
            f"任務記憶被降噪={bool(summary.get('task_memory_suppressed'))} "
            f"全域 insight 命中={bool(summary.get('insight_hit'))} "
            f"retrieval 命中={bool(summary.get('retrieval_hit'))} "
            f"cache 命中={cache_hits} "
            f"qdrant 命中={summary.get('qdrant_hit_count', 0)} "
            f"擴展命中={summary.get('expanded_hit_count', 0)} "
            f"寫入次數={len(writes)}"
        )
        if stages or targets or agents:
            print(
                f"{prefix} 階段={stages or []} "
                f"注入目標={targets or []} "
                f"agents={agents or []}"
            )
        related_task_ids = list(summary.get("related_task_ids", []) or [])
        insight_ids = list(summary.get("insight_ids", []) or [])
        if related_task_ids or insight_ids:
            print(
                f"{prefix} 相關任務 id={related_task_ids[:3]} "
                f"insight id={insight_ids[:3]}"
            )
        suppressed_task_ids = list(summary.get("suppressed_task_ids", []) or [])
        if suppressed_task_ids:
            print(f"{prefix} 被降噪的任務 id={suppressed_task_ids[:3]}")


setup_utf8_log = BenchmarkLogger.setup_utf8_log
restore_stdio = BenchmarkLogger.restore_stdio
open_utf8_text_log = BenchmarkLogger.open_utf8_text_log
write_indented_block = BenchmarkLogger.write_indented_block
write_json_line = BenchmarkLogger.write_json_line
token_usage_summary = BenchmarkLogger.token_usage_summary
graph_memory_trace_summary = BenchmarkLogger.graph_memory_trace_summary
print_memory_usage_summary = BenchmarkLogger.print_memory_usage_summary

