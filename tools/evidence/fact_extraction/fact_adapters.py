from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any, Iterable, Protocol

from utils.network_utils import normalize_text

from .fact_store import TaskFactStore
from .completeness_contract import (
    AbsenceCheck,
    CompletenessContract,
    SetDifferenceDerivation,
)
from .models import EvidenceFact


class FactAdapter(Protocol):
    """將既有證據結果轉換為可追溯的任務事實。"""

    def supports(self, item: dict[str, Any], *, source_scope: str) -> bool: ...

    def convert(
        self,
        item: dict[str, Any],
        *,
        question: str,
        source_scope: str,
    ) -> list[EvidenceFact]: ...


class SemanticFactAdapter:
    """讀取 Search、附件或媒體模組已抽取的語意事實。"""

    def supports(self, item: dict[str, Any], *, source_scope: str) -> bool:
        return bool(self._fact_dicts(item))

    def convert(
        self,
        item: dict[str, Any],
        *,
        question: str,
        source_scope: str,
    ) -> list[EvidenceFact]:
        tool_name = normalize_text(item.get("tool_name"))
        result: list[EvidenceFact] = []
        for value in self._fact_dicts(item):
            fact = EvidenceFact.from_dict(value)
            if fact.grounding_status != "grounded":
                continue
            source_type = fact.source_type or self._source_type(tool_name, source_scope)
            result.append(replace(fact, source_type=source_type))
        return result

    def _fact_dicts(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        values: list[Any] = []
        values.extend(list(item.get("semantic_facts") or []))
        values.extend(list(raw.get("semantic_facts") or []))
        for evidence_item in list(raw.get("evidence_items") or []):
            if isinstance(evidence_item, dict):
                values.extend(list(evidence_item.get("semantic_facts") or []))
        retrieval = raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
        values.extend(list(retrieval.get("semantic_facts") or []))
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, dict):
                continue
            key = str(value.get("fact_id") or "") or repr(sorted(value.items()))
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    @staticmethod
    def _source_type(tool_name: str, source_scope: str) -> str:
        if tool_name == "search":
            return "search"
        if tool_name in {"attachment_reader", "video_evidence", "video_transcript"}:
            return "attachment"
        return source_scope or tool_name or "evidence"


class DeterministicHandlerFactAdapter:
    """直接轉換已驗證的 Handler 輸出，不再呼叫語言模型。"""

    _TOOLS = {
        "deterministic_handler_router",
        "attachment_strategy_handler",
        "deterministic_solver",
        "python_calculator",
    }

    def supports(self, item: dict[str, Any], *, source_scope: str) -> bool:
        return normalize_text(item.get("tool_name")) in self._TOOLS

    def convert(
        self,
        item: dict[str, Any],
        *,
        question: str,
        source_scope: str,
    ) -> list[EvidenceFact]:
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        trust = item.get("handler_trust") if isinstance(item.get("handler_trust"), dict) else {}
        output_type = normalize_text(
            trust.get("effective_output_type")
            or item.get("effective_output_type")
            or item.get("output_type")
            or trust.get("output_type")
            or raw.get("output_type")
        )
        value = normalize_text(
            item.get("value")
            or trust.get("answer")
            or raw.get("answer")
            or raw.get("final_answer")
        )
        finality = trust.get("finality") if isinstance(trust.get("finality"), dict) else {}
        if not finality and isinstance(item.get("finality"), dict):
            finality = dict(item.get("finality") or {})
        finality_status = normalize_text(finality.get("status"))
        legacy_finality = not finality_status
        evidence_valid = bool(
            item.get("evidence_valid")
            or trust.get("trusted")
            or trust.get("usable_as_intermediate")
        )
        if output_type not in {"final_answer", "intermediate_value"} or not value:
            return []
        if output_type == "final_answer" and (
            not evidence_valid
            or (not legacy_finality and finality_status not in {"final", "legacy_accepted"})
        ):
            return []
        role = "ANSWER_SUPPORT" if output_type == "final_answer" else "BRIDGE"
        relation = normalize_text(
            item.get("semantic_role")
            or trust.get("semantic_role")
            or raw.get("semantic_role")
            or "has deterministic result"
        )
        supporting_inputs = self._strings(
            item.get("supporting_inputs")
            or trust.get("supporting_inputs")
            or raw.get("supporting_inputs")
        )
        evidence_text = normalize_text(
            item.get("output_text") or trust.get("evidence_text") or raw.get("evidence_text")
        )
        source_id = normalize_text(
            item.get("handler_name") or raw.get("handler_name") or item.get("tool_name")
        )
        fact_id = self._fact_id(source_id, relation, value, source_scope)
        return [
            EvidenceFact(
                fact_id=fact_id,
                subject=normalize_text(question) or source_id,
                relation=relation,
                object=value,
                qualifiers={
                    "output_type": output_type,
                    "answer_binding": (
                        "direct" if output_type == "final_answer" else "bridge"
                    ),
                    "answer_requirement": normalize_text(question),
                    "finality_status": finality_status or "legacy_accepted",
                    "finality": dict(finality),
                },
                polarity="positive",
                role=role,
                evidence_spans=[evidence_text] if evidence_text else supporting_inputs[:2],
                context=evidence_text or " | ".join(supporting_inputs),
                source_id=source_id,
                source_type="handler" if source_scope == "evidence_prepare" else "stage1_tool",
                source_title=source_id,
                grounding_status="grounded",
                extraction_method="deterministic_adapter",
            )
        ]

    @staticmethod
    def _strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [normalize_text(value)] if normalize_text(value) else []
        if isinstance(value, (list, tuple, set)):
            return [normalize_text(item) for item in value if normalize_text(item)]
        return []

    @staticmethod
    def _fact_id(source_id: str, relation: str, value: str, scope: str) -> str:
        digest = hashlib.sha1(
            f"{source_id}\x1f{relation}\x1f{value}\x1f{scope}".encode("utf-8")
        ).hexdigest()[:16]
        return f"handler-{digest}"


class SearchContractFactAdapter:
    """將 Search direct contract 轉換為可供驗證與推導的事實。"""

    def supports(self, item: dict[str, Any], *, source_scope: str) -> bool:
        if normalize_text(item.get("tool_name")) != "search":
            return False
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        return any(
            isinstance(evidence_item, dict) and evidence_item.get("direct_contracts")
            for evidence_item in list(raw.get("evidence_items") or [])
        )

    def convert(
        self,
        item: dict[str, Any],
        *,
        question: str,
        source_scope: str,
    ) -> list[EvidenceFact]:
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        result: list[EvidenceFact] = []
        for evidence_item in list(raw.get("evidence_items") or []):
            if not isinstance(evidence_item, dict):
                continue
            context = normalize_text(evidence_item.get("text"))
            evidence_id = normalize_text(evidence_item.get("evidence_id"))
            source_id = normalize_text(evidence_item.get("source_id")) or evidence_id
            source_title = normalize_text(evidence_item.get("title")) or source_id
            for contract in list(evidence_item.get("direct_contracts") or []):
                if not isinstance(contract, dict):
                    continue
                subject = normalize_text(contract.get("subject"))
                relation = normalize_text(contract.get("relation"))
                object_value = normalize_text(contract.get("object"))
                answer_span = normalize_text(contract.get("answer_span"))
                qualifiers = {
                    str(key): str(value)
                    for key, value in dict(contract.get("qualifiers") or {}).items()
                }
                if not self._valid_direct_contract(
                    subject=subject,
                    relation=relation,
                    object_value=object_value,
                    answer_span=answer_span,
                    grounding_status=normalize_text(contract.get("grounding_status")),
                    answer_binding=normalize_text(qualifiers.get("answer_binding")),
                    context=context,
                ):
                    continue
                goal_id = normalize_text(contract.get("goal_id"))
                fact_id = self._fact_id(evidence_id, goal_id, object_value)
                qualifiers.update(
                    {
                        "evidence_id": evidence_id,
                        "url": normalize_text(evidence_item.get("url")),
                        "answer_binding": "direct",
                        "answer_requirement": normalize_text(
                            contract.get("answer_requirement")
                        ),
                    }
                )
                result.append(
                    EvidenceFact(
                        fact_id=fact_id,
                        subject=subject,
                        relation=relation,
                        object=object_value,
                        qualifiers=qualifiers,
                        polarity="positive",
                        role="ANSWER_SUPPORT",
                        goal_id=goal_id,
                        evidence_spans=list(contract.get("evidence_spans") or [])[:2]
                        or [answer_span],
                        context=context,
                        source_id=source_id,
                        source_type="search",
                        source_title=source_title,
                        grounding_status="grounded",
                        extraction_method="direct_contract_adapter",
                    )
                )
        return result

    @staticmethod
    def _valid_direct_contract(
        *,
        subject: str,
        relation: str,
        object_value: str,
        answer_span: str,
        grounding_status: str,
        answer_binding: str,
        context: str,
    ) -> bool:
        if not all((subject, relation, object_value, answer_span, context)):
            return False
        if grounding_status.casefold() != "grounded":
            return False
        if answer_binding.casefold() != "direct":
            return False
        if object_value.casefold() != answer_span.casefold():
            return False
        return object_value.casefold() in context.casefold()

    @staticmethod
    def _fact_id(evidence_id: str, goal_id: str, object_value: str) -> str:
        payload = f"{evidence_id}\x1f{goal_id}\x1f{object_value}"
        return "contract-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class TaskFactCollector:
    """透過來源 Adapter 將各類證據寫入同一個任務事實庫。"""

    def __init__(self, adapters: Iterable[FactAdapter] | None = None) -> None:
        self.adapters = list(
            adapters
            or [
                SemanticFactAdapter(),
                SearchContractFactAdapter(),
                DeterministicHandlerFactAdapter(),
            ]
        )

    def collect_item(
        self,
        store: TaskFactStore,
        item: Any,
        *,
        question: str,
        source_scope: str,
    ) -> int:
        if not isinstance(item, dict):
            return 0
        self._collect_audit_records(store, item)
        added = 0
        for adapter in self.adapters:
            if adapter.supports(item, source_scope=source_scope):
                added += store.extend(
                    adapter.convert(
                        item,
                        question=question,
                        source_scope=source_scope,
                    )
                )
        return added

    def _collect_audit_records(
        self,
        store: TaskFactStore,
        item: dict[str, Any],
    ) -> None:
        raw = item.get("raw_result")
        containers = [item]
        if isinstance(raw, dict):
            containers.append(raw)
            diagnostics = raw.get("diagnostics")
            if isinstance(diagnostics, dict):
                containers.append(diagnostics)
            retrieval = raw.get("retrieval")
            if isinstance(retrieval, dict):
                containers.append(retrieval)
        seen: set[tuple[str, str]] = set()
        for container in containers:
            for value in list(container.get("completeness_contracts") or []):
                if not isinstance(value, dict):
                    continue
                contract = CompletenessContract.from_dict(value)
                key = ("contract", contract.contract_id)
                if key not in seen:
                    store.add_completeness_contract(contract)
                    seen.add(key)
            for value in list(container.get("absence_checks") or []):
                if not isinstance(value, dict):
                    continue
                check = AbsenceCheck.from_dict(value)
                key = ("absence", check.check_id)
                if key not in seen:
                    store.add_absence_check(check)
                    seen.add(key)
            for value in list(container.get("set_difference_derivations") or []):
                if not isinstance(value, dict):
                    continue
                derivation = SetDifferenceDerivation.from_dict(value)
                key = ("set_difference", derivation.derivation_id)
                if key not in seen:
                    store.add_set_difference_derivation(derivation)
                    seen.add(key)

    def collect_many(
        self,
        store: TaskFactStore,
        items: Iterable[Any],
        *,
        question: str,
        source_scope: str,
    ) -> int:
        return sum(
            self.collect_item(
                store,
                item,
                question=question,
                source_scope=source_scope,
            )
            for item in items
        )


__all__ = [
    "DeterministicHandlerFactAdapter",
    "FactAdapter",
    "SemanticFactAdapter",
    "SearchContractFactAdapter",
    "TaskFactCollector",
]
