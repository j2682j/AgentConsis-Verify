from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
import unicodedata
from typing import Any, Iterable, Mapping

from utils.network_utils import normalize_text


VALID_ABSENCE_STATUSES = {"absent", "present", "unknown"}


@dataclass(frozen=True)
class CompletenessContract:
    """描述一個可用於 closed-world 判斷的完整來源範圍。"""

    contract_id: str
    scope_id: str
    source_id: str
    source_type: str
    expected_units: int | None
    processed_units: int
    complete: bool
    completion_reason: str
    unit_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompletenessContract":
        expected_value = value.get("expected_units")
        try:
            expected = int(expected_value) if expected_value not in (None, "") else None
        except (TypeError, ValueError):
            expected = None
        try:
            processed = max(0, int(value.get("processed_units", 0)))
        except (TypeError, ValueError):
            processed = 0
        return cls(
            contract_id=normalize_text(str(value.get("contract_id") or "")),
            scope_id=normalize_text(str(value.get("scope_id") or "")),
            source_id=normalize_text(str(value.get("source_id") or "")),
            source_type=normalize_text(str(value.get("source_type") or "")),
            expected_units=expected,
            processed_units=processed,
            complete=bool(value.get("complete", False)),
            completion_reason=normalize_text(
                str(value.get("completion_reason") or "")
            ),
            unit_ids=[
                normalize_text(str(item))
                for item in list(value.get("unit_ids") or [])
                if normalize_text(str(item))
            ],
            warnings=[
                normalize_text(str(item))
                for item in list(value.get("warnings") or [])
                if normalize_text(str(item))
            ],
        )


@dataclass(frozen=True)
class AbsenceCheck:
    """保存指定詞在一個完整性範圍內的可稽核查找結果。"""

    check_id: str
    scope_id: str
    target: str
    normalized_target: str
    status: str
    matched_unit_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AbsenceCheck":
        status = normalize_text(str(value.get("status") or "unknown")).lower()
        if status not in VALID_ABSENCE_STATUSES:
            status = "unknown"
        return cls(
            check_id=normalize_text(str(value.get("check_id") or "")),
            scope_id=normalize_text(str(value.get("scope_id") or "")),
            target=normalize_text(str(value.get("target") or "")),
            normalized_target=normalize_text(
                str(value.get("normalized_target") or "")
            ),
            status=status,
            matched_unit_ids=[
                normalize_text(str(item))
                for item in list(value.get("matched_unit_ids") or [])
                if normalize_text(str(item))
            ],
            reason=normalize_text(str(value.get("reason") or "")),
        )


@dataclass(frozen=True)
class SetDifferenceDerivation:
    """保存完整集合與已觀察集合之間的差異推導。"""

    derivation_id: str
    universe_fact_ids: list[str] = field(default_factory=list)
    observed_fact_ids: list[str] = field(default_factory=list)
    missing_values: list[str] = field(default_factory=list)
    completeness_contract_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SetDifferenceDerivation":
        return cls(
            derivation_id=normalize_text(str(value.get("derivation_id") or "")),
            universe_fact_ids=_strings(value.get("universe_fact_ids")),
            observed_fact_ids=_strings(value.get("observed_fact_ids")),
            missing_values=_strings(value.get("missing_values")),
            completeness_contract_ids=_strings(
                value.get("completeness_contract_ids")
            ),
        )


class CompletenessContractBuilder:
    """依來源 metadata 建立保守的範圍完整性契約。"""

    _INCOMPLETE_WARNING_RE = re.compile(
        r"(?:truncat|partial|missing|failed|error|timeout|ocr|unreadable)",
        re.IGNORECASE,
    )

    def build(
        self,
        *,
        scope_id: str,
        source_id: str,
        source_type: str,
        expected_units: int | None,
        processed_units: int,
        complete: bool | None = None,
        completion_reason: str = "",
        unit_ids: Iterable[str] = (),
        warnings: Iterable[str] = (),
    ) -> CompletenessContract:
        expected = max(0, int(expected_units)) if expected_units is not None else None
        processed = max(0, int(processed_units))
        normalized_warnings = [
            normalize_text(str(item))
            for item in warnings
            if normalize_text(str(item))
        ]
        warning_blocks_completion = any(
            self._INCOMPLETE_WARNING_RE.search(item) for item in normalized_warnings
        )
        if complete is None:
            complete = bool(
                expected is not None
                and processed >= expected
                and not warning_blocks_completion
            )
        else:
            complete = bool(complete and not warning_blocks_completion)
        reason = normalize_text(completion_reason)
        if not reason:
            if warning_blocks_completion:
                reason = "source_warning_indicates_incomplete_scope"
            elif complete:
                reason = "all_expected_units_processed"
            elif expected is None:
                reason = "expected_unit_count_unknown"
            else:
                reason = "processed_unit_count_incomplete"
        normalized_scope = normalize_text(scope_id) or normalize_text(source_id)
        normalized_source = normalize_text(source_id) or normalized_scope
        contract_id = self._contract_id(
            normalized_scope,
            normalized_source,
            expected,
            processed,
        )
        return CompletenessContract(
            contract_id=contract_id,
            scope_id=normalized_scope,
            source_id=normalized_source,
            source_type=normalize_text(source_type),
            expected_units=expected,
            processed_units=processed,
            complete=complete,
            completion_reason=reason,
            unit_ids=list(
                dict.fromkeys(
                    normalize_text(str(item))
                    for item in unit_ids
                    if normalize_text(str(item))
                )
            ),
            warnings=normalized_warnings,
        )

    def from_attachment_payload(
        self,
        payload: Mapping[str, Any],
    ) -> CompletenessContract:
        provenance = dict(payload.get("provenance") or {})
        native = dict(payload.get("native_metadata") or {})
        source_id = normalize_text(
            str(provenance.get("file_path") or provenance.get("source") or "attachment")
        )
        source_type = normalize_text(
            str(provenance.get("file_type") or "attachment")
        )
        unit_ids: list[str] = []
        processed = 0
        expected: int | None = None
        truncated = False

        text_blocks = list(payload.get("text_blocks") or [])
        for index, _ in enumerate(text_blocks, start=1):
            unit_ids.append(f"T{index}")
        processed += len(text_blocks)

        tables = [item for item in list(payload.get("tables") or []) if isinstance(item, Mapping)]
        for table_index, table in enumerate(tables, start=1):
            rows = list(table.get("rows") or [])
            unit_ids.extend(
                f"TABLE{table_index}-R{row_index}"
                for row_index in range(1, len(rows) + 1)
            )
            processed += len(rows)
            truncated = truncated or bool(table.get("truncated", False))

        metadata_expected = native.get("row_count") or native.get("page_count")
        try:
            expected = int(metadata_expected) if metadata_expected not in (None, "") else None
        except (TypeError, ValueError):
            expected = None
        if expected is None and processed:
            expected = processed
        parse_status = normalize_text(
            str(provenance.get("parse_status") or native.get("parse_status") or "success")
        ).lower()
        warnings = list(native.get("warnings") or [])
        if truncated:
            warnings.append("attachment content truncated")
        complete = bool(parse_status == "success" and not truncated and processed > 0)
        return self.build(
            scope_id=f"attachment:{source_id}",
            source_id=source_id,
            source_type=source_type,
            expected_units=expected,
            processed_units=processed,
            complete=complete,
            completion_reason=(
                "attachment_parser_completed" if complete else "attachment_scope_incomplete"
            ),
            unit_ids=unit_ids,
            warnings=warnings,
        )

    @staticmethod
    def _contract_id(
        scope_id: str,
        source_id: str,
        expected_units: int | None,
        processed_units: int,
    ) -> str:
        payload = "\x1f".join(
            [scope_id, source_id, str(expected_units), str(processed_units)]
        )
        return "CC-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class AbsenceChecker:
    """使用完整來源單位索引判定 present、absent 或 unknown。"""

    def check(
        self,
        *,
        contract: CompletenessContract,
        target: str,
        units: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> AbsenceCheck:
        normalized_target = self.normalize_target(target)
        unit_map = dict(units)
        matched = [
            normalize_text(str(unit_id))
            for unit_id, text in unit_map.items()
            if self._contains(text, target)
        ]
        if matched:
            status = "present"
            reason = "target_found_in_scope"
        elif contract.complete:
            status = "absent"
            reason = "complete_scope_contains_no_target"
        else:
            status = "unknown"
            reason = "absence_unverifiable_in_incomplete_scope"
        raw = "\x1f".join([contract.contract_id, normalized_target, status])
        return AbsenceCheck(
            check_id="AC-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
            scope_id=contract.scope_id,
            target=normalize_text(target),
            normalized_target=normalized_target,
            status=status,
            matched_unit_ids=matched,
            reason=reason,
        )

    @staticmethod
    def normalize_target(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", normalize_text(value)).casefold()
        return " ".join(normalized.split())

    def _contains(self, source: str, target: str) -> bool:
        source_key = self.normalize_target(source)
        target_key = self.normalize_target(target)
        if not source_key or not target_key:
            return False
        if re.fullmatch(r"[\w'-]+", target_key):
            return bool(re.search(rf"(?<!\w){re.escape(target_key)}(?!\w)", source_key))
        return target_key in source_key


def _strings(value: Any) -> list[str]:
    return [
        normalize_text(str(item))
        for item in list(value or [])
        if normalize_text(str(item))
    ]


__all__ = [
    "AbsenceCheck",
    "AbsenceChecker",
    "CompletenessContract",
    "CompletenessContractBuilder",
    "SetDifferenceDerivation",
    "VALID_ABSENCE_STATUSES",
]
