from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from utils.network_utils import normalize_text

from ..query.relation_plan import RelationGoal


@dataclass(frozen=True)
class DocumentVerification:
    """Record whether one complete document contains the requested term."""

    document_id: str
    record_id: str
    title: str
    target: str
    status: str
    content_scope: str
    content_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NegativeVerificationResult:
    """Store the auditable result of an explicit absence check."""

    goal_id: str
    resolved: bool
    resolved_values: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    verifications: list[DocumentVerification] = field(default_factory=list)
    missing_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "resolved": self.resolved,
            "resolved_values": list(self.resolved_values),
            "evidence_ids": list(self.evidence_ids),
            "verifications": [item.to_dict() for item in self.verifications],
            "missing_reason": self.missing_reason,
        }


class FullDocumentVerifier:
    """Verify explicit negative goals only against complete, untruncated documents."""

    def verify(
        self,
        *,
        goal: RelationGoal,
        documents: Iterable[Any],
        corpus_documents: Iterable[Mapping[str, Any]] = (),
    ) -> NegativeVerificationResult:
        if goal.polarity != "negative":
            return NegativeVerificationResult(
                goal_id=goal.goal_id,
                resolved=False,
                missing_reason="goal_is_not_negative",
            )
        target = normalize_text(goal.target)
        if not target:
            return NegativeVerificationResult(
                goal_id=goal.goal_id,
                resolved=False,
                missing_reason="missing_negative_target",
            )

        observed_documents = list(documents)
        corpus = list(corpus_documents)
        complete_documents = self._complete_document_groups(
            observed_documents=observed_documents,
            corpus_documents=corpus,
        )
        if goal.verification_scope == "collection":
            return self._verify_collection(
                goal=goal,
                target=target,
                complete_documents=complete_documents,
                corpus_documents=corpus,
                observed_documents=observed_documents,
            )

        relevant = self._relevant_documents(goal, complete_documents)
        if not relevant:
            return NegativeVerificationResult(
                goal_id=goal.goal_id,
                resolved=False,
                missing_reason="complete_full_document_required",
            )
        verifications = [self._verify_document(item, target) for item in relevant]
        absent = [item for item in verifications if item.status == "absent_verified"]
        resolved = bool(absent) and not any(
            item.status == "present" for item in verifications
        )
        return NegativeVerificationResult(
            goal_id=goal.goal_id,
            resolved=resolved,
            resolved_values=[item.title for item in absent if item.title],
            evidence_ids=[item.document_id for item in absent if item.document_id],
            verifications=verifications,
            missing_reason="" if resolved else "target_present_or_document_incomplete",
        )

    def _verify_collection(
        self,
        *,
        goal: RelationGoal,
        target: str,
        complete_documents: list[Any],
        corpus_documents: Iterable[Mapping[str, Any]],
        observed_documents: Iterable[Any],
    ) -> NegativeVerificationResult:
        expected = self._expected_collection_records(
            corpus_documents,
            observed_documents=observed_documents,
        )
        if not expected:
            return NegativeVerificationResult(
                goal_id=goal.goal_id,
                resolved=False,
                missing_reason="collection_records_required",
            )
        complete_by_record: dict[str, Any] = {}
        for document in complete_documents:
            record_id = normalize_text(self._field(document, "record_id"))
            if record_id:
                complete_by_record.setdefault(record_id, document)

        verifications: list[DocumentVerification] = []
        missing_records: list[str] = []
        for record_id, metadata in expected.items():
            document = complete_by_record.get(record_id)
            if document is None:
                missing_records.append(record_id)
                continue
            verification = self._verify_document(document, target)
            if not verification.title:
                verification = DocumentVerification(
                    **{
                        **verification.to_dict(),
                        "title": normalize_text(metadata.get("title", "")),
                    }
                )
            verifications.append(verification)

        absent = [item for item in verifications if item.status == "absent_verified"]
        resolved = (
            not missing_records
            and len(verifications) == len(expected)
            and bool(absent)
        )
        return NegativeVerificationResult(
            goal_id=goal.goal_id,
            resolved=resolved,
            resolved_values=[item.title for item in absent if item.title],
            evidence_ids=[item.document_id for item in absent if item.document_id],
            verifications=verifications,
            missing_reason=(
                "" if resolved else "collection_full_document_coverage_incomplete"
            ),
        )

    def _expected_collection_records(
        self,
        corpus_documents: Iterable[Mapping[str, Any]],
        *,
        observed_documents: Iterable[Any],
    ) -> dict[str, dict[str, str]]:
        corpus = list(corpus_documents)
        observed_record_ids = {
            normalize_text(self._field(document, "record_id"))
            for document in observed_documents
            if normalize_text(self._field(document, "record_id"))
        }
        relevant_parent_urls = {
            normalize_text(str(document.get("parent_url") or "")).casefold().rstrip("/")
            for document in corpus
            if normalize_text(str(document.get("record_id") or ""))
            in observed_record_ids
            and normalize_text(str(document.get("parent_url") or ""))
        }
        records: dict[str, dict[str, str]] = {}
        for document in corpus:
            record_id = normalize_text(str(document.get("record_id") or ""))
            content_url = normalize_text(str(document.get("content_url") or ""))
            scope = normalize_text(str(document.get("content_scope") or ""))
            if not record_id or not content_url or scope == "full_document":
                continue
            parent_url = normalize_text(
                str(document.get("parent_url") or "")
            ).casefold().rstrip("/")
            if relevant_parent_urls and parent_url not in relevant_parent_urls:
                continue
            records.setdefault(
                record_id,
                {
                    "title": normalize_text(str(document.get("title") or "")),
                    "content_url": content_url,
                },
            )
        return records

    def _relevant_documents(self, goal: RelationGoal, documents: list[Any]) -> list[Any]:
        if len(documents) <= 1:
            return documents
        subject_terms = {
            term.casefold()
            for term in re.findall(r"[\w'-]{3,}", normalize_text(goal.subject))
        }
        if not subject_terms:
            return documents
        relevant = []
        for document in documents:
            haystack = " ".join(
                [
                    normalize_text(self._field(document, "title")),
                    normalize_text(self._field(document, "text")),
                ]
            ).casefold()
            if any(term in haystack for term in subject_terms):
                relevant.append(document)
        return relevant

    def _verify_document(self, document: Any, target: str) -> DocumentVerification:
        text = normalize_text(self._field(document, "text"))
        pattern = re.compile(rf"(?<!\w){re.escape(target)}(?!\w)", re.IGNORECASE)
        status = "present" if pattern.search(text) else "absent_verified"
        return DocumentVerification(
            document_id=normalize_text(self._field(document, "document_id")),
            record_id=normalize_text(self._field(document, "record_id")),
            title=normalize_text(self._field(document, "title")),
            target=target,
            status=status,
            content_scope=normalize_text(self._field(document, "content_scope")),
            content_complete=bool(self._field(document, "content_complete", False)),
        )

    def _complete_document_groups(
        self,
        *,
        observed_documents: list[Any],
        corpus_documents: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        observed_groups = self._group_complete_documents(observed_documents)
        corpus_groups = self._group_complete_documents(corpus_documents)
        if not corpus_groups:
            return list(observed_groups.values())

        results: list[dict[str, Any]] = []
        for key, group in corpus_groups.items():
            observed = observed_groups.get(key)
            if observed is None:
                continue
            results.append(
                {
                    **group,
                    "document_id": observed["document_id"],
                    "title": observed["title"] or group["title"],
                }
            )
        for key, group in observed_groups.items():
            if key not in corpus_groups:
                results.append(group)
        return results

    def _group_complete_documents(self, documents: Iterable[Any]) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for document in documents:
            if self._field(document, "content_scope") != "full_document":
                continue
            if not bool(self._field(document, "content_complete", False)):
                continue
            if bool(self._field(document, "content_truncated", False)):
                continue
            record_id = normalize_text(self._field(document, "record_id"))
            url = normalize_text(self._field(document, "url"))
            key = record_id.casefold() or url.casefold().rstrip("/")
            if not key:
                continue
            document_id = normalize_text(
                self._field(document, "document_id")
                or self._field(document, "id")
            )
            group = groups.setdefault(
                key,
                {
                    "document_id": document_id,
                    "record_id": record_id,
                    "title": normalize_text(self._field(document, "title")),
                    "url": url,
                    "text_parts": [],
                    "content_scope": "full_document",
                    "content_complete": True,
                    "content_truncated": False,
                },
            )
            text = normalize_text(self._field(document, "text"))
            if text and text not in group["text_parts"]:
                group["text_parts"].append(text)
        return {
            key: {
                **group,
                "text": "\n".join(group.pop("text_parts")),
            }
            for key, group in groups.items()
        }

    def _field(self, value: Any, key: str, default: Any = "") -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)


__all__ = [
    "DocumentVerification",
    "FullDocumentVerifier",
    "NegativeVerificationResult",
]
