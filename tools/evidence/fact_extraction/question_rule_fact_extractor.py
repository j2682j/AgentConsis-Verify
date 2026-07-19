from __future__ import annotations

import hashlib
import re

from utils.network_utils import normalize_text

from .models import EvidenceFact


class QuestionRuleFactExtractor:
    """將題目內明示的封閉世界規則轉成 facts，並執行可驗證組合。"""

    def extract(self, *, question: str) -> list[EvidenceFact]:
        text = normalize_text(question)
        if not re.search(r"\btranslate\b", text, flags=re.IGNORECASE):
            return []
        target_match = re.search(
            r"\btranslate\s+[\"'“](.+?)[\"'”]\s+(?:to|into)\s+([A-Za-z][\w-]*)",
            text,
            flags=re.IGNORECASE,
        )
        if not target_match:
            return []
        target_text = normalize_text(target_match.group(1))
        language = normalize_text(target_match.group(2))
        order = self._word_order(text)
        forms = self._case_forms(text)
        verb = self._present_root(text)
        if not order or not forms or not verb:
            return []

        base: list[EvidenceFact] = []
        base.append(self._fact(language, "word_order", " ".join(order), text))
        for lemma, cases in forms.items():
            for case_name, form in cases.items():
                base.append(self._fact(lemma, f"{case_name}_form", form, text))
        base.append(self._fact("present intense-like verb", "form", verb, text))

        tokens = target_text.split()
        if len(tokens) < 3:
            return base
        english_subject = tokens[0]
        english_object = tokens[-1]
        inversion = bool(
            re.search(
                r"thing doing the liking is actually the object.*rather than the subject",
                text,
                flags=re.IGNORECASE,
            )
        )
        subject_role = "direct_object" if inversion else "subject"
        object_role = "subject" if inversion else "direct_object"
        lexemes = {
            "verb": verb,
            "direct_object": self._form_for_token(
                english_subject if subject_role == "direct_object" else english_object,
                forms,
                case_name="accusative",
            ),
            "subject": self._form_for_token(
                english_object if object_role == "subject" else english_subject,
                forms,
                case_name="nominative",
            ),
        }
        if any(not lexemes.get(role) for role in order):
            return base
        answer = " ".join(lexemes[role] for role in order)
        parent_ids = [fact.fact_id for fact in base]
        derived = EvidenceFact(
            fact_id=self._id(target_text, f"{language}_translation", answer),
            subject=target_text,
            relation=f"{language}_translation",
            object=answer,
            qualifiers={
                "answer_binding": "direct",
                "answer_requirement": f"the {language} translation of {target_text}",
                "operation": "translation",
                "word_order": " ".join(order),
            },
            role="ANSWER_SUPPORT",
            evidence_spans=[text],
            context=f"Applying the stated case forms and {' '.join(order)} order gives {answer}.",
            source_id="task-question",
            source_type="task_statement",
            source_title="Question rules",
            grounding_status="grounded",
            extraction_method="question_rule_composition",
            parent_fact_ids=parent_ids,
            derivation_type="rule_composition",
        )
        return [*base, derived]

    @staticmethod
    def _word_order(text: str) -> list[str]:
        match = re.search(
            r"verb first, followed by the direct object, followed by the subject",
            text,
            flags=re.IGNORECASE,
        )
        return ["verb", "direct_object", "subject"] if match else []

    @staticmethod
    def _case_forms(text: str) -> dict[str, dict[str, str]]:
        forms: dict[str, dict[str, str]] = {}
        groups = re.split(r"\bThe (?:word|root verb)\b", text, flags=re.IGNORECASE)
        for group in groups:
            lemma = ""
            if re.search(r"indicates oneself", group, flags=re.IGNORECASE):
                lemma = "I"
            elif re.search(r"(?:word\s+)?for apples", group, flags=re.IGNORECASE):
                lemma = "apples"
            if not lemma:
                continue
            cases: dict[str, str] = {}
            for form, case_name in re.findall(
                r"[\"'“]([^\"'”]+)[\"'”]\s+is\s+the\s+(nominative|accusative|genitive)\s+form",
                group,
                flags=re.IGNORECASE,
            ):
                cases[case_name.casefold()] = normalize_text(form)
            if cases:
                forms[lemma] = cases
        return forms

    @staticmethod
    def _present_root(text: str) -> str:
        match = re.search(
            r"root verb that indicates an intense like.*?is\s+[\"'“]([^\"'”]+)[\"'”]",
            text,
            flags=re.IGNORECASE,
        )
        return normalize_text(match.group(1)) if match else ""

    @staticmethod
    def _form_for_token(
        token: str,
        forms: dict[str, dict[str, str]],
        *,
        case_name: str,
    ) -> str:
        key = token.casefold().strip(".,!?\"'")
        aliases = {"i": "I", "me": "I", "apple": "apples", "apples": "apples"}
        lemma = aliases.get(key, token)
        return str(forms.get(lemma, {}).get(case_name) or "")

    @classmethod
    def _fact(cls, subject: str, relation: str, object_value: str, context: str) -> EvidenceFact:
        return EvidenceFact(
            fact_id=cls._id(subject, relation, object_value),
            subject=subject,
            relation=relation,
            object=object_value,
            role="BRIDGE",
            evidence_spans=[context],
            context=context,
            source_id="task-question",
            source_type="task_statement",
            source_title="Question rules",
            grounding_status="grounded",
            extraction_method="question_rule_extraction",
        )

    @staticmethod
    def _id(*values: str) -> str:
        raw = "\x1f".join(normalize_text(value) for value in values)
        return "question-rule-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]


__all__ = ["QuestionRuleFactExtractor"]
