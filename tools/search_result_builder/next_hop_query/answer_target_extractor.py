from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.network_utils import normalize_text

from ..query.semantic_impact import SemanticImpactScorer


@dataclass
class AnswerTarget:
    """
    保存從主問句抽取出的答案焦點與語意角色。

    Args:
        - phrase: 疑問焦點片語。
        - head: 焦點中心詞。
        - lemma: 中心詞 lemma。
        - role: person、date、duration、distance 等答案角色。
        - unit: 問題要求的單位。
        - wh_word: 主疑問詞。
        - confidence: 抽取結果信心值。
        - method: dependency、ontology 或 embedding fallback。

    Returns:
        - AnswerTarget: Retrieval Controller 使用的答案目標。
    """

    phrase: str = ""
    head: str = ""
    lemma: str = ""
    role: str = ""
    unit: str = ""
    wh_word: str = ""
    confidence: float = 0.0
    method: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "phrase": self.phrase,
            "head": self.head,
            "lemma": self.lemma,
            "role": self.role,
            "unit": self.unit,
            "wh_word": self.wh_word,
            "confidence": round(self.confidence, 6),
            "method": self.method,
        }


class AnswerTargetExtractor:
    """
    使用 spaCy dependency focus、lemma/unit ontology 與 encoder fallback 抽取答案目標。

    Args:
        - nlp: spaCy language pipeline。
        - semantic_scorer: 無法由 dependency/ontology 判斷時使用的 encoder。
        - embedding_threshold: embedding fallback 最低相似度。

    Returns:
        - AnswerTargetExtractor: 問句答案焦點抽取器。
    """

    ROLE_LEMMAS = {
        "person": {
            "author",
            "director",
            "employee",
            "founder",
            "name",
            "person",
            "recipient",
            "scientist",
            "winner",
        },
        "organization": {
            "agency",
            "company",
            "institution",
            "organization",
            "publisher",
            "university",
        },
        "place": {
            "city",
            "country",
            "destination",
            "location",
            "place",
            "site",
        },
        "date": {"date", "day", "month", "year"},
        "duration": {"duration", "hour", "minute", "second", "time"},
        "distance": {"distance", "kilometer", "metre", "meter", "mile"},
        "volume": {"volume", "liter", "litre"},
        "percentage": {"percentage", "percent", "rate", "ratio"},
        "number": {
            "amount",
            "count",
            "many",
            "number",
            "quantity",
            "total",
        },
    }
    UNIT_ROLES = {
        "hour": "duration",
        "minute": "duration",
        "second": "duration",
        "week": "duration",
        "kilometer": "distance",
        "km": "distance",
        "meter": "distance",
        "metre": "distance",
        "mile": "distance",
        "liter": "volume",
        "litre": "volume",
        "ml": "volume",
        "m^3": "volume",
        "percent": "percentage",
        "percentage": "percentage",
        "%": "percentage",
    }
    ROLE_DESCRIPTIONS = {
        "person": "a human name or identity",
        "organization": "an organization, company, institution, or agency",
        "place": "a geographic place, city, country, or physical location",
        "date": "a calendar date, year, month, or day",
        "duration": "an amount of elapsed time expressed in hours, minutes, or seconds",
        "distance": "a physical distance expressed in kilometers, meters, or miles",
        "volume": "a physical volume expressed in cubic meters or liters",
        "percentage": "a percentage, rate, ratio, or proportion",
        "number": "a numeric count, quantity, amount, or total",
    }
    WH_WORDS = {"how", "what", "when", "where", "which", "who"}

    def __init__(
        self,
        *,
        nlp: Any | None = None,
        semantic_scorer: SemanticImpactScorer | None = None,
        embedding_threshold: float = 0.35,
    ) -> None:
        self.nlp = nlp
        self.semantic_scorer = semantic_scorer
        self.embedding_threshold = embedding_threshold

    def extract(self, question: str) -> AnswerTarget:
        text = normalize_text(question)
        if not text or self.nlp is None:
            return AnswerTarget()

        doc = self.nlp(text)
        try:
            iter(doc)
        except TypeError:
            return self._fallback_target(text)
        wh_token = self._main_wh_token(doc)
        if wh_token is None:
            return AnswerTarget()

        focus = self._focus_token(wh_token, doc)
        phrase = self._focus_phrase(wh_token, focus)
        lemma = self._lemma(focus)
        unit, unit_role = self._unit_role(wh_token, focus, phrase, doc)
        role = unit_role or self._ontology_role(lemma)
        method = "dependency+unit_ontology" if unit_role else "dependency+lemma_ontology"
        confidence = 0.95 if unit_role else 0.9

        if not role:
            role = self._wh_role(wh_token, focus)
            if role:
                method = "dependency+wh_role"
                confidence = 0.85

        if not role:
            role, similarity = self._embedding_role(phrase or lemma)
            if role:
                method = "dependency+embedding_fallback"
                confidence = similarity

        return AnswerTarget(
            phrase=phrase,
            head=str(getattr(focus, "text", "") or ""),
            lemma=lemma,
            role=role,
            unit=unit,
            wh_word=str(getattr(wh_token, "lemma_", "") or getattr(wh_token, "text", "")).lower(),
            confidence=confidence if role else 0.0,
            method=method if role else "dependency_unresolved",
        )

    def _main_wh_token(self, doc: Any) -> Any | None:
        candidates = [
            token
            for token in doc
            if self._lemma(token) in self.WH_WORDS
        ]
        if not candidates:
            return None
        root = next((token for token in doc if getattr(token, "dep_", "") == "ROOT"), None)
        if root is None:
            return candidates[0]
        return min(
            candidates,
            key=lambda token: (
                self._dependency_distance(token, root),
                int(getattr(token, "i", 0)),
            ),
        )

    def _focus_token(self, wh_token: Any, doc: Any) -> Any:
        wh_lemma = self._lemma(wh_token)
        head = getattr(wh_token, "head", wh_token)
        if wh_lemma == "which":
            return head
        if wh_lemma == "what":
            if getattr(head, "pos_", "") in {"NOUN", "PROPN"}:
                return head
            root = next((token for token in doc if getattr(token, "dep_", "") == "ROOT"), head)
            subjects = [
                child
                for child in getattr(root, "children", [])
                if getattr(child, "dep_", "") in {"nsubj", "nsubjpass", "attr"}
                and getattr(child, "pos_", "") in {"NOUN", "PROPN"}
            ]
            return subjects[0] if subjects else head
        if wh_lemma == "how":
            if self._lemma(head) in {"many", "much"}:
                quantity_head = getattr(head, "head", head)
                if getattr(quantity_head, "pos_", "") in {"NOUN", "PROPN"}:
                    return quantity_head
            if getattr(head, "pos_", "") in {"NOUN", "PROPN", "ADJ", "ADV"}:
                return head
        return wh_token

    def _focus_phrase(self, wh_token: Any, focus: Any) -> str:
        tokens = {wh_token, focus}
        for child in getattr(focus, "children", []):
            if getattr(child, "dep_", "") in {
                "advmod",
                "amod",
                "compound",
                "det",
                "nummod",
                "quantmod",
            }:
                tokens.add(child)
                tokens.update(
                    grandchild
                    for grandchild in getattr(child, "children", [])
                    if getattr(grandchild, "dep_", "") in {"amod", "advmod", "compound"}
                )
        ordered = sorted(tokens, key=lambda token: int(getattr(token, "i", 0)))
        return normalize_text(" ".join(str(getattr(token, "text", "")) for token in ordered))

    def _unit_role(
        self,
        wh_token: Any,
        focus: Any,
        phrase: str,
        doc: Any,
    ) -> tuple[str, str]:
        candidates = [self._lemma(focus)]
        candidates.extend(
            self._lemma(token)
            for token in doc
            if (
                int(getattr(token, "i", 0))
                >= max(0, int(getattr(focus, "i", 0)) - 2)
                and int(getattr(token, "i", 0))
                <= int(getattr(focus, "i", 0)) + 3
            )
        )
        candidates.extend(normalize_text(phrase).lower().split())
        for value in candidates:
            key = value.lower().rstrip("s")
            if (
                key in {"day", "month", "year"}
                and self._lemma(wh_token) == "how"
            ):
                return key, "duration"
            if key in self.UNIT_ROLES:
                return key, self.UNIT_ROLES[key]
        return "", ""

    def _ontology_role(self, lemma: str) -> str:
        for role, lemmas in self.ROLE_LEMMAS.items():
            if lemma in lemmas:
                return role
        return ""

    def _wh_role(self, wh_token: Any, focus: Any) -> str:
        wh_lemma = self._lemma(wh_token)
        focus_lemma = self._lemma(focus)
        if wh_lemma == "who":
            return "person"
        if wh_lemma == "where":
            return "place"
        if wh_lemma == "when":
            return "date"
        phrase = self._focus_phrase(wh_token, focus).lower()
        if wh_lemma == "how" and focus_lemma == "far":
            return "distance"
        if wh_lemma == "how" and focus_lemma == "long":
            return "duration"
        if wh_lemma == "how" and any(
            word in phrase.split() for word in {"many", "much"}
        ):
            return "number"
        return ""

    def _fallback_target(self, question: str) -> AnswerTarget:
        words = normalize_text(question).split()
        if not words:
            return AnswerTarget()
        lowered = [word.lower().strip("?!.,:;") for word in words]
        wh_word = lowered[0]
        role = ""
        phrase_words = words[:3]
        if wh_word == "who":
            role = "person"
        elif wh_word == "where":
            role = "place"
        elif wh_word == "when":
            role = "date"
        elif wh_word == "how" and len(lowered) > 1:
            if lowered[1] in {"many", "much"}:
                role = "number"
            elif lowered[1] == "long":
                role = "duration"
            elif lowered[1] == "far":
                role = "distance"
        return AnswerTarget(
            phrase=normalize_text(" ".join(phrase_words)),
            head=words[min(1, len(words) - 1)],
            lemma=lowered[min(1, len(lowered) - 1)],
            role=role,
            wh_word=wh_word,
            confidence=0.55 if role else 0.0,
            method="lexical_fallback_no_dependency_parser",
        )

    def _embedding_role(self, phrase: str) -> tuple[str, float]:
        if not phrase or self.semantic_scorer is None:
            return "", 0.0
        roles = list(self.ROLE_DESCRIPTIONS)
        descriptions = [self.ROLE_DESCRIPTIONS[role] for role in roles]
        try:
            similarities = self.semantic_scorer.semantic_similarities(
                phrase,
                descriptions,
            )
        except Exception:
            return "", 0.0
        if not similarities:
            return "", 0.0
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        score = float(similarities[best_index])
        if score < self.embedding_threshold:
            return "", score
        return roles[best_index], score

    def _dependency_distance(self, token: Any, root: Any) -> int:
        current = token
        for distance in range(20):
            if current is root:
                return distance
            parent = getattr(current, "head", current)
            if parent is current:
                break
            current = parent
        return 20

    def _lemma(self, token: Any) -> str:
        return str(
            getattr(token, "lemma_", "")
            or getattr(token, "text", "")
            or ""
        ).strip().lower()


__all__ = ["AnswerTarget", "AnswerTargetExtractor"]
