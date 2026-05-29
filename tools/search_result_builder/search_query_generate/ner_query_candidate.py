from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SENTENCE = """A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes,
where each axis has a label word at both ends.
Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"""


@dataclass
class EntityCandidate:
    """
    儲存由 spaCy NER 抽出的實體候選。

    Args:
        - text: 實體文字。
        - label: spaCy NER 實體類型，compound entity 會標記為 COMPOUND。
        - start: 實體在原始問題中的起始字元位置。
        - end: 實體在原始問題中的結束字元位置。
        - source: 實體來源，例如 spacy 或 spacy_compound。
        - score: 根據實體類型與文字特徵計算的排序分數。
        - parts: compound entity 由哪些原始 spaCy entities 合成。

    Returns:
        - EntityCandidate: 可用於 query 組合的實體候選。
    """

    text: str
    label: str
    start: int = -1
    end: int = -1
    source: str = "spacy"
    score: float = 0.0
    parts: list[str] = field(default_factory=list)


@dataclass
class NerQueryCandidate:
    """
    儲存由 spaCy NER 實體組成的搜尋 query 候選。

    Args:
        - query: 可直接送入搜尋引擎的 query。
        - entities: 組成 query 的實體文字。
        - reason: 產生此 query 的原因。
        - score: query 的啟發式排序分數。

    Returns:
        - NerQueryCandidate: NER-based query candidate。
    """

    query: str
    entities: list[str] = field(default_factory=list)
    reason: str = ""
    score: float = 0.0


class NerQueryCandidateGenerator:
    """
    使用 spaCy NER 抽取問題中的實體，並補上 generic compound entity candidates。

    Args:
        - spacy_model: spaCy 模型名稱，例如 en_core_web_md。

    Returns:
        - NerQueryCandidateGenerator: 可抽取 entities 並產生 query candidates 的工具。
    """

    CONNECTIVE_WORDS = {"and", "of", "for", "in", "on", "the", "at", "by", "to", "&"}
    MERGEABLE_LABELS = {
        "ORG",
        "WORK_OF_ART",
        "EVENT",
        "PERSON",
        "GPE",
        "LOC",
        "FAC",
        "LAW",
        "NORP",
        "PRODUCT",
    }
    MAX_COMPOUND_CHARS = 20
    MAX_COMPOUND_WORDS = 8

    LABEL_SCORE = {
        "COMPOUND": 0.95,
        "DATE": 1.0,
        "TIME": 0.9,
        "ORG": 0.9,
        "WORK_OF_ART": 0.9,
        "EVENT": 0.85,
        "PERSON": 0.8,
        "GPE": 0.8,
        "LOC": 0.75,
        "PRODUCT": 0.75,
        "NORP": 0.7,
        "FAC": 0.7,
        "LAW": 0.7,
        "LANGUAGE": 0.6,
        "CARDINAL": 0.45,
        "ORDINAL": 0.45,
        "QUANTITY": 0.45,
        "PERCENT": 0.45,
        "MONEY": 0.45,
    }

    def __init__(self, *, spacy_model: str = "en_core_web_md") -> None:
        self.spacy_model = spacy_model
        self._nlp = self._load_spacy_model()

    def extract_entities(self, question: str) -> list[EntityCandidate]:
        """
        使用 spaCy 從問題文字抽取 NER 實體，並補上相鄰結構形成的 compound entities。

        Args:
            - question: 原始問題。

        Returns:
            - list[EntityCandidate]: 依照重要性排序後的實體候選。
        """
        entities = self._extract_spacy_entities(question)
        compound_entities = self._merge_compound_entities(question, entities)
        return self._dedupe_entities(entities + compound_entities)

    def generate(self, question: str, *, num_candidates: int = 8) -> list[NerQueryCandidate]:
        """
        根據 spaCy NER 實體產生搜尋 query 候選。

        Args:
            - question: 原始問題。
            - num_candidates: 最多輸出的 query candidate 數量。

        Returns:
            - list[NerQueryCandidate]: 依照分數排序後的 query candidates。
        """
        entities = self.extract_entities(question)
        candidates: list[NerQueryCandidate] = []
        top_entities = entities[:8]

        if top_entities:
            candidates.append(self._combined_query(top_entities[:4]))

        for entity in top_entities:
            candidates.append(
                NerQueryCandidate(
                    query=entity.text,
                    entities=[entity.text],
                    reason=f"spaCy 單一實體: {entity.label}",
                    score=entity.score,
                )
            )

        candidates.extend(self._pair_queries(top_entities))
        candidates.extend(self._label_group_queries(top_entities))

        cleaned = self._dedupe_queries(candidates)
        cleaned.sort(key=lambda item: item.score, reverse=True)
        return cleaned[:num_candidates]

    def _load_spacy_model(self) -> Any:
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("找不到 spaCy，請先執行: pip install spacy") from exc

        try:
            return spacy.load(self.spacy_model)
        except OSError as exc:
            raise RuntimeError(
                f"找不到 spaCy 模型 {self.spacy_model}，請先執行: "
                f"python -m spacy download {self.spacy_model}"
            ) from exc

    def _extract_spacy_entities(self, question: str) -> list[EntityCandidate]:
        doc = self._nlp(question)
        entities: list[EntityCandidate] = []
        for ent in doc.ents:
            text = self._clean_text(ent.text)
            if not text:
                continue
            entities.append(
                EntityCandidate(
                    text=text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    source="spacy",
                    score=self._score_entity(text, ent.label_),
                    parts=[text],
                )
            )
        return self._dedupe_entities(entities)

    def _merge_compound_entities(
        self,
        question: str,
        entities: list[EntityCandidate],
    ) -> list[EntityCandidate]:
        ordered = sorted(
            [entity for entity in entities if entity.label in self.MERGEABLE_LABELS],
            key=lambda entity: (entity.start, entity.end),
        )
        compounds: list[EntityCandidate] = []

        for start_index, start_entity in enumerate(ordered):
            group = [start_entity]
            for next_entity in ordered[start_index + 1 :]:
                previous = group[-1]
                if next_entity.start < previous.end:
                    continue

                gap = question[previous.end : next_entity.start]
                if not self._is_mergeable_gap(gap):
                    break

                candidate_group = group + [next_entity]
                candidate_text = question[candidate_group[0].start : candidate_group[-1].end].strip()
                if not self._is_valid_compound(candidate_text, candidate_group):
                    break

                group = candidate_group
                compounds.append(self._build_compound_entity(candidate_text, group))

        return compounds

    def _is_mergeable_gap(self, gap: str) -> bool:
        normalized = re.sub(r"[\s,.:;()\[\]{}'\"-]+", " ", gap.lower()).strip()
        if not normalized:
            return True
        return all(word in self.CONNECTIVE_WORDS for word in normalized.split())

    def _is_valid_compound(self, text: str, group: list[EntityCandidate]) -> bool:
        if len(group) < 2:
            return False
        if len(text) > self.MAX_COMPOUND_CHARS:
            return False
        if len(text.split()) > self.MAX_COMPOUND_WORDS:
            return False
        return True

    def _build_compound_entity(self, text: str, group: list[EntityCandidate]) -> EntityCandidate:
        avg_score = sum(entity.score for entity in group) / len(group)
        return EntityCandidate(
            text=self._clean_text(text),
            label="COMPOUND",
            start=group[0].start,
            end=group[-1].end,
            source="spacy_compound",
            score=round(min(1.0, avg_score + 0.05), 3),
            parts=[entity.text for entity in group],
        )

    def _combined_query(self, entities: list[EntityCandidate]) -> NerQueryCandidate:
        query = self._join_query_terms([entity.text for entity in entities])
        score = sum(entity.score for entity in entities) / max(1, len(entities))
        return NerQueryCandidate(
            query=query,
            entities=[entity.text for entity in entities],
            reason="spaCy 高分實體組合",
            score=round(score, 3),
        )

    def _pair_queries(self, entities: list[EntityCandidate]) -> list[NerQueryCandidate]:
        candidates: list[NerQueryCandidate] = []
        for left_index, left in enumerate(entities[:6]):
            for right in entities[left_index + 1 : 8]:
                if left.label == right.label and left.label not in {"DATE", "ORG"}:
                    continue
                query = self._join_query_terms([left.text, right.text])
                if not query:
                    continue
                candidates.append(
                    NerQueryCandidate(
                        query=query,
                        entities=[left.text, right.text],
                        reason=f"spaCy 實體配對: {left.label}+{right.label}",
                        score=round((left.score + right.score) / 2, 3),
                    )
                )
        return candidates

    def _label_group_queries(self, entities: list[EntityCandidate]) -> list[NerQueryCandidate]:
        candidates: list[NerQueryCandidate] = []
        orgs = [entity for entity in entities if entity.label in {"ORG", "COMPOUND"}]
        dates = [entity for entity in entities if entity.label == "DATE"]
        works = [entity for entity in entities if entity.label in {"WORK_OF_ART", "EVENT", "LAW"}]

        for org in orgs[:3]:
            for date in dates[:3]:
                selected = [org, date]
                if works:
                    selected.insert(1, works[0])
                candidates.append(
                    NerQueryCandidate(
                        query=self._join_query_terms([entity.text for entity in selected]),
                        entities=[entity.text for entity in selected],
                        reason="spaCy ORG/DATE 約束組合",
                        score=round(sum(entity.score for entity in selected) / len(selected), 3),
                    )
                )
        return candidates

    def _dedupe_entities(self, entities: list[EntityCandidate]) -> list[EntityCandidate]:
        best_by_key: dict[str, EntityCandidate] = {}
        for entity in entities:
            key = self._normalize(entity.text)
            if not key:
                continue
            previous = best_by_key.get(key)
            if previous is None or entity.score > previous.score:
                best_by_key[key] = entity

        ordered = list(best_by_key.values())
        ordered.sort(key=lambda item: (item.score, len(item.text)), reverse=True)
        return ordered

    def _dedupe_queries(self, candidates: list[NerQueryCandidate]) -> list[NerQueryCandidate]:
        best_by_key: dict[str, NerQueryCandidate] = {}
        for candidate in candidates:
            query = self._join_query_terms([candidate.query])
            key = self._normalize(query)
            if not key:
                continue
            candidate.query = query
            previous = best_by_key.get(key)
            if previous is None or candidate.score > previous.score:
                best_by_key[key] = candidate
        return list(best_by_key.values())

    def _score_entity(self, text: str, label: str) -> float:
        base = self.LABEL_SCORE.get(label, 0.5)
        length_bonus = min(0.2, max(0.0, len(text.split()) - 1) * 0.05)
        rare_bonus = 0.1 if any(char.isdigit() for char in text) else 0.0
        return round(min(1.0, base + length_bonus + rare_bonus), 3)

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:!?()[]{}")
        return cleaned if len(cleaned) >= 2 else ""

    def _join_query_terms(self, terms: list[str]) -> str:
        cleaned_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = self._clean_text(term)
            key = self._normalize(cleaned)
            if cleaned and key not in seen:
                cleaned_terms.append(cleaned)
                seen.add(key)
        return " ".join(cleaned_terms)

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spaCy NER-based search query candidates.")
    parser.add_argument("--question", default="", help="Question text. If omitted, stdin or interactive input is used.")
    parser.add_argument("--question-file", default="", help="Read question text from a UTF-8 text file.")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--spacy-model", default="en_core_web_md")
    parser.add_argument("--show-entities", action="store_true", help="Output extracted entities instead of query candidates.")
    return parser.parse_args(argv)


def resolve_question(args: argparse.Namespace) -> str:
    """
    從 CLI、檔案、stdin 或互動輸入取得問題文字。

    Args:
        - args: CLI 參數。

    Returns:
        - str: 要分析的問題文字。
    """
    if args.question:
        return str(args.question).strip()

    if args.question_file:
        return Path(args.question_file).read_text(encoding="utf-8").strip()

    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            return stdin_text

    print("Enter question. Finish with an empty line:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)

    question = "\n".join(lines).strip()
    return question or SENTENCE


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    question = resolve_question(args)
    generator = NerQueryCandidateGenerator(spacy_model=args.spacy_model)

    if args.show_entities:
        entities = generator.extract_entities(question)
        print(json.dumps([asdict(entity) for entity in entities], ensure_ascii=False, indent=2))
        return

    candidates = generator.generate(question, num_candidates=args.num_candidates)
    print(json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
