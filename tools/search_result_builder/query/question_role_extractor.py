from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from .semantic_impact import SemanticImpactScorer


@dataclass
class QuestionContext:
    """
    主問句定位階段產生的 sentence / clause context。

    Args:
        - text: context 文字。
        - start: context 在原始 question 的起始位置。
        - end: context 在原始 question 的結束位置。
        - source: context 來源。
        - has_valid_question_head: 是否包含有效 question head。
        - has_answer_instruction: 是否為答案格式或答案輸出指令。
        - selected_reason: 被選為 main context 的原因。

    Returns:
        - QuestionContext: 可供 QuestionRoleExtractor 使用的主問句 context。
    """

    text: str
    start: int
    end: int
    source: str = "context"
    has_valid_question_head: bool = False
    has_answer_instruction: bool = False
    selected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionRoleCandidate:
    """
    問題開頭或疑問語意附近的候選 answer-role span。

    Args:
        - text: 候選文字。
        - start: 候選在原始問題中的起始位置。
        - end: 候選在原始問題中的結束位置。
        - source: 候選來源。

    Returns:
        - QuestionRoleCandidate: 可供語意分類的候選片段。
    """

    text: str
    start: int
    end: int
    source: str = "question_head"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionRole:
    """
    原始問題的答案角色與答案目標摘要。

    Args:
        - head_span: 最能表示問題問法的片段。
        - answer_role: 語意空間判斷出的答案角色。
        - answer_target: 去除疑問詞後的答案目標片段。
        - confidence: 最佳角色與第二佳角色的相似度差。
        - role_scores: 各答案角色的相似度。
        - candidate_spans: 曾被評估的候選片段。
        - main_context: 主問句定位結果。

    Returns:
        - QuestionRole: 給 span role classification 使用的問題角色。
    """

    head_span: str = ""
    answer_role: str = "unknown"
    answer_target: str = ""
    confidence: float = 0.0
    role_scores: dict[str, float] = field(default_factory=dict)
    candidate_spans: list[dict[str, Any]] = field(default_factory=list)
    main_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuestionRoleExtractor:
    """
    先以分層 context priority 定位主問句，再用 embedding similarity 判斷答案角色。

    Args:
        - scorer: 共用的 encoder embedding scorer。
        - max_candidates: 最多評估的 question-head 候選數。
        - max_window_tokens: WH 片段附近最多保留的 token 數。

    Returns:
        - QuestionRoleExtractor: 輕量 answer-role 抽取器。
    """

    ROLE_PROTOTYPES: dict[str, str] = {
        "count": "asking for a count, number, total, quantity, or how many items",
        "measurement": (
            "asking for a measured value such as volume, distance, size, duration, "
            "speed, area, amount, or value with units"
        ),
        "person": "asking for a person, author, writer, actor, speaker, sender, recipient, or name",
        "location": "asking for a place, location, city, country, setting, site, or geographic answer",
        "title": "asking for the title, name, episode, paper, book, song, film, work, or document",
        "date": "asking for a date, year, time, publication date, day, month, or period",
        "boolean": "asking for yes or no, true or false, whether something is possible",
        "choice": "asking which option, object, item, ball, route, or selected choice",
        "text_span": "asking for exact text, word, phrase, string, label, code, or wording",
        "unknown": "asking for an unspecified short answer",
    }
    ROLE_ORDER = [
        "count",
        "measurement",
        "person",
        "location",
        "title",
        "date",
        "boolean",
        "choice",
        "text_span",
        "unknown",
    ]
    WH_RE = re.compile(
        r"\b(how\s+many|how\s+much|what|which|who|whom|whose|where|when|whether|can|could|is|are|was|were)\b",
        flags=re.IGNORECASE,
    )
    ANSWER_TARGET_PREFIX_RE = re.compile(
        r"^(how\s+many|how\s+much|what|which|who|whom|whose|where|when|whether|can|could|is|are|was|were)\b\s*",
        flags=re.IGNORECASE,
    )
    AUXILIARY_RE = re.compile(
        r"^(is|are|was|were|be|been|being|do|does|did|can|could|would|should|will|the|a|an)\b\s*",
        flags=re.IGNORECASE,
    )
    HOW_TARGET_RE = re.compile(
        r"\b(how\s+many|how\s+much)\s+([A-Za-z0-9][A-Za-z0-9'^\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'^\-]*){0,5})",
        flags=re.IGNORECASE,
    )
    HEAD_STOP_RE = re.compile(
        r"\b(was|were|is|are|be|been|being|do|does|did|would|should|will|published|authored|written|called|located|has|had|have|take)\b.*$",
        flags=re.IGNORECASE,
    )
    ANSWER_INSTRUCTION_RE = re.compile(
        r"^\s*(give|return|provide|please\s+provide|please\s+use|round|do\s+not\s+use|for\s+this\s+question|based\s+on)\b",
        flags=re.IGNORECASE,
    )
    CLAUSE_BOUNDARY_RE = re.compile(
        r"(\?|;|\n+|(?=\b(?:Give|Return|Provide|Please provide|Please use|Round|Do not use|For this question|Based on)\b))",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        scorer: SemanticImpactScorer,
        max_candidates: int = 8,
        max_window_tokens: int = 9,
    ) -> None:
        self.scorer = scorer
        self.max_candidates = max(1, max_candidates)
        self.max_window_tokens = max(3, max_window_tokens)
        self._nlp: Any | None = None
        self._nlp_load_attempted = False

    def extract(self, question: str) -> QuestionRole:
        """
        從原始問題抽出答案角色。

        Args:
            - question: 原始任務問題。

        Returns:
            - QuestionRole: 問題角色與答案目標。
        """
        text = normalize_text(question)
        if not text:
            return QuestionRole()

        main_context = self.locate_main_question_context(text)
        candidates = self.extract_candidates(main_context.text, base_offset=main_context.start)
        if not candidates:
            candidates = [
                QuestionRoleCandidate(
                    text=main_context.text[:120],
                    start=main_context.start,
                    end=min(main_context.end, main_context.start + 120),
                    source="main_context_fallback",
                )
            ]

        prototypes = [self.ROLE_PROTOTYPES[role] for role in self.ROLE_ORDER]
        best_candidate = candidates[0]
        best_role = "unknown"
        best_confidence = 0.0
        best_scores: dict[str, float] = {}
        best_value = float("-inf")

        for candidate in candidates[: self.max_candidates]:
            relation_text = self._role_relation_text(
                question=text,
                main_context=main_context.text,
                candidate=candidate,
            )
            try:
                similarities = self.scorer.semantic_similarities(relation_text, prototypes)
            except Exception:
                similarities = [0.0 for _ in prototypes]
            score_map = self._question_role_scores(candidate, similarities)
            role, confidence, value = self._best_role(score_map)
            if value > best_value:
                best_candidate = candidate
                best_role = role
                best_confidence = confidence
                best_scores = score_map
                best_value = value

        return QuestionRole(
            head_span=best_candidate.text,
            answer_role=best_role,
            answer_target=self._answer_target_from_head(best_candidate.text),
            confidence=best_confidence,
            role_scores=best_scores,
            candidate_spans=[candidate.to_dict() for candidate in candidates[: self.max_candidates]],
            main_context=main_context.to_dict(),
        )

    def locate_main_question_context(self, question: str) -> QuestionContext:
        """
        用固定優先序定位主問句 context，不使用加權分數。

        Args:
            - question: 原始任務問題。

        Returns:
            - QuestionContext: 被選中的主問句或 fallback 全文 context。
        """
        text = normalize_text(question)
        if not text:
            return QuestionContext(text="", start=0, end=0, source="empty")

        clauses = self.segment_clauses(self.segment_sentences(text))
        if not clauses:
            return QuestionContext(
                text=text,
                start=0,
                end=len(text),
                source="fallback_full_question",
                selected_reason="no_clause_found",
            )

        valid_indices = [index for index, clause in enumerate(clauses) if clause.has_valid_question_head]
        if valid_indices:
            index = valid_indices[-1]
            selected = self._merge_following_answer_instructions(
                clauses,
                index,
                selected_reason="last_valid_question_head_with_following_instruction",
            )
            if selected.text:
                return selected

        if valid_indices:
            index = valid_indices[0]
            selected = self._merge_following_answer_instructions(
                clauses,
                index,
                selected_reason="first_valid_question_head_with_following_instruction",
            )
            if selected.text:
                return selected

        instruction_indices = [
            index for index, clause in enumerate(clauses) if clause.has_answer_instruction
        ]
        if instruction_indices:
            clause = clauses[instruction_indices[-1]]
            return QuestionContext(
                text=clause.text,
                start=clause.start,
                end=clause.end,
                source=clause.source,
                has_valid_question_head=clause.has_valid_question_head,
                has_answer_instruction=True,
                selected_reason="last_answer_instruction",
            )

        return QuestionContext(
            text=text,
            start=0,
            end=len(text),
            source="fallback_full_question",
            selected_reason="no_question_head_or_instruction",
        )

    def segment_sentences(self, question: str) -> list[QuestionContext]:
        """
        優先使用 spaCy 句子切分；失敗時只用問號、驚嘆號和換行 fallback。

        Args:
            - question: 原始任務問題。

        Returns:
            - list[QuestionContext]: sentence contexts。
        """
        text = normalize_text(question)
        doc = self._spacy_doc(text)
        if doc is not None:
            sentences: list[QuestionContext] = []
            for sent in doc.sents:
                start = int(sent.start_char)
                end = int(sent.end_char)
                segment = normalize_text(text[start:end]).strip()
                if segment:
                    sentences.append(
                        QuestionContext(
                            text=segment,
                            start=start,
                            end=end,
                            source="spacy_sentence",
                        )
                    )
            if sentences:
                return sentences

        return self._fallback_sentence_segments(text)

    def segment_clauses(self, sentences: list[QuestionContext]) -> list[QuestionContext]:
        """
        只在安全邊界切 clause，避免任意用逗號或句號切壞專名。

        Args:
            - sentences: sentence contexts。

        Returns:
            - list[QuestionContext]: clause contexts。
        """
        clauses: list[QuestionContext] = []
        for sentence in sentences:
            for start, end in self._safe_clause_ranges(sentence.text):
                absolute_start = sentence.start + start
                absolute_end = sentence.start + end
                segment = normalize_text(sentence.text[start:end]).strip(" \t\r\n")
                if not segment:
                    continue
                clause = QuestionContext(
                    text=segment,
                    start=absolute_start,
                    end=absolute_end,
                    source=f"{sentence.source}:clause",
                )
                clause.has_valid_question_head = self._context_has_valid_question_head(clause)
                clause.has_answer_instruction = self.is_answer_instruction(clause)
                clauses.append(clause)
        return clauses

    def is_answer_instruction(self, context: QuestionContext) -> bool:
        """
        判斷 context 是否是答案輸出或格式指令。

        Args:
            - context: sentence / clause context。

        Returns:
            - bool: 若為答案指令則 True。
        """
        return self.ANSWER_INSTRUCTION_RE.search(context.text) is not None

    def extract_candidates(
        self,
        question: str,
        *,
        base_offset: int = 0,
    ) -> list[QuestionRoleCandidate]:
        """
        從已定位的 main context 建立 question-head 候選，不直接決定角色。

        Args:
            - question: main question context。
            - base_offset: context 在原始 question 的起始 offset。

        Returns:
            - list[QuestionRoleCandidate]: 候選疑問語意片段。
        """
        text = normalize_text(question)
        candidates: list[QuestionRoleCandidate] = []
        candidates.extend(self._how_target_candidates(text, base_offset=base_offset))
        candidates.extend(self._wh_window_candidates(text, base_offset=base_offset))
        candidates.extend(self._spacy_head_candidates(text, base_offset=base_offset))
        return self._dedupe_candidates(candidates)[: self.max_candidates]

    def _how_target_candidates(
        self,
        question: str,
        *,
        base_offset: int = 0,
    ) -> list[QuestionRoleCandidate]:
        candidates: list[QuestionRoleCandidate] = []
        for match in self.HOW_TARGET_RE.finditer(question):
            if not self._valid_question_head_match(question, match):
                continue
            segment = normalize_text(match.group(0)).strip(" ,.;:!?")
            prefix_match = re.match(r"^(how\s+many|how\s+much)\s+", segment, flags=re.IGNORECASE)
            if prefix_match:
                prefix = prefix_match.group(0)
                rest = self.HEAD_STOP_RE.sub("", segment[len(prefix) :]).strip(" ,.;:!?")
                if rest:
                    segment = normalize_text(prefix + rest).strip(" ,.;:!?")
            if segment:
                candidates.append(
                    QuestionRoleCandidate(
                        text=segment,
                        start=base_offset + int(match.start()),
                        end=base_offset + int(match.start()) + len(segment),
                        source="how_target",
                    )
                )
            break
        return candidates

    def _wh_window_candidates(
        self,
        question: str,
        *,
        base_offset: int = 0,
    ) -> list[QuestionRoleCandidate]:
        candidates: list[QuestionRoleCandidate] = []
        for match in self._iter_question_wh_matches(question):
            start = int(match.start())
            tail = question[start:]
            tokens = list(re.finditer(r"\S+", tail))
            if not tokens:
                continue
            for window in (3, 5, self.max_window_tokens):
                selected = tokens[: min(window, len(tokens))]
                end = start + int(selected[-1].end())
                segment = normalize_text(question[start:end]).strip(" ,.;:!?")
                if segment:
                    candidates.append(
                        QuestionRoleCandidate(
                            text=segment,
                            start=base_offset + start,
                            end=base_offset + end,
                            source=f"wh_window_{window}",
                        )
                    )
            break
        return candidates

    def _spacy_head_candidates(
        self,
        question: str,
        *,
        base_offset: int = 0,
    ) -> list[QuestionRoleCandidate]:
        doc = self._spacy_doc(question)
        if doc is None:
            return []
        wh_match = next(self._iter_question_wh_matches(question), None)
        if wh_match is None:
            return []
        wh_start = int(wh_match.start())
        max_end = min(len(question), wh_start + 180)
        candidates: list[QuestionRoleCandidate] = []
        try:
            chunks = list(doc.noun_chunks)
        except Exception:
            chunks = []
        for chunk in chunks:
            start = int(chunk.start_char)
            end = int(chunk.end_char)
            if start < wh_start or start > max_end:
                continue
            text = normalize_text(question[start:end]).strip(" ,.;:!?")
            if text:
                candidates.append(
                    QuestionRoleCandidate(
                        text=text,
                        start=base_offset + start,
                        end=base_offset + end,
                        source="question_head_noun_chunk",
                    )
                )
        return candidates

    def _role_relation_text(
        self,
        *,
        question: str,
        main_context: str,
        candidate: QuestionRoleCandidate,
    ) -> str:
        return normalize_text(
            "Full question: "
            + question
            + "\nMain question context: "
            + main_context
            + "\nQuestion head candidate: "
            + candidate.text
            + "\nThis candidate expresses what kind of answer the question asks for."
        )

    def _question_role_scores(
        self,
        candidate: QuestionRoleCandidate,
        similarities: list[float],
    ) -> dict[str, float]:
        scores = {
            role: float(score)
            for role, score in zip(self.ROLE_ORDER, similarities, strict=False)
        }
        lowered = normalize_text(candidate.text).lower()
        if lowered.startswith("how many"):
            scores["count"] += 0.12
        if lowered.startswith("how much"):
            scores["measurement"] += 0.10
        if lowered.startswith("who") or lowered.startswith("whom") or lowered.startswith("whose"):
            scores["person"] += 0.08
        if lowered.startswith("where"):
            scores["location"] += 0.08
        if lowered.startswith("when"):
            scores["date"] += 0.08
        if lowered.startswith("which"):
            scores["choice"] += 0.08
        if lowered.startswith(("whether", "can ", "could ", "is ", "are ", "was ", "were ")):
            scores["boolean"] += 0.06
        if any(term in lowered for term in ["highest number", "number of", "count", "total"]):
            scores["count"] += 0.06
        if any(term in lowered for term in ["volume", "distance", "duration", "size", "hours", "m^3", "speed"]):
            scores["measurement"] += 0.06
        if any(term in lowered for term in ["location", "setting", "place", "site"]):
            scores["location"] += 0.05
        if any(term in lowered for term in ["title", "called", "name", "script", "paper"]):
            scores["title"] += 0.05
        if any(term in lowered for term in ["exact", "word", "phrase", "text", "string", "label"]):
            scores["text_span"] += 0.05
        if candidate.source.startswith(("how_target", "wh_window")):
            for role in self.ROLE_ORDER:
                scores[role] += 0.015
        if candidate.source == "how_target":
            scores["count"] += 0.06
            scores["measurement"] += 0.025
            for role in self.ROLE_ORDER:
                scores[role] += 0.06
        if candidate.source == "question_head_noun_chunk":
            scores["unknown"] -= 0.02
        return {role: round(score, 6) for role, score in scores.items()}

    def _best_role(
        self,
        score_map: dict[str, float],
    ) -> tuple[str, float, float]:
        pairs = [(role, round(float(score), 6)) for role, score in score_map.items()]
        pairs.sort(key=lambda item: item[1], reverse=True)
        best_role, best_score = pairs[0] if pairs else ("unknown", 0.0)
        second_score = pairs[1][1] if len(pairs) > 1 else 0.0
        confidence = round(max(0.0, best_score - second_score), 6)
        return best_role, confidence, best_score

    def _answer_target_from_head(self, head_span: str) -> str:
        target = normalize_text(head_span)
        target = self.ANSWER_TARGET_PREFIX_RE.sub("", target).strip(" ,.;:!?")
        for _ in range(3):
            updated = self.AUXILIARY_RE.sub("", target).strip(" ,.;:!?")
            if updated == target:
                break
            target = updated
        return target or normalize_text(head_span)

    def _merge_following_answer_instructions(
        self,
        clauses: list[QuestionContext],
        index: int,
        *,
        selected_reason: str,
    ) -> QuestionContext:
        selected = clauses[index]
        end_index = index
        while end_index + 1 < len(clauses) and clauses[end_index + 1].has_answer_instruction:
            end_index += 1
        if end_index == index:
            return QuestionContext(
                text=selected.text,
                start=selected.start,
                end=selected.end,
                source=selected.source,
                has_valid_question_head=selected.has_valid_question_head,
                has_answer_instruction=selected.has_answer_instruction,
                selected_reason=selected_reason,
            )
        merged_text = normalize_text(
            " ".join(clause.text for clause in clauses[index : end_index + 1])
        )
        return QuestionContext(
            text=merged_text,
            start=clauses[index].start,
            end=clauses[end_index].end,
            source="merged_question_context",
            has_valid_question_head=True,
            has_answer_instruction=True,
            selected_reason=selected_reason,
        )

    def _fallback_sentence_segments(self, question: str) -> list[QuestionContext]:
        ranges = self._split_keep_boundaries(question, re.compile(r"(?<=[?!])\s+|\n+"))
        return [
            QuestionContext(
                text=normalize_text(question[start:end]).strip(),
                start=start,
                end=end,
                source="fallback_sentence",
            )
            for start, end in ranges
            if normalize_text(question[start:end]).strip()
        ]

    def _safe_clause_ranges(self, text: str) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        for match in self.CLAUSE_BOUNDARY_RE.finditer(text):
            boundary_start = int(match.start())
            boundary_end = int(match.end())
            if match.group(0) == "?" and not self._is_sentence_question_mark(text, boundary_start):
                continue
            if boundary_start > start:
                ranges.append((start, boundary_end if match.group(0) in {"?", ";"} else boundary_start))
            if match.group(0) in {"?", ";"}:
                start = boundary_end
            else:
                start = boundary_start
        if start < len(text):
            ranges.append((start, len(text)))
        return self._dedupe_ranges(ranges, text)

    def _is_sentence_question_mark(self, text: str, index: int) -> bool:
        prefix = text[:index]
        if prefix.count('"') % 2 == 1 or prefix.count("“") > prefix.count("”"):
            return False
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if next_char and not next_char.isspace() and next_char not in {'"', "'", ")", "]"}:
            return False
        return True

    def _split_keep_boundaries(self, text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        for match in pattern.finditer(text):
            end = int(match.start())
            if end > start:
                ranges.append((start, end))
            start = int(match.end())
        if start < len(text):
            ranges.append((start, len(text)))
        return self._dedupe_ranges(ranges, text)

    def _dedupe_ranges(self, ranges: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
        output: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for start, end in ranges:
            start = max(0, min(len(text), start))
            end = max(start, min(len(text), end))
            segment = normalize_text(text[start:end]).strip()
            if not segment or (start, end) in seen:
                continue
            output.append((start, end))
            seen.add((start, end))
        return output

    def _context_has_valid_question_head(self, context: QuestionContext) -> bool:
        return next(self._iter_question_wh_matches(context.text), None) is not None

    def _iter_question_wh_matches(self, question: str):
        for match in self.WH_RE.finditer(question):
            if self._valid_question_head_match(question, match):
                yield match

    def _valid_question_head_match(self, question: str, match: re.Match[str]) -> bool:
        start = int(match.start())
        token = match.group(0)
        token_lower = token.lower()
        prefix = question[:start]
        if prefix.count('"') % 2 == 1 or prefix.count("“") > prefix.count("”"):
            return False
        previous = re.search(r"([A-Za-z][A-Za-z'-]*)\s*$", prefix)
        previous_word = previous.group(1) if previous else ""
        clause_prefix = re.split(r"[?!;\n]", prefix)[-1]
        at_clause_start = not clause_prefix.strip()
        comma_clause_prefix = re.split(r",", clause_prefix)[-1]
        at_comma_clause_start = not comma_clause_prefix.strip()
        short_prefix = len(clause_prefix.strip().split()) <= 5
        auxiliary_heads = {"can", "could", "is", "are", "was", "were"}
        if token_lower in auxiliary_heads:
            return at_clause_start
        if not at_clause_start and token[:1].isupper() and previous_word[:1].isupper():
            return False
        if not at_clause_start and not at_comma_clause_start and not short_prefix:
            return False
        return True

    def _dedupe_candidates(self, candidates: list[QuestionRoleCandidate]) -> list[QuestionRoleCandidate]:
        output: list[QuestionRoleCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = normalize_text(candidate.text)
            key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
            if len(cleaned) < 2 or not key or key in seen:
                continue
            output.append(
                QuestionRoleCandidate(
                    text=cleaned,
                    start=max(0, int(candidate.start)),
                    end=max(int(candidate.start), int(candidate.end)),
                    source=candidate.source,
                )
            )
            seen.add(key)
        return output

    def _spacy_doc(self, question: str) -> Any | None:
        if self._nlp is None and not self._nlp_load_attempted:
            self._nlp_load_attempted = True
            try:
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_md")
                except Exception:
                    self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None
        if self._nlp is None:
            return None
        try:
            return self._nlp(question)
        except Exception:
            return None


__all__ = [
    "QuestionContext",
    "QuestionRole",
    "QuestionRoleCandidate",
    "QuestionRoleExtractor",
]
