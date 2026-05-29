from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from .model_query_candidate import ModelQueryCandidateGenerator, QueryCandidate
from .ner_query_candidate import EntityCandidate, NerQueryCandidate, NerQueryCandidateGenerator
from .token_prob_compute import TextUnitScore, TokenProbabilityAnalyzer, TokenProbabilityQueryCandidate
from tools.search_tool import SearchTool


ROOT = Path(__file__).resolve().parent

SENTENCE = """A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes,
where each axis has a label word at both ends.
Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"""


@dataclass
class CombinedQueryCandidate:
    """
    儲存跨來源合併後的 query candidate。

    Args:
        - query: 合併後的 query 文字。
        - sources: 此 query 來自哪些產生器。
        - score: 合併後排序分數。
        - details: 各來源的原始資訊。

    Returns:
        - CombinedQueryCandidate: 可交給 search pipeline 使用的候選 query。
    """

    query: str
    sources: list[str]
    score: float = 0.0
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class QuerySignalBundle:
    """
    儲存一次 query 產生流程中三個來源的原始訊號。

    Args:
        - model_queries: qwen3:4b 產生的完整搜尋 query。
        - ner_queries: spaCy NER 組合出的 query。
        - ner_entities: spaCy 抽出的實體硬條件。
        - token_scores: 對 NER 實體計算出的 token probability 分數。

    Returns:
        - QuerySignalBundle: 可供多種搜尋策略共用的訊號集合。
    """

    model_queries: list[QueryCandidate]
    ner_queries: list[NerQueryCandidate]
    ner_entities: list[EntityCandidate]
    token_scores: list[TextUnitScore]


@dataclass
class SearchExperimentQuery:
    """
    儲存單一實驗策略選出的 query 與其依據。

    Args:
        - query: 要實際送入 search tool 的 query。
        - reason: 此 query 被選中的原因。
        - matched_terms: 此 query 命中的 spaCy/token 交集詞。
        - score: 策略排序分數。

    Returns:
        - SearchExperimentQuery: 一筆實驗用 query。
    """

    query: str
    reason: str
    matched_terms: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class SearchExperimentResult:
    """
    儲存單一搜尋策略的 query 選擇與實際搜尋結果。

    Args:
        - name: 實驗策略名稱。
        - description: 實驗策略說明。
        - queries: 策略選出的 query。
        - search_results: 每個 query 的搜尋結果摘要。

    Returns:
        - SearchExperimentResult: 一組可比較的搜尋策略輸出。
    """

    name: str
    description: str
    queries: list[SearchExperimentQuery]
    search_results: list[dict[str, Any]] = field(default_factory=list)


class SearchQueryCombiner:
    """
    整合 model query、spaCy NER query 與 token probability query 的候選產生流程。

    Args:
        - model_generator: qwen3:4b query candidate generator。
        - ner_generator: spaCy NER query candidate generator。
        - token_analyzer: token probability analyzer。

    Returns:
        - SearchQueryCombiner: 可產生、合併並實驗多來源 query candidates 的工具。
    """

    def __init__(
        self,
        *,
        model_generator: ModelQueryCandidateGenerator | None = None,
        ner_generator: NerQueryCandidateGenerator | None = None,
        token_analyzer: TokenProbabilityAnalyzer | None = None,
    ) -> None:
        self.model_generator = model_generator
        self.ner_generator = ner_generator
        self.token_analyzer = token_analyzer

    def collect_signals(
        self,
        question: str,
        *,
        num_model_candidates: int = 6,
        num_ner_candidates: int = 8,
        num_token_candidates: int = 8,
    ) -> QuerySignalBundle:
        """
        產生 model、spaCy NER 與 token probability 三種 query 訊號。

        Args:
            - question: 原始問題。
            - num_model_candidates: qwen3:4b 產生 query 數量。
            - num_ner_candidates: spaCy NER query 數量。
            - num_token_candidates: token probability 保留數量。

        Returns:
            - QuerySignalBundle: 三種來源的原始訊號集合。
        """
        model_generator = self.model_generator or ModelQueryCandidateGenerator()
        ner_generator = self.ner_generator or NerQueryCandidateGenerator()
        token_analyzer = self.token_analyzer or TokenProbabilityAnalyzer()

        model_queries = model_generator.generate(question, num_candidates=num_model_candidates)
        ner_entities = ner_generator.extract_entities(question)
        ner_queries = ner_generator.generate(question, num_candidates=num_ner_candidates)
        token_units = [entity.text for entity in ner_entities if entity.text in question]
        token_scores = token_analyzer.score_text_units(question, token_units)[:num_token_candidates]

        return QuerySignalBundle(
            model_queries=model_queries,
            ner_queries=ner_queries,
            ner_entities=ner_entities,
            token_scores=token_scores,
        )

    def combine(
        self,
        question: str,
        *,
        num_model_candidates: int = 6,
        num_ner_candidates: int = 8,
        num_token_candidates: int = 8,
        enable_model: bool = True,
        enable_ner: bool = True,
        enable_token_probability: bool = False,
    ) -> list[CombinedQueryCandidate]:
        """
        產生並合併多來源 query candidates。

        Args:
            - question: 原始問題。
            - num_model_candidates: model query 產生數量。
            - num_ner_candidates: NER query 產生數量。
            - num_token_candidates: token probability query 產生數量。
            - enable_model: 是否啟用 qwen3:4b query generator。
            - enable_ner: 是否啟用 spaCy NER query generator。
            - enable_token_probability: 是否啟用 token probability generator。

        Returns:
            - list[CombinedQueryCandidate]: 去重且排序後的 query candidates。
        """
        raw_candidates: list[CombinedQueryCandidate] = []

        ner_candidates: list[NerQueryCandidate] = []
        if enable_model:
            raw_candidates.extend(self._model_candidates(question, num_model_candidates))
        if enable_ner:
            ner_candidates = self._ner_candidates(question, num_ner_candidates)
            raw_candidates.extend(self._wrap_ner_candidates(ner_candidates))
        if enable_token_probability:
            text_units = self._token_text_units(ner_candidates)
            raw_candidates.extend(self._token_candidates(question, text_units, num_token_candidates))

        return self._merge_candidates(raw_candidates)

    def build_experiments(
        self,
        signals: QuerySignalBundle,
        *,
        top_k: int = 3,
    ) -> list[SearchExperimentResult]:
        """
        根據三個 query 訊號建立三組搜尋實驗。

        Args:
            - signals: collect_signals() 產生的三來源訊號。
            - top_k: 每組策略最多選出的 query 數量。

        Returns:
            - list[SearchExperimentResult]: 三組待搜尋的實驗 query。
        """
        token_terms = self._top_token_terms(signals.token_scores)
        ner_terms = [entity.text for entity in signals.ner_entities]
        supported_terms = self._intersect_terms(token_terms, ner_terms)

        return [
            SearchExperimentResult(
                name="strategy_1_model_query_with_ner_lowprob_terms",
                description=(
                    "在模型產生的完整 query 中，優先選出同時包含 spaCy entity "
                    "與低 token probability 詞的前 3 個完整 query。"
                ),
                queries=self._strategy_model_queries_ranked_by_supported_terms(
                    signals.model_queries,
                    supported_terms,
                    top_k=top_k,
                ),
            ),
            SearchExperimentResult(
                name="strategy_2_supported_terms_only",
                description=(
                    "只取同時出現在 spaCy entity 與低 token probability 的前 3 個詞，"
                    "直接作為搜尋 query。"
                ),
                queries=self._strategy_supported_terms_only(supported_terms, top_k=top_k),
            ),
            SearchExperimentResult(
                name="strategy_3_three_way_intersection_query",
                description=(
                    "找 model query、spaCy entity、低 token probability 的三方交集，"
                    "拿產生交集的完整 model query 搜尋。"
                ),
                queries=self._strategy_three_way_intersection(
                    signals.model_queries,
                    supported_terms,
                    top_k=top_k,
                ),
            ),
        ]

    def run_search_experiments(
        self,
        question: str,
        *,
        num_model_candidates: int = 6,
        num_ner_candidates: int = 8,
        num_token_candidates: int = 8,
        top_k: int = 3,
        backend: str = "searxng",
        max_results: int = 3,
    ) -> dict[str, Any]:
        """
        產生三來源 query 訊號，建立三組策略，並對每組 query 做實際搜尋。

        Args:
            - question: 原始問題。
            - num_model_candidates: qwen3:4b 產生 query 數量。
            - num_ner_candidates: spaCy NER query 數量。
            - num_token_candidates: token probability 保留數量。
            - top_k: 每組策略最多搜尋的 query 數量。
            - backend: SearchTool 使用的搜尋後端。
            - max_results: 每個 query 最多保留搜尋結果數量。

        Returns:
            - dict[str, Any]: signals、experiments 與 search results。
        """
        signals = self.collect_signals(
            question,
            num_model_candidates=num_model_candidates,
            num_ner_candidates=num_ner_candidates,
            num_token_candidates=num_token_candidates,
        )
        experiments = self.build_experiments(signals, top_k=top_k)

        searcher = SearchTool(backend=backend)
        for experiment in experiments:
            experiment.search_results = [
                self._search_query(searcher, query.query, backend=backend, max_results=max_results)
                for query in experiment.queries
            ]

        return {
            "signals": self._serialize_signals(signals),
            "experiments": [asdict(experiment) for experiment in experiments],
        }

    def _model_candidates(self, question: str, limit: int) -> list[CombinedQueryCandidate]:
        generator = self.model_generator or ModelQueryCandidateGenerator()
        candidates = generator.generate(question, num_candidates=limit)
        return [
            CombinedQueryCandidate(
                query=candidate.query,
                sources=["model"],
                score=0.8 + float(candidate.score_hint or 0.0),
                details={"model": asdict(candidate)},
            )
            for candidate in candidates
        ]

    def _ner_candidates(self, question: str, limit: int) -> list[NerQueryCandidate]:
        generator = self.ner_generator or NerQueryCandidateGenerator()
        return generator.generate(question, num_candidates=limit)

    def _wrap_ner_candidates(self, candidates: list[NerQueryCandidate]) -> list[CombinedQueryCandidate]:
        return [
            CombinedQueryCandidate(
                query=candidate.query,
                sources=["ner"],
                score=float(candidate.score or 0.0),
                details={"ner": asdict(candidate)},
            )
            for candidate in candidates
        ]

    def _token_candidates(
        self,
        question: str,
        text_units: list[str],
        limit: int,
    ) -> list[CombinedQueryCandidate]:
        if not text_units:
            return []
        analyzer = self.token_analyzer or TokenProbabilityAnalyzer()
        candidates = analyzer.generate_candidates(question, text_units, top_k=limit)
        return [
            CombinedQueryCandidate(
                query=candidate.query,
                sources=["token_probability"],
                score=float(candidate.score or 0.0),
                details={"token_probability": asdict(candidate)},
            )
            for candidate in candidates
        ]

    def _token_text_units(self, ner_candidates: list[NerQueryCandidate]) -> list[str]:
        units: list[str] = []
        for candidate in ner_candidates:
            units.extend(candidate.entities)
            units.append(candidate.query)
        return self._dedupe_text(units)

    def _strategy_model_queries_ranked_by_supported_terms(
        self,
        model_queries: list[QueryCandidate],
        supported_terms: list[str],
        *,
        top_k: int,
    ) -> list[SearchExperimentQuery]:
        ranked: list[SearchExperimentQuery] = []
        for candidate in model_queries:
            matched = self._terms_in_query(candidate.query, supported_terms)
            if not matched:
                continue
            ranked.append(
                SearchExperimentQuery(
                    query=candidate.query,
                    reason="完整 model query 命中 spaCy entity 且該 entity 屬於低機率詞",
                    matched_terms=matched,
                    score=float(len(matched)),
                )
            )
        ranked.sort(key=lambda item: (item.score, len(item.query)), reverse=True)
        return ranked[:top_k]

    def _strategy_supported_terms_only(
        self,
        supported_terms: list[str],
        *,
        top_k: int,
    ) -> list[SearchExperimentQuery]:
        return [
            SearchExperimentQuery(
                query=term,
                reason="此詞同時被 spaCy 抽出，且 token probability 排名低",
                matched_terms=[term],
                score=float(top_k - index),
            )
            for index, term in enumerate(supported_terms[:top_k])
        ]

    def _strategy_three_way_intersection(
        self,
        model_queries: list[QueryCandidate],
        supported_terms: list[str],
        *,
        top_k: int,
    ) -> list[SearchExperimentQuery]:
        selected: list[SearchExperimentQuery] = []
        used_queries: set[str] = set()
        for term in supported_terms:
            for candidate in model_queries:
                key = self._normalize(candidate.query)
                if key in used_queries or not self._contains_term(candidate.query, term):
                    continue
                selected.append(
                    SearchExperimentQuery(
                        query=candidate.query,
                        reason="此完整 model query 產生了 model/spaCy/token probability 三方交集",
                        matched_terms=[term],
                        score=float(len(supported_terms) - supported_terms.index(term)),
                    )
                )
                used_queries.add(key)
                break
            if len(selected) >= top_k:
                break
        return selected

    def _search_query(
        self,
        searcher: SearchTool,
        query: str,
        *,
        backend: str,
        max_results: int,
    ) -> dict[str, Any]:
        payload = searcher.run(
            {
                "input": query,
                "backend": backend,
                "mode": "structured",
                "max_results": max_results,
                "fetch_full_page": False,
            }
        )
        if not isinstance(payload, dict):
            return {"query": query, "backend": backend, "results": [], "raw": payload}

        results = []
        for result in payload.get("results", [])[:max_results]:
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                }
            )
        return {
            "query": query,
            "backend": payload.get("backend", backend),
            "answer": payload.get("answer"),
            "notices": payload.get("notices", []),
            "results": results,
        }

    def _serialize_signals(self, signals: QuerySignalBundle) -> dict[str, Any]:
        return {
            "model_queries": [asdict(candidate) for candidate in signals.model_queries],
            "ner_queries": [asdict(candidate) for candidate in signals.ner_queries],
            "ner_entities": [asdict(entity) for entity in signals.ner_entities],
            "token_scores": [asdict(score) for score in signals.token_scores],
        }

    def _top_token_terms(self, scores: list[TextUnitScore]) -> list[str]:
        return [score.text_unit for score in scores if score.logprob_avg is not None]

    def _intersect_terms(self, left_terms: list[str], right_terms: list[str]) -> list[str]:
        result: list[str] = []
        right_normalized = {self._normalize(term): term for term in right_terms}
        for term in left_terms:
            key = self._normalize(term)
            if key in right_normalized and key not in {self._normalize(value) for value in result}:
                result.append(right_normalized[key])
        return result

    def _terms_in_query(self, query: str, terms: list[str]) -> list[str]:
        return [term for term in terms if self._contains_term(query, term)]

    def _contains_term(self, query: str, term: str) -> bool:
        query_norm = self._normalize_for_match(query)
        term_norm = self._normalize_for_match(term)
        if not term_norm:
            return False
        return term_norm in query_norm

    def _merge_candidates(self, candidates: list[CombinedQueryCandidate]) -> list[CombinedQueryCandidate]:
        merged: dict[str, CombinedQueryCandidate] = {}
        for candidate in candidates:
            key = self._normalize(candidate.query)
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                continue
            existing.sources = self._dedupe_text(existing.sources + candidate.sources)
            existing.score = max(existing.score, candidate.score) + 0.1
            existing.details.update(candidate.details)

        ordered = list(merged.values())
        ordered.sort(key=lambda item: item.score, reverse=True)
        return ordered

    def _dedupe_text(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = self._normalize(value)
            if key and key not in seen:
                result.append(value)
                seen.add(key)
        return result

    def _normalize(self, value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _normalize_for_match(self, value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
        return f" {' '.join(cleaned.split())} "


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine and test search query candidates from multiple generators.")
    parser.add_argument("--question", default="", help="Question text. If omitted, stdin or default sentence is used.")
    parser.add_argument("--question-file", default="", help="Read question text from a UTF-8 text file.")
    parser.add_argument("--disable-model", action="store_true")
    parser.add_argument("--disable-ner", action="store_true")
    parser.add_argument("--enable-token-probability", action="store_true")
    parser.add_argument("--run-experiments", action="store_true")
    parser.add_argument("--backend", default="searxng")
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-model-candidates", type=int, default=6)
    parser.add_argument("--num-ner-candidates", type=int, default=8)
    parser.add_argument("--num-token-candidates", type=int, default=8)
    return parser.parse_args(argv)


def resolve_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question.strip()
    if args.question_file:
        return Path(args.question_file).read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            return stdin_text
    return SENTENCE


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    question = resolve_question(args)
    combiner = SearchQueryCombiner()

    if args.run_experiments:
        payload = combiner.run_search_experiments(
            question,
            num_model_candidates=args.num_model_candidates,
            num_ner_candidates=args.num_ner_candidates,
            num_token_candidates=args.num_token_candidates,
            top_k=args.top_k,
            backend=args.backend,
            max_results=args.max_results,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    candidates = combiner.combine(
        question,
        num_model_candidates=args.num_model_candidates,
        num_ner_candidates=args.num_ner_candidates,
        num_token_candidates=args.num_token_candidates,
        enable_model=not args.disable_model,
        enable_ner=not args.disable_ner,
        enable_token_probability=args.enable_token_probability,
    )
    print(json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
