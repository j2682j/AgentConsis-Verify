from decimal import Decimal, InvalidOperation
import os
import re
from typing import Any, Optional

_EMBEDDING_MODEL = None


def answer_equivalence(answer_a: str, answer_b: str) -> bool:
    """answer_equivalence 的主要實作。"""
    math_a = extract_math_answer(answer_a)
    math_b = extract_math_answer(answer_b)

    if math_a is not None and math_b is not None:
        return math_a == math_b

    type_a = detect_answer_type(answer_a)
    type_b = detect_answer_type(answer_b)

    if type_a != type_b:
        return normalize_for_exact(answer_a) == normalize_for_exact(answer_b)

    if type_a == "choice":
        return extract_choice_answer(answer_a) == extract_choice_answer(answer_b)

    if type_a == "math":
        return extract_math_answer(answer_a) == extract_math_answer(answer_b)

    info_a = extract_key_info(answer_a)
    info_b = extract_key_info(answer_b)

    cheap_result = cheap_key_match(info_a, info_b)
    if cheap_result is not None:
        return cheap_result

    vector_result = vector_semantic_equivalence(answer_a, answer_b)
    if vector_result is not None:
        return vector_result

    return False


def vector_semantic_equivalence(
    answer_a: str,
    answer_b: str,
    *,
    threshold: float | None = None,
) -> Optional[bool]:
    """Return vector-based semantic equivalence when embeddings are available."""
    text_a = normalize_for_embedding(answer_a)
    text_b = normalize_for_embedding(answer_b)
    if not text_a or not text_b:
        return None
    if text_a == text_b:
        return True

    try:
        score = semantic_similarity_score(text_a, text_b)
    except Exception:
        return None

    if score is None:
        return None

    cutoff = threshold
    if cutoff is None:
        cutoff = float(os.getenv("ANSWER_EQUIVALENCE_VECTOR_THRESHOLD", "0.82"))
    return score >= cutoff


def semantic_similarity_score(answer_a: str, answer_b: str) -> float | None:
    """Compute cosine similarity using the configured local embedding model."""
    model = _get_embedding_model()
    if model is None:
        return None

    embeddings = model.encode(
        [answer_a, answer_b],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return float(embeddings[0] @ embeddings[1])


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    model_name = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    except Exception:
        _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def normalize_for_embedding(text: Any) -> str:
    normalized = normalize_text(text).lower()
    normalized = normalized.strip(" \t\r\n\"'`")
    normalized = re.sub(r"^(answer|final answer|final_answer)\s*[:=]\s*", "", normalized)
    return normalized.strip()


def normalize_text(text: Any) -> str:
    """normalize_text 的主要實作。"""
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("：", ":")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_exact(text: Any) -> str:
    """normalize_for_exact 的主要實作。"""
    text = normalize_text(text).lower()
    text = text.strip(" \t\r\n\"'`")
    return text


def extract_choice_answer(text: Any) -> Optional[str]:
    """extract_choice_answer 的主要實作。"""
    text = normalize_text(text)

    direct = re.fullmatch(r"\(?([A-D])\)?", text, re.IGNORECASE)
    if direct:
        return direct.group(1).upper()

    labeled_patterns = [
        r"(?:answer|final answer|final_answer)\s*[:=]\s*\(?([A-D])\)?",
        r"\boption\s+([A-D])\b",
        r"\bchoice\s+([A-D])\b",
    ]
    for pattern in labeled_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def extract_math_answer(text: Any) -> Optional[str]:
    """extract_math_answer 的主要實作。"""
    text = normalize_text(text)

    patterns = [
        r"(?:answer|final answer|final_answer)\s*[:=]\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
        r"\\boxed\{([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\}",
        r"=\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*$",
        r"\b([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return normalize_number(m.group(1))

    nums = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    if len(nums) == 1:
        return normalize_number(nums[0])

    return None


def normalize_number(value: str) -> str:
    """normalize_number 的主要實作。"""
    try:
        dec = Decimal(str(value).replace(",", ""))
        normalized = format(dec, "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized or "0"
    except (InvalidOperation, ValueError):
        return value.strip()


def detect_answer_type(text: Any) -> str:
    """detect_answer_type 的主要實作。"""
    if extract_choice_answer(text) is not None:
        return "choice"
    if extract_math_answer(text) is not None:
        return "math"
    return "free_form"


def extract_key_info(text: Any) -> dict[str, Any]:
    """extract_key_info 的主要實作。"""
    text = normalize_text(text)

    lower = text.lower()
    lower = re.sub(r"[^\w\s:/.-]", " ", lower)
    tokens = [t for t in lower.split() if len(t) > 1]

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "that",
        "this",
        "it",
        "as",
        "with",
        "by",
        "from",
        "answer",
        "final",
        "therefore",
        "so",
        "result",
    }
    keywords = [t for t in tokens if t not in stopwords]

    numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    dates = re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text)

    return {
        "normalized_text": normalize_for_exact(text),
        "keywords": sorted(set(keywords))[:20],
        "numbers": [normalize_number(n) for n in numbers],
        "dates": dates,
    }


def cheap_key_match(info_a: dict[str, Any], info_b: dict[str, Any]) -> Optional[bool]:
    """cheap_key_match 的主要實作。"""
    if info_a["normalized_text"] == info_b["normalized_text"]:
        return True

    if (
        info_a["numbers"]
        and info_b["numbers"]
        and info_a["numbers"] == info_b["numbers"]
    ):
        kw_a = set(info_a["keywords"])
        kw_b = set(info_b["keywords"])
        if not kw_a or not kw_b or len(kw_a & kw_b) >= 1:
            return True

    kw_a = set(info_a["keywords"])
    kw_b = set(info_b["keywords"])
    if kw_a and kw_b:
        overlap = len(kw_a & kw_b)
        union = len(kw_a | kw_b)
        if union > 0 and overlap / union >= 0.8:
            return True
        if overlap == 0 and info_a["numbers"] != info_b["numbers"]:
            return False

    return None


def should_use_calculator(question: str) -> bool:
    """should_use_calculator 的主要實作。"""
    text = normalize_text(question).lower()

    math_keywords = [
        "calculate",
        "compute",
        "solve",
        "math",
        "percentage",
        "percent",
        "sum",
        "difference",
        "product",
        "quotient",
        "sqrt",
        "square root",
    ]

    has_math_keyword = any(keyword in text for keyword in math_keywords)
    has_operator = any(op in text for op in ["+", "-", "*", "/", "=", "(", ")"])
    number_count = len(re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text))

    if has_operator and number_count >= 2:
        return True

    if has_math_keyword and number_count >= 1:
        return True

    return False


def should_use_search(question: str) -> bool:
    """should_use_search 的主要實作。"""
    text = normalize_text(question).lower()

    search_keywords = [
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "difference",
        "compare",
        "explain",
        "history",
        "capital",
        "country",
        "language",
        "learn",
    ]

    if should_use_calculator(question):
        word_problem_markers = [
            "how much",
            "how many",
            "total",
            "remainder",
            "left",
            "each",
            "every",
            "per day",
            "per hour",
            "per item",
            "cost",
            "price",
            "earn",
            "make",
            "dollars",
            "$",
            "sold",
            "sell",
            "buys",
            "spent",
            "remaining",
        ]
        if any(marker in text for marker in word_problem_markers):
            return False

    if any(keyword in text for keyword in search_keywords):
        return True

    if not should_use_calculator(question):
        return True

    return False
