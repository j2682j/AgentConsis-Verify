"""
Conservative answer matching for GAIA benchmark reports.

This module is intentionally stricter than the system-level answer
equivalence used for agent consensus. Benchmark evaluation should avoid
embedding or semantic similarity because short GAIA answers are often
format-sensitive lists, names, codes, or exact labels.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any


_LABEL_RE = re.compile(
    r"^\s*(?:final\s*answer|final_answer|answer)\s*[:=]\s*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
_THOUSANDS_NUMBER_RE = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
)


def exact_match(predicted: Any, expected: Any) -> bool:
    """Return True only for format-safe GAIA exact matches."""
    pred = clean_answer(predicted)
    exp = clean_answer(expected)
    if not pred or not exp:
        return False

    if should_compare_as_list(pred, exp):
        return normalize_list(pred) == normalize_list(exp)

    if normalize_scalar(pred) == normalize_scalar(exp):
        return True

    pred_yes_no = normalize_yes_no(pred)
    exp_yes_no = normalize_yes_no(exp)
    if pred_yes_no and pred_yes_no == exp_yes_no:
        return True

    pred_num = normalize_numeric_scalar(pred)
    exp_num = normalize_numeric_scalar(exp)
    if pred_num is not None and pred_num == exp_num:
        return True

    return False


def partial_match(predicted: Any, expected: Any) -> bool:
    """Return a conservative diagnostic partial match."""
    if exact_match(predicted, expected):
        return True

    pred = clean_answer(predicted)
    exp = clean_answer(expected)
    if not pred or not exp:
        return False

    if should_compare_as_list(pred, exp):
        pred_items = normalize_list(pred)
        exp_items = normalize_list(exp)
        if not pred_items or not exp_items:
            return False
        matched = sum(1 for item in exp_items if item in set(pred_items))
        return matched / len(exp_items) >= 0.7

    pred_norm = normalize_scalar(pred)
    exp_norm = normalize_scalar(exp)
    if len(exp_norm) >= 8 and exp_norm in pred_norm:
        return True
    if len(pred_norm) >= 8 and pred_norm in exp_norm:
        return True

    pred_words = set(pred_norm.split())
    exp_words = set(exp_norm.split())
    if not pred_words or not exp_words:
        return False
    return len(pred_words & exp_words) / len(exp_words) >= 0.7


def clean_answer(value: Any) -> str:
    """Remove wrappers and labels without changing answer semantics."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = text.strip()
    text = text.strip("`")
    text = re.sub(r"^\s*\*\*(.*?)\*\*\s*$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^\\boxed\{(.+)\}$", r"\1", text.strip(), flags=re.DOTALL)
    text = _LABEL_RE.sub("", text)
    text = text.strip()
    text = text.strip("[]")
    text = text.strip()
    return text


def normalize_scalar(value: Any) -> str:
    """Normalize a single non-list answer conservatively."""
    text = clean_answer(value).lower()
    text = text.replace("\uff1a", ":")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ",", text)
    text = text.rstrip(".,;:!?")

    words = text.split()
    if len(words) > 1 and words[0] in {"the", "a", "an"}:
        text = " ".join(words[1:])

    numeric = normalize_numeric_scalar(text)
    if numeric is not None:
        return numeric
    return text


def normalize_yes_no(value: Any) -> str | None:
    text = normalize_scalar(value)
    if text in {"yes", "y"}:
        return "yes"
    if text in {"no", "n"}:
        return "no"
    return None


def normalize_numeric_scalar(value: Any) -> str | None:
    text = clean_answer(value)
    if not text:
        return None
    text = text.strip()
    text = text.rstrip(".,;:!?")
    if not _NUMBER_RE.fullmatch(text):
        return None
    return normalize_number(text)


def normalize_number(value: str) -> str:
    try:
        dec = Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return value.strip().lower()
    normalized = format(dec, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def should_compare_as_list(predicted: Any, expected: Any) -> bool:
    pred = clean_answer(predicted)
    exp = clean_answer(expected)
    if "," not in pred and "," not in exp:
        return False

    pred_items = split_list(pred, force="," in exp)
    exp_items = split_list(exp, force="," in pred)
    return len(pred_items) > 1 or len(exp_items) > 1


def normalize_list(value: Any) -> list[str]:
    items = split_list(clean_answer(value), force=True)
    return [normalize_scalar(item) for item in items if normalize_scalar(item)]


def split_list(value: str, *, force: bool = False) -> list[str]:
    text = clean_answer(value)
    if not text:
        return []
    if "," not in text:
        return [text]
    if not force and _THOUSANDS_NUMBER_RE.fullmatch(text.strip()):
        return [text]
    return [part.strip() for part in text.split(",")]
