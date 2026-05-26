"""deterministic solver 共用工具。"""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction
from typing import Any


NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def clean_text(value: Any) -> str:
    """清理文字空白。"""
    return " ".join(str(value or "").split()).strip()


def lower_text(value: Any) -> str:
    """清理文字並轉成小寫。"""
    return clean_text(value).lower()


def extract_numbers(text: str) -> list[Decimal]:
    """抽取文字中的數字。"""
    numbers: list[Decimal] = []
    for match in NUMBER_RE.finditer(text):
        try:
            numbers.append(Decimal(match.group(0).replace(",", "")))
        except Exception:
            continue
    return numbers


def format_decimal(value: Decimal | int | float | Fraction) -> str:
    """格式化數值，避免多餘的小數零。"""
    if isinstance(value, Fraction):
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    decimal_value = Decimal(str(value))
    if decimal_value == decimal_value.to_integral():
        return str(decimal_value.quantize(Decimal("1")))
    return format(decimal_value.normalize(), "f")


def split_items(text: str) -> list[str]:
    """把清單文字切成項目。"""
    text = clean_text(text).strip(" .")
    if not text:
        return []
    if "\n" in text:
        items = [part.strip(" -\t") for part in text.splitlines()]
    elif ";" in text:
        items = [part.strip() for part in text.split(";")]
    else:
        items = [part.strip() for part in text.split(",")]
    return [item for item in items if item]


def extract_quoted(text: str) -> list[str]:
    """抽取引號中的文字。"""
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', str(text or ""))
    return [double_quoted or single_quoted for double_quoted, single_quoted in matches if double_quoted or single_quoted]


def first_quoted_value(text: str) -> str:
    """回傳第一個引號文字。"""
    values = extract_quoted(text)
    return values[0] if values else ""
