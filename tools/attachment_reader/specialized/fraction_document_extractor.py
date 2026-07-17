from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image, ImageOps


@dataclass(frozen=True)
class FractionOccurrence:
    text: str
    order: int
    source: str = "literal_slash"


@dataclass(frozen=True)
class FractionProblem:
    numerator: int
    denominator: int
    order: int
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0


@dataclass
class FractionDocumentArtifact:
    literal_fractions: list[FractionOccurrence] = field(default_factory=list)
    sample_problems: list[FractionProblem] = field(default_factory=list)
    ocr_text: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "literal_fractions": [asdict(item) for item in self.literal_fractions],
            "sample_problems": [asdict(item) for item in self.sample_problems],
            "ocr_text": self.ocr_text,
            "diagnostics": dict(self.diagnostics),
        }


class FractionDocumentExtractor:
    """Extract slash fractions and vertically typeset sample fractions from an image."""

    _FRACTION_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")

    def __init__(self, *, tesseract_path: str | None = None) -> None:
        self.tesseract_path = self._resolve_tesseract(tesseract_path)

    def extract(self, file_path: str | Path) -> FractionDocumentArtifact:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        ocr_text = self._ocr(image, psm=11)
        literal = [
            FractionOccurrence(
                text=f"{match.group(1)}/{match.group(2)}",
                order=index,
            )
            for index, match in enumerate(self._FRACTION_RE.finditer(ocr_text), start=1)
        ]
        words = self._tsv_words(image, psm=11)
        sample_top = self._sample_section_top(words, image.height)
        problems, row_diagnostics = self._sample_problems(image, sample_top)
        return FractionDocumentArtifact(
            literal_fractions=literal,
            sample_problems=problems,
            ocr_text=ocr_text,
            diagnostics={
                "tesseract_path": self.tesseract_path,
                "image_size": [image.width, image.height],
                "sample_section_top": sample_top,
                "literal_fraction_count": len(literal),
                "sample_problem_count": len(problems),
                **row_diagnostics,
            },
        )

    def _sample_problems(
        self,
        image: Image.Image,
        sample_top: int,
    ) -> tuple[list[FractionProblem], dict[str, Any]]:
        anchors = self._row_anchors(image, sample_top)
        problems: list[FractionProblem] = []
        crops: list[dict[str, Any]] = []
        for order, (left, center_y) in enumerate(anchors[:20], start=1):
            x0 = max(0, left + 18)
            x1 = min(image.width, left + 56)
            y0 = max(0, center_y - 12)
            y1 = min(image.height, center_y + 24)
            crop = image.crop((x0 + 4, y0, max(x0 + 8, x1 - 4), y1))
            combined_values, raw = self._read_fraction_crop(crop)
            numerator = self._read_number_crop(
                image.crop((x0, max(0, center_y - 6), x1, min(image.height, center_y + 9))),
                allow_one=True,
            )
            denominator = self._read_number_crop(
                image.crop((x0, max(0, center_y + 11), x1, min(image.height, center_y + 27))),
                allow_one=False,
            )
            if numerator is None and len(combined_values) >= 2:
                numerator = combined_values[0]
            if denominator in {None, 0, 1}:
                remaining = [value for value in combined_values if value != numerator and value > 0]
                if remaining:
                    denominator = remaining[-1]
            values = [value for value in (numerator, denominator) if value is not None]
            crops.append({
                "order": order,
                "bbox": [x0, y0, x1, y1],
                "raw_ocr": raw,
                "values": values,
            })
            if numerator is None or denominator in {None, 0}:
                continue
            problems.append(
                FractionProblem(
                    numerator=numerator,
                    denominator=denominator,
                    order=order,
                    bbox=(x0, y0, x1, y1),
                    confidence=1.0 if len(values) == 2 else 0.75,
                )
            )
        return problems, {"row_anchor_count": len(anchors), "sample_crops": crops}

    def _row_anchors(self, image: Image.Image, sample_top: int) -> list[tuple[int, int]]:
        gray = ImageOps.grayscale(image)
        width, height = gray.size
        x_limit = max(45, min(width // 5, 180))
        pixels = gray.load()
        active: list[int] = []
        for y in range(max(0, sample_top), height):
            dark = sum(1 for x in range(8, x_limit) if pixels[x, y] < 105)
            if dark >= 2:
                active.append(y)
        bands = self._bands(active)
        candidates: list[tuple[int, int]] = []
        for start, end in bands:
            band_height = end - start + 1
            if not 5 <= band_height <= 18:
                continue
            center = (start + end) // 2
            left = self._leftmost_dark(gray, center, x_limit)
            if left < 0 or left > 45:
                continue
            candidates.append((left, center))
        result: list[tuple[int, int]] = []
        for candidate in candidates:
            if result and candidate[1] - result[-1][1] < 28:
                continue
            result.append(candidate)
        return result

    def _leftmost_dark(self, gray: Image.Image, center_y: int, x_limit: int) -> int:
        pixels = gray.load()
        for x in range(8, x_limit):
            count = sum(
                1
                for y in range(max(0, center_y - 7), min(gray.height, center_y + 8))
                if pixels[x, y] < 105
            )
            if count >= 2:
                return x
        return -1

    def _read_fraction_crop(self, crop: Image.Image) -> tuple[list[int], str]:
        enlarged = ImageOps.grayscale(crop).resize(
            (crop.width * 6, crop.height * 6),
            Image.Resampling.LANCZOS,
        )
        enlarged = ImageOps.autocontrast(enlarged)
        raw = self._ocr(
            enlarged,
            psm=6,
            extra_args=["-c", "tessedit_char_whitelist=0123456789"],
        )
        values = [int(value) for value in re.findall(r"\d+", raw)]
        return values, raw.strip()

    def _read_number_crop(self, crop: Image.Image, *, allow_one: bool) -> int | None:
        enlarged = crop.convert("L").resize(
            (crop.width * 12, crop.height * 12),
            Image.Resampling.LANCZOS,
        )
        for threshold in (180, 160, 140, 200):
            binary = enlarged.point(lambda pixel, limit=threshold: 0 if pixel < limit else 255)
            raw = self._ocr(
                binary,
                psm=10,
                extra_args=["-c", "tessedit_char_whitelist=0123456789"],
            )
            match = re.search(r"\d+", raw)
            if match:
                value = int(match.group(0))
                if value > 1 or (allow_one and value == 1):
                    return value
        return None

    def _sample_section_top(self, words: list[dict[str, Any]], height: int) -> int:
        candidates = [
            int(word["top"]) + int(word["height"])
            for word in words
            if str(word.get("text") or "").casefold().startswith("sample")
        ]
        return min(candidates) + 20 if candidates else int(height * 0.52)

    def _tsv_words(self, image: Image.Image, *, psm: int) -> list[dict[str, Any]]:
        output = self._run(image, ["--psm", str(psm), "tsv"])
        lines = output.splitlines()
        if not lines:
            return []
        headers = lines[0].split("\t")
        result = []
        for line in lines[1:]:
            values = line.split("\t")
            if len(values) != len(headers):
                continue
            row = dict(zip(headers, values, strict=False))
            if str(row.get("text") or "").strip():
                result.append(row)
        return result

    def _ocr(
        self,
        image: Image.Image,
        *,
        psm: int,
        extra_args: list[str] | None = None,
    ) -> str:
        return self._run(image, ["--psm", str(psm), *(extra_args or [])])

    def _run(self, image: Image.Image, args: list[str]) -> str:
        with tempfile.TemporaryDirectory(prefix="scp_fraction_ocr_") as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            image.save(image_path)
            completed = subprocess.run(
                [self.tesseract_path, str(image_path), "stdout", *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Tesseract OCR failed")
        return completed.stdout

    def _resolve_tesseract(self, explicit: str | None) -> str:
        candidates = [
            explicit,
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        ]
        for value in candidates:
            if value and Path(value).is_file():
                return str(value)
        raise RuntimeError("Tesseract executable is not available")

    def _bands(self, values: list[int]) -> list[tuple[int, int]]:
        if not values:
            return []
        bands: list[tuple[int, int]] = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            bands.append((start, previous))
            start = previous = value
        bands.append((start, previous))
        return bands


__all__ = [
    "FractionDocumentArtifact",
    "FractionDocumentExtractor",
    "FractionOccurrence",
    "FractionProblem",
]
