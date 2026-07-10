"""
privacy/layout_detector.py
----------------------------
Stage 1 of the Privacy Engine: Local Layout Analysis.

This detector needs ZERO legible text. It works purely on ink geometry —
where on the page is there handwriting/print, and where are the visual
"lines" of a form — using classical OpenCV operations (grayscale threshold,
horizontal ink-density projection, ruled-underline detection via
morphological horizontal-line extraction).

This is the piece that actually solves the core problem OCR-only detection
couldn't: on a heavily degraded/handwritten scan where Tesseract reads
literal noise, layout analysis can still tell "there's a dense block of
ink-on-ruled-lines in the top 25% of the page, typical of a patient-info
header" — because it never has to read a single word to say that.

Confidence here is intentionally moderate (0.45-0.7): this is one vote
among several in the fusion stage (privacy/confidence.py), not a
standalone verdict.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from privacy.types import DetectedRegion

DETECTOR_NAME = "layout"


def _pil_to_gray_np(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("L"))
    return arr


def _ink_row_profile(gray: np.ndarray, thresh: int = 180) -> np.ndarray:
    """Fraction of dark pixels per row — the classic 'horizontal projection profile'."""
    dark = (gray < thresh).astype(np.uint8)
    return dark.sum(axis=1) / float(gray.shape[1])


def _find_ink_bands(row_profile: np.ndarray, min_density: float = 0.02, min_gap: int = 6) -> List[Tuple[int, int]]:
    """Group rows with meaningful ink density into contiguous (start, end) bands, merging small gaps."""
    is_ink = row_profile > min_density
    bands = []
    start = None
    gap = 0
    for y, val in enumerate(is_ink):
        if val:
            if start is None:
                start = y
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > min_gap:
                    bands.append((start, y - gap))
                    start = None
                    gap = 0
    if start is not None:
        bands.append((start, len(is_ink) - 1))
    return bands


def _detect_ruled_underlines(gray: np.ndarray) -> List[int]:
    """
    Detect long horizontal ruled lines (common under Name:/Address:/Age: fields
    on prescription pads) via morphological horizontal-line extraction. Returns
    the y-position of each detected line.
    """
    if not CV2_AVAILABLE:
        return []
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    width = gray.shape[1]
    kernel_len = max(20, width // 6)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    contours, _ = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= kernel_len:
            lines.append(y)
    return sorted(lines)


class LayoutDetector:
    """Structural, OCR-independent detector for probable header/footer/signature blocks."""

    name = DETECTOR_NAME

    def detect(self, image: Image.Image) -> List[DetectedRegion]:
        width, height = image.size
        gray = _pil_to_gray_np(image)
        row_profile = _ink_row_profile(gray)
        bands = _find_ink_bands(row_profile)
        underlines = _detect_ruled_underlines(gray)

        regions: List[DetectedRegion] = []

        # --- Header signal: ruled underlines clustered in the top ~40% of the page ---
        header_underlines = [y for y in underlines if y < height * 0.40]
        if header_underlines:
            header_bottom = max(header_underlines) + 8
            confidence = min(0.70, 0.45 + 0.06 * len(header_underlines))  # more ruled lines -> more confident it's a form header
            regions.append(
                DetectedRegion(
                    bbox=(0, 0, width, min(height, header_bottom)),
                    field_type="header_block",
                    confidence=confidence,
                    detector=DETECTOR_NAME,
                )
            )
        elif bands:
            # No ruled lines found, but if there's a dense, well-separated ink band right at the
            # top followed by a clear gap, that's still a weak-but-real structural signal.
            first_band = bands[0]
            band_height = first_band[1] - first_band[0]
            if first_band[0] < height * 0.05 and band_height < height * 0.35 and len(bands) > 1:
                gap_to_next = bands[1][0] - first_band[1]
                if gap_to_next > height * 0.02:
                    regions.append(
                        DetectedRegion(
                            bbox=(0, 0, width, first_band[1] + 8),
                            field_type="header_block",
                            confidence=0.45,
                            detector=DETECTOR_NAME,
                        )
                    )

        # --- Footer signal: ruled underline(s) in the bottom ~20%, typical of a signature line ---
        footer_underlines = [y for y in underlines if y > height * 0.80]
        if footer_underlines:
            footer_top = min(footer_underlines) - 10
            regions.append(
                DetectedRegion(
                    bbox=(0, max(0, footer_top), width, height - max(0, footer_top)),
                    field_type="footer_block",
                    confidence=0.55,
                    detector=DETECTOR_NAME,
                )
            )

        return regions

    def find_rx_marker_band(self, image: Image.Image) -> "int | None":
        """
        Purely structural fallback attempt to locate a printed 'Rx' glyph position
        WITHOUT OCR: prescription pads often have a visually isolated short ink
        band (the Rx symbol) sitting alone between the ruled header lines and the
        denser medication-body ink band. This is intentionally conservative —
        returns None (defer to OCR-assisted detection) unless the structural
        pattern is unambiguous.
        """
        width, height = image.size
        gray = _pil_to_gray_np(image)
        row_profile = _ink_row_profile(gray)
        bands = _find_ink_bands(row_profile)
        underlines = _detect_ruled_underlines(gray)
        header_underlines = [y for y in underlines if y < height * 0.40]
        if not header_underlines:
            return None
        header_end = max(header_underlines)

        # Look for a small, isolated ink band shortly after the last header underline,
        # clearly separated from both the header above and a denser block below.
        candidates = [b for b in bands if b[0] > header_end + 4 and (b[1] - b[0]) < height * 0.08]
        if not candidates:
            return None
        # Prefer the first such isolated band after the header.
        candidate = candidates[0]
        return candidate[1] + 6
