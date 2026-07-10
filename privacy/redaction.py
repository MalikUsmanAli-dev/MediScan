"""
privacy/redaction.py
-----------------------
Stage 5: Automatic Redaction Generator, plus the safety-net checks that sit
around it (coverage sanity check, medicine-overlap check, guaranteed
fallback baseline) and the masking methods themselves (blur / black box /
pixelation).

User-supplied overrides (manual top/bottom bands, Rx-aware header band,
custom rectangles) are modeled as a "user" detector here with confidence
1.0 — they're not a guess, so they always clear the auto-mask threshold
and are never subject to confidence fusion discounting.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFilter

from privacy.types import DetectedRegion, FusedRegion
from privacy.layout_detector import LayoutDetector
from privacy.ocr_detector import group_words_into_lines

try:
    from modules.medicine_dictionary import ALL_MEDICINE_NAMES
except ImportError:
    ALL_MEDICINE_NAMES = []

MEDICINE_STRENGTH_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(mg|mcg|g|ml|iu|%)\b", re.IGNORECASE)
MEDICINE_FUZZY_CUTOFF = 0.72
HIGH_COVERAGE_WARNING_THRESHOLD = 0.55
FALLBACK_TOP_PERCENT = 10.0

RX_MARKER_TOKENS = {"rx", "px", "bx", "rz", "rc", "fx", "ry", "ix"}


# --------------------------------------------------------------------------- #
# User-override region builders (confidence = 1.0, detector = "user")
# --------------------------------------------------------------------------- #
def manual_band_regions(image_size: Tuple[int, int], top_percent: float = 0.0, bottom_percent: float = 0.0) -> List[DetectedRegion]:
    width, height = image_size
    regions = []
    if top_percent > 0:
        h = int(height * (top_percent / 100.0))
        if h > 0:
            regions.append(DetectedRegion((0, 0, width, h), "manual_top", 1.0, "user"))
    if bottom_percent > 0:
        h = int(height * (bottom_percent / 100.0))
        if h > 0:
            regions.append(DetectedRegion((0, height - h, width, h), "manual_bottom", 1.0, "user"))
    return regions


def custom_regions_from_config(image_size: Tuple[int, int], regions_cfg: List[Dict]) -> List[DetectedRegion]:
    width, height = image_size
    out = []
    for r in regions_cfg:
        left = int(width * (max(0.0, r.get("left_pct", 0)) / 100.0))
        top = int(height * (max(0.0, r.get("top_pct", 0)) / 100.0))
        w = int(width * (max(0.0, r.get("width_pct", 0)) / 100.0))
        h = int(height * (max(0.0, r.get("height_pct", 0)) / 100.0))
        if w > 0 and h > 0:
            out.append(DetectedRegion((left, top, w, h), "manual_custom", 1.0, "user"))
    return out


def find_rx_marker_top_ocr(image: Image.Image, language: str = "eng", psm: str = "6", search_band_pct: float = 60.0):
    """OCR-assisted Rx marker search (kept from the original, tested implementation)."""
    lines = group_words_into_lines(image, language, psm)
    height = image.size[1]
    band_height = height * (search_band_pct / 100.0)
    best_top = None
    for words in lines:
        for w in words:
            if w["top"] > band_height or len(w["text"].strip()) > 4:
                continue
            cleaned = re.sub(r"[^a-z]", "", w["text"].lower())
            if cleaned in RX_MARKER_TOKENS:
                if best_top is None or w["top"] < best_top:
                    best_top = w["top"]
    return best_top


def smart_header_region(image: Image.Image, fallback_top_percent: float, language: str = "eng", psm: str = "6") -> Tuple[List[DetectedRegion], bool]:
    """
    Structural-first, OCR-assisted-second Rx detection: try the layout
    detector's purely structural guess first; if that's inconclusive, try
    OCR-assisted; if both fail, fall back to a fixed top percentage.
    Returns (regions, rx_marker_found).
    """
    width, height = image.size
    layout = LayoutDetector()
    rx_top = layout.find_rx_marker_band(image)
    source = "layout structural"
    if rx_top is None:
        rx_top = find_rx_marker_top_ocr(image, language, psm)
        source = "ocr-assisted"

    if rx_top is not None:
        h = min(height, rx_top + 6)
        return [DetectedRegion((0, 0, width, h), "manual_top_rx", 1.0, "user")], True
    return manual_band_regions((width, height), top_percent=fallback_top_percent, bottom_percent=0.0), False


# --------------------------------------------------------------------------- #
# Safety checks
# --------------------------------------------------------------------------- #
def _line_looks_like_medicine(line_text: str) -> bool:
    if MEDICINE_STRENGTH_RE.search(line_text):
        return True
    if ALL_MEDICINE_NAMES:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", line_text)
        for tok in tokens:
            if difflib.get_close_matches(tok, ALL_MEDICINE_NAMES, n=1, cutoff=MEDICINE_FUZZY_CUTOFF):
                return True
    return False


def find_probable_medicine_line_boxes(image: Image.Image, language: str = "eng", psm: str = "6") -> List[Tuple[int, int, int, int]]:
    """Locate lines that look like medicine/dosage content, purely to check redaction overlap against."""
    lines = group_words_into_lines(image, language, psm)
    boxes = []
    for words in lines:
        if not words:
            continue
        line_text = " ".join(w["text"] for w in words)
        if _line_looks_like_medicine(line_text):
            left = min(w["left"] for w in words)
            top = min(w["top"] for w in words)
            right = max(w["left"] + w["width"] for w in words)
            bottom = max(w["top"] + w["height"] for w in words)
            boxes.append((left, top, right - left, bottom - top))
    return boxes


def _boxes_overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def regions_overlap_medicine_lines(regions: List[FusedRegion], medicine_boxes: List[Tuple[int, int, int, int]]) -> List[str]:
    overlapping = []
    for r in regions:
        for mb in medicine_boxes:
            if _boxes_overlap(r.bbox, mb):
                overlapping.append(r.field_type)
                break
    return overlapping


def compute_redaction_coverage(image_size: Tuple[int, int], regions: List[FusedRegion]) -> float:
    """Deduplicated (rasterized) fraction of the image covered by the given regions."""
    if not regions:
        return 0.0
    width, height = image_size
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for r in regions:
        x, y, w, h = r.bbox
        draw.rectangle((max(0, x), max(0, y), min(width, x + w), min(height, y + h)), fill=1)
    return sum(mask.getdata()) / float(width * height)


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def apply_mask(image: Image.Image, regions: List[FusedRegion], method: str = "blackbox", padding: int = 3) -> Image.Image:
    sanitized = image.copy().convert("RGB")
    if not regions:
        return sanitized

    if method == "blackbox":
        draw = ImageDraw.Draw(sanitized)
        for r in regions:
            x, y, w, h = r.bbox
            box = (max(0, x - padding), max(0, y - padding), x + w + padding, y + h + padding)
            draw.rectangle(box, fill=(0, 0, 0))
        return sanitized

    for r in regions:
        x, y, w, h = r.bbox
        box = (max(0, x - padding), max(0, y - padding), min(sanitized.width, x + w + padding), min(sanitized.height, y + h + padding))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = sanitized.crop(box)
        if method == "blur":
            processed = crop.filter(ImageFilter.GaussianBlur(radius=10))
        elif method == "pixelate":
            sw, sh = max(1, crop.width // 8), max(1, crop.height // 8)
            processed = crop.resize((sw, sh), Image.NEAREST).resize(crop.size, Image.NEAREST)
        else:
            processed = crop
        sanitized.paste(processed, box)

    return sanitized


# --------------------------------------------------------------------------- #
# Legacy alternate strategy: whitelist mode (redact everything except
# recognized clinical lines). Kept available as an explicit opt-in strategy
# rather than folded into confidence fusion, since it's a different paradigm
# (deterministic exclusion list vs. probabilistic detection) — tested
# previously to work well on clean text but risk over-redaction on very
# noisy scans, which the coverage check below still guards against.
# --------------------------------------------------------------------------- #
from modules.medicine_dictionary import FREQUENCY_ABBREVIATIONS as _FREQ_ABBR

CLINICAL_SAFE_KEYWORDS = {
    "rx", "sig", "cap", "cap.", "caps", "capsule", "capsules", "tab", "tab.", "tabs",
    "tablet", "tablets", "mg", "mcg", "ml", "iu", "syrup", "drops", "drop", "injection",
    "inj", "inj.", "dose", "doses", "dosage", "daily", "meals", "before", "after",
    "morning", "night", "bedtime", "day", "days", "week", "weeks", "refill", "qty",
    "quantity", "route", "oral", "topical", "prn", "sos", "stat",
}
CLINICAL_SAFE_KEYWORDS |= {k.lower() for k in _FREQ_ABBR.keys()}


def _line_is_clinically_safe(line_text: str) -> bool:
    if _line_looks_like_medicine(line_text):
        return True
    tokens = {re.sub(r"[^a-z0-9.]", "", w.lower()) for w in line_text.split()}
    return bool(tokens & CLINICAL_SAFE_KEYWORDS)


def whitelist_regions(image: Image.Image, language: str = "eng", psm: str = "6") -> Tuple[List[DetectedRegion], int, int]:
    """Redact every line EXCEPT ones confidently recognized as clinical content."""
    lines = group_words_into_lines(image, language, psm)
    regions: List[DetectedRegion] = []
    safe_count = 0
    for words in lines:
        if not words:
            continue
        line_text = " ".join(w["text"] for w in words)
        left = min(w["left"] for w in words)
        top = min(w["top"] for w in words)
        right = max(w["left"] + w["width"] for w in words)
        bottom = max(w["top"] + w["height"] for w in words)
        if _line_is_clinically_safe(line_text):
            safe_count += 1
        else:
            regions.append(DetectedRegion((left, top, right - left, bottom - top), "whitelist_redacted", 1.0, "whitelist"))
    return regions, safe_count, len(lines)
