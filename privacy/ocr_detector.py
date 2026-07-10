"""
privacy/ocr_detector.py
--------------------------
Stage 2 of the Privacy Engine: OCR-assisted detection.

This is the SAME keyword/label-matching logic the original privacy_filter.py
used, migrated here largely as-is — but its role has changed. It used to be
the single source of truth for what to redact; now it's one detector among
several, and its confidence output reflects that (scaled down, and fused
with layout/regex evidence rather than trusted alone).

Still useful and still tested to work well on clean/printed text — the
failure mode we found was specifically on handwritten/degraded scans, where
this detector's output should simply carry less weight, not be discarded.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from privacy.types import DetectedRegion

DETECTOR_NAME = "ocr"

LABEL_TOKENS = {
    "doctor_name": {"dr", "dr.", "doctor", "consultant", "physician"},
    "hospital_name": {"hospital", "clinic", "centre", "center"},
    "patient_id": {"mrn", "mr#", "mr.no", "reg#", "reg", "registration", "id#", "id"},
    "address": {"address", "add", "add.", "residence"},
    "patient_name": {"name", "patient", "mr", "mrs", "ms", "pt", "pt.", "s/o", "d/o", "w/o"},
}
VALUE_WORD_SPAN = {
    "patient_name": 3,
    "patient_id": 3,
    "address": 8,
    "doctor_name": 3,
    "hospital_name": 4,
}
# Label-based detection carries lower base confidence than before (this detector is now
# "supporting evidence", not sole authority) — real confidence still scales with OCR word confidence.
LABEL_BASE_CONFIDENCE = 0.62

MIN_WORDS_FOR_CONFIDENT_DETECTION = 6


def _clean_token(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def group_words_into_lines(image: Image.Image, language: str = "eng", psm: str = "6") -> List[List[Dict]]:
    """Shared helper: group Tesseract word-level output into visual lines with boxes."""
    if not PYTESSERACT_AVAILABLE:
        return []
    config = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(image, lang=language, config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    groups: Dict[tuple, List[Dict]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(
            {
                "text": word,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "conf": data["conf"][i] if isinstance(data["conf"][i], (int, float)) else -1,
            }
        )
    return [groups[k] for k in sorted(groups.keys())]


class OCRDetector:
    """Label/keyword-based detector, demoted to a supporting-evidence role in the fused pipeline."""

    name = DETECTOR_NAME

    def detect(self, image: Image.Image, enabled_fields: set, language: str = "eng", psm: str = "6") -> Tuple[List[DetectedRegion], bool, int]:
        """Returns (regions, low_confidence_flag, total_word_count)."""
        if not PYTESSERACT_AVAILABLE or not enabled_fields:
            return [], False, 0

        lines = group_words_into_lines(image, language, psm)
        total_words = sum(len(words) for words in lines)
        low_confidence = total_words < MIN_WORDS_FOR_CONFIDENT_DETECTION

        regions: List[DetectedRegion] = []
        for words in lines:
            if not words:
                continue
            lowered_tokens = [_clean_token(w["text"]) for w in words]
            for candidate_field in ("doctor_name", "hospital_name", "patient_id", "address", "patient_name"):
                if candidate_field not in enabled_fields:
                    continue
                tokens = LABEL_TOKENS[candidate_field]
                label_idx = None
                for idx, tok in enumerate(lowered_tokens):
                    if tok in tokens:
                        label_idx = idx
                if label_idx is None:
                    continue
                span = VALUE_WORD_SPAN.get(candidate_field, 3)
                value_words = words[label_idx + 1: label_idx + 1 + span]
                if not value_words:
                    continue
                word_confs = [w["conf"] for w in value_words if w["conf"] >= 0]
                avg_conf = (sum(word_confs) / len(word_confs) / 100.0) if word_confs else 0.5
                fused_conf = min(0.95, LABEL_BASE_CONFIDENCE * (0.5 + 0.5 * avg_conf))
                for w in value_words:
                    regions.append(
                        DetectedRegion(
                            bbox=(w["left"], w["top"], w["width"], w["height"]),
                            field_type=candidate_field,
                            confidence=fused_conf,
                            detector=DETECTOR_NAME,
                        )
                    )
                break  # one label match per line

        return regions, low_confidence, total_words
