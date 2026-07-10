"""
privacy/regex_detector.py
----------------------------
Stage 3 of the Privacy Engine: Regex detection.

Phone numbers, emails, national ID numbers, and DOB/Age. These patterns are
inherently high-precision — a Pakistani CNIC format (#####-#######-#) or an
email address rarely occurs by coincidence — so they carry higher base
confidence than the label-based OCR detector, and can often justify
auto-masking on their own even without corroboration from other detectors.

Still depends on OCR to produce the text to pattern-match against, so it
inherits the same "can't read what it can't read" limitation on cursive
handwriting — hence why this is fused with layout evidence rather than relied
on alone.
"""

from __future__ import annotations

import re
from typing import List

from PIL import Image

from privacy.types import DetectedRegion
from privacy.ocr_detector import group_words_into_lines

DETECTOR_NAME = "regex"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?:\+92[-\s]?|0)3\d{2}[-\s]?\d{7}\b"
    r"|\b0\d{2,4}[-\s]?\d{6,8}\b"
    r"|\b\+?\d{1,3}[-\s]?\(?\d{2,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}\b"
)
CNIC_RE = re.compile(r"\b\d{5}-\d{7}-\d{1}\b|\b\d{13}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
AGE_NUM_RE = re.compile(r"\b\d{1,3}\b")
DOB_KEYWORDS = {"dob", "dob:", "d.o.b", "d.o.b.", "birth", "born"}
AGE_KEYWORDS = {"age", "age:"}

# High base confidence: these patterns are inherently low-false-positive-rate.
BASE_CONFIDENCE = {
    "email": 0.92,
    "phone_number": 0.85,
    "national_id": 0.9,
    "dob_age": 0.8,
}


def _clean_token(word: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", word.lower())


def _words_overlapping_span(words, line_text: str, start: int, end: int):
    overlapping = []
    cursor = 0
    for w in words:
        w_start = cursor
        w_end = cursor + len(w["text"])
        if w_start < end and w_end > start:
            overlapping.append(w)
        cursor = w_end + 1
    return overlapping


class RegexDetector:
    """High-precision pattern-based detector for phone/email/national ID/DOB-age."""

    name = DETECTOR_NAME

    def detect(self, image: Image.Image, enabled_fields: set, language: str = "eng", psm: str = "6") -> List[DetectedRegion]:
        if not enabled_fields:
            return []
        lines = group_words_into_lines(image, language, psm)
        regions: List[DetectedRegion] = []

        for words in lines:
            if not words:
                continue
            line_text = " ".join(w["text"] for w in words)
            lowered_tokens = [_clean_token(w["text"]) for w in words]

            if "email" in enabled_fields:
                for m in EMAIL_RE.finditer(line_text):
                    for w in _words_overlapping_span(words, line_text, m.start(), m.end()):
                        regions.append(DetectedRegion(
                            bbox=(w["left"], w["top"], w["width"], w["height"]),
                            field_type="email", confidence=BASE_CONFIDENCE["email"], detector=DETECTOR_NAME,
                        ))

            if "phone_number" in enabled_fields:
                for m in PHONE_RE.finditer(line_text):
                    for w in _words_overlapping_span(words, line_text, m.start(), m.end()):
                        regions.append(DetectedRegion(
                            bbox=(w["left"], w["top"], w["width"], w["height"]),
                            field_type="phone_number", confidence=BASE_CONFIDENCE["phone_number"], detector=DETECTOR_NAME,
                        ))

            if "national_id" in enabled_fields:
                for m in CNIC_RE.finditer(line_text):
                    for w in _words_overlapping_span(words, line_text, m.start(), m.end()):
                        regions.append(DetectedRegion(
                            bbox=(w["left"], w["top"], w["width"], w["height"]),
                            field_type="national_id", confidence=BASE_CONFIDENCE["national_id"], detector=DETECTOR_NAME,
                        ))

            if "dob_age" in enabled_fields:
                has_dob_kw = any(tok in DOB_KEYWORDS for tok in lowered_tokens)
                has_age_kw = any(tok in AGE_KEYWORDS for tok in lowered_tokens)
                if has_dob_kw:
                    m = DATE_RE.search(line_text)
                    if m:
                        for w in _words_overlapping_span(words, line_text, m.start(), m.end()):
                            regions.append(DetectedRegion(
                                bbox=(w["left"], w["top"], w["width"], w["height"]),
                                field_type="dob_age", confidence=BASE_CONFIDENCE["dob_age"], detector=DETECTOR_NAME,
                            ))
                if has_age_kw:
                    age_idx = next((i for i, tok in enumerate(lowered_tokens) if tok in AGE_KEYWORDS), None)
                    if age_idx is not None:
                        for w in words[age_idx + 1: age_idx + 2]:
                            if AGE_NUM_RE.fullmatch(_clean_token(w["text"])):
                                regions.append(DetectedRegion(
                                    bbox=(w["left"], w["top"], w["width"], w["height"]),
                                    field_type="dob_age", confidence=BASE_CONFIDENCE["dob_age"], detector=DETECTOR_NAME,
                                ))
        return regions
