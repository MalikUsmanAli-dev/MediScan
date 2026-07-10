"""
ocr_engine.py
--------------
100% offline prescription-reading engine built on Tesseract OCR
(via pytesseract) plus a lightweight, rule-based extraction layer.

No network calls, no API keys, no cloud AI — everything runs locally.
This module:
    1. Runs Tesseract OCR on the chosen (pre-processed) image and pulls
       word-level text + confidence scores.
    2. Groups words back into lines and scans each line for:
         - a known medicine name (fuzzy-matched against a local
           offline dictionary in `medicine_dictionary.py`)
         - a dosage strength (e.g. "500mg", "5ml")
         - a frequency / usage instruction (e.g. "1-0-1", "BD", "twice daily")
    3. Produces the same structured `AnalysisResult` shape the rest of
       the app expects, so the Streamlit UI barely had to change.

Because there is no AI reasoning step, confidence scores are derived
directly from Tesseract's OCR confidence combined with how closely a
detected token matches the offline medicine dictionary.
"""

from __future__ import annotations

import difflib
import os
import platform
import re
import shutil
import time
from typing import Dict, List, Optional

from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from modules.medicine_dictionary import ALL_MEDICINE_NAMES, FREQUENCY_ABBREVIATIONS
from modules.analysis_types import MedicineItem, AnalysisResult

DEFAULT_LANGUAGE = "eng"
DEFAULT_OEM_PSM = "--oem 3 --psm 6"

# Tesseract's page-segmentation mode matters a lot for prescriptions, which
# often aren't a clean uniform paragraph. Exposed in Settings so users can
# try alternatives when the default misses lines.
PSM_PRESETS = {
    "6": "Assume a single uniform block of text (default)",
    "4": "Assume a single column of text of variable sizes",
    "11": "Sparse text — find as much text as possible, no particular order",
    "12": "Sparse text with orientation/script detection",
    "3": "Fully automatic page segmentation (no OSD)",
}


def build_config(psm: str = "6", oem: str = "3") -> str:
    return f"--oem {oem} --psm {psm}"

# Windows installers (e.g. the UB-Mannheim build) don't always add tesseract.exe
# to PATH, so pytesseract can't find it via shutil.which. Check the common
# install locations as a fallback before giving up.
WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]

STRENGTH_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s?(mg|mcg|g|ml|iu|%)\b", re.IGNORECASE
)
DOSE_TRIPLE_PATTERN = re.compile(r"\b(\d)\s*[-–]\s*(\d)\s*[-–]\s*(\d)\b")
ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FREQUENCY_ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
PHRASE_FREQ_PATTERN = re.compile(
    r"\b(once|twice|thrice|two times|three times|four times)\s+(a\s+day|daily)\b"
    r"|\b(before|after)\s+meals?\b"
    r"|\bevery\s+\d+\s+hours?\b",
    re.IGNORECASE,
)
NOTE_KEYWORDS = re.compile(
    r"\b(dr\.?|doctor|patient|name|age|date|diagnosis|weight|follow[- ]?up|clinic|hospital)\b",
    re.IGNORECASE,
)

FUZZY_CUTOFF = 0.72          # lowered from 0.78 — real OCR text on handwriting has more noise
MAX_FALLBACK_CANDIDATES = 8  # cap how many "unverified" rows we surface if strict matching finds nothing


def set_tesseract_cmd(path: str) -> None:
    """Manually point pytesseract at a specific tesseract.exe / tesseract binary."""
    if PYTESSERACT_AVAILABLE and path:
        pytesseract.pytesseract.tesseract_cmd = path


def _try_autolocate_windows() -> Optional[str]:
    """Best-effort search of common Windows install locations for tesseract.exe."""
    for candidate in WINDOWS_FALLBACK_PATHS:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def tesseract_is_available() -> bool:
    """
    Check whether the Tesseract binary is installed and reachable.
    Also auto-configures pytesseract if the binary is found in a known
    Windows install path but isn't on PATH yet.
    """
    if not PYTESSERACT_AVAILABLE:
        return False

    # Respect an explicit override (env var or one set via Settings/set_tesseract_cmd).
    env_override = os.getenv("TESSERACT_CMD")
    if env_override and os.path.isfile(env_override):
        pytesseract.pytesseract.tesseract_cmd = env_override

    if shutil.which("tesseract") is not None:
        return True

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass

    if platform.system() == "Windows":
        found = _try_autolocate_windows()
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
            try:
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                return False

    return False


class TesseractPrescriptionAnalyzer:
    """Fully offline OCR + rule-based structured extraction engine."""

    def __init__(self, language: str = DEFAULT_LANGUAGE, config: str = DEFAULT_OEM_PSM):
        if not PYTESSERACT_AVAILABLE:
            raise ImportError(
                "The 'pytesseract' package is not installed. Run: pip install pytesseract"
            )
        if not tesseract_is_available():
            raise RuntimeError(
                "The Tesseract OCR engine was not found on this system. Install it, then make "
                "sure it's on your PATH (or set its exact path in Settings):\n"
                "  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, "
                "default path is C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
                "  Ubuntu/Debian: sudo apt install tesseract-ocr\n"
                "  macOS (Homebrew): brew install tesseract"
            )
        self.language = language
        self.config = config

    # ------------------------------------------------------------------ #
    def analyze(self, image: Image.Image) -> AnalysisResult:
        """Run OCR + rule-based extraction on a single (already pre-processed) image."""
        start = time.perf_counter()
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
            lines = self._group_into_lines(data)
            full_text = "\n".join(line["text"] for line in lines if line["text"].strip())
            medicines, note_lines = self._extract_medicines(lines)

            word_confidences = [c for c in data.get("conf", []) if isinstance(c, (int, float)) and c >= 0]
            overall_conf = int(round(sum(word_confidences) / len(word_confidences))) if word_confidences else 0

            elapsed = time.perf_counter() - start
            return AnalysisResult(
                success=True,
                medicines=medicines,
                overall_confidence=overall_conf,
                prescription_quality="Unknown",  # filled in by caller using local image heuristics
                doctor_notes="\n".join(note_lines) if note_lines else None,
                raw_response=full_text,
                processing_time_s=elapsed,
                ocr_word_count=len(word_confidences),
                engine="tesseract",
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            return AnalysisResult(
                success=False,
                error_message=self._friendly_error(exc),
                processing_time_s=elapsed,
                engine="tesseract",
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _group_into_lines(data: dict) -> List[Dict]:
        """Reconstruct OCR words into lines using Tesseract's block/par/line indices."""
        groups: Dict[tuple, Dict] = {}
        n = len(data.get("text", []))
        for i in range(n):
            word = data["text"][i].strip()
            conf = data["conf"][i]
            if not word:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if key not in groups:
                groups[key] = {"words": [], "confs": []}
            groups[key]["words"].append(word)
            if isinstance(conf, (int, float)) and conf >= 0:
                groups[key]["confs"].append(conf)

        lines = []
        for key in sorted(groups.keys()):
            g = groups[key]
            text = " ".join(g["words"])
            avg_conf = sum(g["confs"]) / len(g["confs"]) if g["confs"] else 0.0
            lines.append({"text": text, "confidence": avg_conf})
        return lines

    # ------------------------------------------------------------------ #
    @classmethod
    def _extract_medicines(cls, lines: List[Dict]) -> tuple[List[MedicineItem], List[str]]:
        medicines: List[MedicineItem] = []
        note_lines: List[str] = []
        leftover_candidates: List[Dict] = []  # lines that weren't confidently classified either way
        seen_names = set()

        for line in lines:
            text = line["text"]
            if not text.strip():
                continue

            strength = cls._find_strength(text)
            frequency = cls._find_frequency(text)
            dict_name, match_ratio = cls._match_dictionary(text)

            is_medicine_line = bool(dict_name) or bool(strength)

            if is_medicine_line:
                name = dict_name or cls._guess_name_from_line(text, strength)
                key = name.lower().strip()
                if key and key not in seen_names:
                    seen_names.add(key)
                    confidence = cls._score_confidence(line["confidence"], match_ratio, bool(strength))
                    medicines.append(
                        MedicineItem(
                            name=name,
                            strength=strength,
                            frequency=frequency,
                            confidence=confidence,
                            source="ocr",
                        )
                    )
                elif key in seen_names and (strength or frequency):
                    # merge additional details into an already-detected medicine
                    for m in medicines:
                        if m.name.lower().strip() == key:
                            m.strength = m.strength or strength
                            m.frequency = m.frequency or frequency
                            break
            elif NOTE_KEYWORDS.search(text):
                note_lines.append(text)
            elif len(text.split()) >= 3:
                note_lines.append(text)
            else:
                # Short, ambiguous line: not clearly a note, but didn't hit our strict
                # medicine-detection rules either (common with garbled handwriting OCR).
                # Keep it as a low-confidence candidate rather than silently dropping it.
                cleaned = re.sub(r"[^A-Za-z\s]", "", text).strip()
                if len(cleaned) >= 3:
                    leftover_candidates.append(line)

        # Fallback: if strict matching found literally nothing, surface the highest-confidence
        # leftover lines as unverified rows so there's something to review/edit instead of an
        # empty table. This matters most on handwritten prescriptions, where Tesseract's
        # confidence and dictionary matching both tend to fail outright.
        if not medicines and leftover_candidates:
            leftover_candidates.sort(key=lambda l: l["confidence"], reverse=True)
            for line in leftover_candidates[:MAX_FALLBACK_CANDIDATES]:
                text = line["text"]
                strength = cls._find_strength(text)
                frequency = cls._find_frequency(text)
                cleaned = re.sub(r"[^A-Za-z\s]", "", text).strip()
                name = f"{cleaned.title()} (unverified)" if cleaned else "Unverified text"
                medicines.append(
                    MedicineItem(
                        name=name,
                        strength=strength,
                        frequency=frequency,
                        confidence=int(round(line["confidence"] * 0.5)),
                        source="ocr-unverified",
                    )
                )

        return medicines, note_lines

    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_strength(text: str) -> Optional[str]:
        m = STRENGTH_PATTERN.search(text)
        if m:
            return f"{m.group(1)}{m.group(2).lower()}"
        return None

    @staticmethod
    def _find_frequency(text: str) -> Optional[str]:
        m = DOSE_TRIPLE_PATTERN.search(text)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = ABBREV_PATTERN.search(text)
        if m:
            abbrev = m.group(1).upper()
            return FREQUENCY_ABBREVIATIONS.get(abbrev, abbrev)
        m = PHRASE_FREQ_PATTERN.search(text)
        if m:
            return m.group(0).strip().lower()
        return None

    @staticmethod
    def _match_dictionary(text: str) -> tuple[Optional[str], float]:
        """Fuzzy-match tokens/bigrams in a line against the offline medicine dictionary."""
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        candidates = tokens + [
            f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)
        ]
        best_name, best_ratio = None, 0.0
        for cand in candidates:
            matches = difflib.get_close_matches(cand, ALL_MEDICINE_NAMES, n=1, cutoff=FUZZY_CUTOFF)
            if matches:
                ratio = difflib.SequenceMatcher(None, cand.lower(), matches[0].lower()).ratio()
                if ratio > best_ratio:
                    best_name, best_ratio = matches[0], ratio
        return best_name, best_ratio

    @staticmethod
    def _guess_name_from_line(text: str, strength: Optional[str]) -> str:
        """Fallback: use the words preceding the dosage strength as the medicine name."""
        cleaned = STRENGTH_PATTERN.sub("", text).strip(" -:,.")
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", cleaned)]
        guess = " ".join(words[:3]) if words else "Unidentified medicine"
        return guess.title()

    @staticmethod
    def _score_confidence(ocr_conf: float, match_ratio: float, has_strength: bool) -> int:
        base = ocr_conf  # 0-100 from Tesseract
        if match_ratio > 0:
            score = 0.55 * base + 0.45 * (match_ratio * 100)
        else:
            score = 0.75 * base
            if has_strength:
                score += 5  # small boost: dosage pattern is a strong medicine-line signal
        return int(max(0, min(100, round(score))))

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "tesseract" in lowered and ("not installed" in lowered or "not found" in lowered or "path" in lowered):
            return (
                "Tesseract OCR engine not found on this system. Install the 'tesseract-ocr' "
                "package for your OS and ensure it is on your PATH."
            )
        return f"OCR processing failed: {message}"


def classify_prescription_quality(sharpness: float, brightness: float, contrast: float) -> str:
    """
    Map locally-computed OpenCV image quality heuristics to a human-readable
    quality label — fully offline, no AI call, replaces the previous
    AI-assessed quality label.
    """
    score = 0
    if sharpness >= 400:
        score += 1
    if sharpness >= 800:
        score += 1
    if 60 <= brightness <= 210:
        score += 1
    if contrast >= 35:
        score += 1
    if contrast >= 55:
        score += 1

    if score >= 4:
        return "Excellent"
    if score >= 3:
        return "Good"
    if score >= 1:
        return "Fair"
    return "Poor"
