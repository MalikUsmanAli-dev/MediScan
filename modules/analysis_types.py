"""
analysis_types.py
-------------------
Shared data structures used by every analysis engine (Tesseract OCR,
Groq cloud AI, and the Hybrid router) so results are interchangeable
no matter which engine produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MedicineItem:
    name: str
    strength: Optional[str] = None
    frequency: Optional[str] = None
    confidence: int = 0
    source: Optional[str] = None  # "ocr" | "kept" | "corrected" | "added" | "ai" | None


@dataclass
class AnalysisResult:
    success: bool
    medicines: List[MedicineItem] = field(default_factory=list)
    overall_confidence: int = 0
    prescription_quality: str = "Unknown"
    doctor_notes: Optional[str] = None
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_s: float = 0.0
    ocr_word_count: int = 0
    # Populated by the engine / router so the UI can show what actually ran.
    engine: str = ""              # "tesseract" | "<provider_key>" | "hybrid"
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    pipeline_note: Optional[str] = None  # human-readable description of what the hybrid pipeline did
