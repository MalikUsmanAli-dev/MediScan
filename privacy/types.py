"""
privacy/types.py
------------------
Shared data structures used across every detector and the engine itself.
Keeping these in one place is what lets new detectors (YOLO-based, PaddleOCR
layout, DocLayout, etc.) plug in later without touching engine.py's logic —
they just need to return a list of DetectedRegion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # (left, top, width, height) in pixel coords


@dataclass
class DetectedRegion:
    """A single piece of evidence from ONE detector that a region may be sensitive."""
    bbox: BBox
    field_type: str          # e.g. "patient_name", "phone_number", "header_block"
    confidence: float        # 0.0-1.0, this detector's own confidence
    detector: str            # which detector produced this, e.g. "layout", "ocr", "regex", "user"


@dataclass
class FusedRegion:
    """Output of confidence fusion: one or more DetectedRegions merged into a single verdict."""
    bbox: BBox
    field_type: str                    # best/primary field label across contributors
    confidence: float                  # fused confidence, 0.0-1.0
    contributing_detectors: List[str] = field(default_factory=list)
    auto_masked: bool = False          # True once thresholding decides to mask it


@dataclass
class PrivacyConfig:
    enabled: bool = True
    fields_enabled: Dict[str, bool] = field(default_factory=dict)  # e.g. {"patient_name": True, ...}
    method: str = "blackbox"           # "blackbox" | "blur" | "pixelate"
    manual_top_pct: float = 0.0
    manual_bottom_pct: float = 0.0
    smart_rx_header: bool = False
    custom_regions: List[Dict] = field(default_factory=list)  # [{"top_pct","left_pct","width_pct","height_pct"}]
    auto_mask_threshold: float = 0.5   # fused confidence at/above this -> auto-masked
    language: str = "eng"
    psm: str = "6"
    strategy: str = "targeted"         # "targeted" | "whitelist" (legacy alternate strategy, still available)


@dataclass
class PrivacyReport:
    """Metadata returned alongside the sanitized image — for UI transparency, the PDF report,
    and future benchmarking/research as required by the spec."""
    enabled: bool
    engine_version: str = "privacy-engine-v1"
    detectors_used: List[str] = field(default_factory=list)
    masked_fields: List[Dict] = field(default_factory=list)  # [{"field": str, "confidence": float, "sources": [str]}]
    method: str = "blackbox"
    processing_time_s: float = 0.0
    region_count: int = 0
    coverage_pct: float = 0.0
    high_coverage: bool = False
    overlaps_medicine: bool = False
    used_fallback_baseline: bool = False
    rx_marker_found: Optional[bool] = None
    ocr_low_confidence: bool = False
    low_confidence_regions: List[Dict] = field(default_factory=list)  # regions below auto-mask threshold, for optional manual review

    @property
    def fields_removed_labels(self) -> List[str]:
        """Convenience list of human-readable strings for UI/PDF display."""
        labels = [m["field_label"] for m in self.masked_fields]
        if self.used_fallback_baseline and "Fallback: top 10% header (no PII detected by any active method)" not in labels:
            labels.append("Fallback: top 10% header (no PII detected by any active method)")
        return labels
