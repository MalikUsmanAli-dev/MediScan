"""
privacy/engine.py
--------------------
The Privacy Engine — single entry point for the rest of MediScan.

    Image
      |
      v
    PrivacyEngine.process(image, config)
      |
      +-- Stage 1: LayoutDetector       (structural, OCR-independent)
      +-- Stage 2: OCRDetector          (label-based, supporting evidence)
      +-- Stage 3: RegexDetector        (high-precision patterns)
      +-- Stage 4: confidence fusion    (combine evidence -> per-region confidence)
      +-- Stage 5: redaction generator  (threshold -> mask, + safety nets)
      |
      v
    (sanitized_image, PrivacyReport)

Only ever called before a cloud AI provider sees the image — the OCR/AI/
Hybrid pipelines themselves are untouched by this module. This is the ONLY
integration point app.py needs to call.
"""

from __future__ import annotations

import time
from typing import List

from PIL import Image

from privacy.types import DetectedRegion, FusedRegion, PrivacyConfig, PrivacyReport
from privacy.layout_detector import LayoutDetector
from privacy.ocr_detector import OCRDetector
from privacy.regex_detector import RegexDetector
from privacy.confidence import fuse_detections
from privacy import redaction as rx

FIELD_LABELS = {
    "patient_name": "Patient Name",
    "patient_id": "Patient ID / MR Number",
    "phone_number": "Phone Number",
    "address": "Address",
    "dob_age": "Date of Birth / Age",
    "email": "Email",
    "national_id": "National ID Number (e.g. CNIC)",
    "doctor_name": "Doctor Name",
    "hospital_name": "Hospital / Clinic Name",
    "header_block": "Header region (structural)",
    "footer_block": "Footer region (structural)",
    "manual_top": "Header region (manual)",
    "manual_top_rx": "Header region (up to Rx marker)",
    "manual_bottom": "Footer region (manual)",
    "manual_custom": "Custom region (manual)",
    "whitelist_redacted": "Non-clinical content (whitelist mode)",
}

ALL_FIELD_KEYS = ["patient_name", "patient_id", "phone_number", "address", "dob_age", "email", "national_id", "doctor_name", "hospital_name"]
DEFAULT_ENABLED_FIELDS = {"patient_name", "patient_id", "phone_number", "address", "dob_age", "email", "national_id"}

# Metadata for the Settings UI checkboxes (label + whether checked by default).
PII_FIELD_META = {
    "patient_name": {"label": "Patient Name", "default": True},
    "patient_id": {"label": "Patient ID / MR Number", "default": True},
    "phone_number": {"label": "Phone Number", "default": True},
    "address": {"label": "Address", "default": True},
    "dob_age": {"label": "Date of Birth / Age", "default": True},
    "email": {"label": "Email", "default": True},
    "national_id": {"label": "National ID Number (e.g. CNIC)", "default": True},
    "doctor_name": {"label": "Doctor Name", "default": False},
    "hospital_name": {"label": "Hospital / Clinic Name", "default": False},
}

REDACTION_METHODS = {
    "blackbox": "Black Box",
    "blur": "Blur",
    "pixelate": "Pixelation",
}


class PrivacyEngine:
    """Confidence-driven, fully local privacy engine. See module docstring for the pipeline."""

    def __init__(self):
        self.layout_detector = LayoutDetector()
        self.ocr_detector = OCRDetector()
        self.regex_detector = RegexDetector()

    def process(self, image: Image.Image, config: PrivacyConfig) -> "tuple[Image.Image, PrivacyReport]":
        start = time.perf_counter()

        if not config.enabled:
            return image.copy(), PrivacyReport(enabled=False, processing_time_s=0.0)

        detectors_used: List[str] = []
        enabled_fields = {k for k, v in config.fields_enabled.items() if v} or DEFAULT_ENABLED_FIELDS
        ocr_low_confidence = False
        rx_marker_found = None

        # ------------------------------------------------------------------ #
        # Content detection: either the confidence-fused "targeted" strategy,
        # or the legacy deterministic "whitelist" strategy.
        # ------------------------------------------------------------------ #
        fused_content: List[FusedRegion] = []
        low_confidence_regions: List[dict] = []

        if config.strategy == "whitelist":
            wl_regions, safe_count, total_lines = rx.whitelist_regions(image, config.language, config.psm)
            detectors_used.append("whitelist")
            ocr_low_confidence = total_lines == 0
            fused_content = [
                FusedRegion(bbox=r.bbox, field_type=r.field_type, confidence=1.0,
                             contributing_detectors=["whitelist"], auto_masked=True)
                for r in wl_regions
            ]
        else:
            all_evidence: List[DetectedRegion] = []

            layout_regions = self.layout_detector.detect(image)
            if layout_regions:
                detectors_used.append("layout")
            all_evidence.extend(layout_regions)

            ocr_regions, ocr_low_confidence, _word_count = self.ocr_detector.detect(
                image, enabled_fields, config.language, config.psm
            )
            if ocr_regions:
                detectors_used.append("ocr")
            all_evidence.extend(ocr_regions)

            regex_regions = self.regex_detector.detect(image, enabled_fields, config.language, config.psm)
            if regex_regions:
                detectors_used.append("regex")
            all_evidence.extend(regex_regions)

            fused_all = fuse_detections(all_evidence, auto_mask_threshold=config.auto_mask_threshold)
            fused_content = [r for r in fused_all if r.auto_masked]
            # Per spec item 5: low-confidence regions are surfaced for optional manual review,
            # never as the default workflow.
            low_confidence_regions = [
                {"field_label": FIELD_LABELS.get(r.field_type, r.field_type), "confidence": r.confidence, "bbox": r.bbox}
                for r in fused_all if not r.auto_masked
            ]

        # ------------------------------------------------------------------ #
        # User overrides: manual top/bottom bands (optionally Rx-aware),
        # custom rectangles. Always confidence 1.0 — not a guess.
        # ------------------------------------------------------------------ #
        user_regions: List[DetectedRegion] = []
        if config.smart_rx_header and config.manual_top_pct > 0:
            top_regions, rx_marker_found = rx.smart_header_region(image, config.manual_top_pct, config.language, config.psm)
            user_regions.extend(top_regions)
        elif config.manual_top_pct > 0:
            user_regions.extend(rx.manual_band_regions(image.size, top_percent=config.manual_top_pct))
        if config.manual_bottom_pct > 0:
            user_regions.extend(rx.manual_band_regions(image.size, bottom_percent=config.manual_bottom_pct))
        user_regions.extend(rx.custom_regions_from_config(image.size, config.custom_regions))
        if user_regions:
            detectors_used.append("user")

        user_fused = [
            FusedRegion(bbox=r.bbox, field_type=r.field_type, confidence=1.0,
                         contributing_detectors=["user"], auto_masked=True)
            for r in user_regions
        ]

        final_regions = fused_content + user_fused

        # ------------------------------------------------------------------ #
        # Guaranteed fallback: never silently send a fully unredacted image.
        # ------------------------------------------------------------------ #
        used_fallback_baseline = False
        if not final_regions:
            fallback = rx.manual_band_regions(image.size, top_percent=rx.FALLBACK_TOP_PERCENT)
            final_regions = [
                FusedRegion(bbox=r.bbox, field_type="fallback_baseline", confidence=1.0,
                             contributing_detectors=["fallback"], auto_masked=True)
                for r in fallback
            ]
            used_fallback_baseline = True
            detectors_used.append("fallback")

        # ------------------------------------------------------------------ #
        # Safety checks + masking
        # ------------------------------------------------------------------ #
        coverage = rx.compute_redaction_coverage(image.size, final_regions)
        high_coverage = coverage >= rx.HIGH_COVERAGE_WARNING_THRESHOLD

        medicine_boxes = rx.find_probable_medicine_line_boxes(image, config.language, config.psm)
        overlap_fields = rx.regions_overlap_medicine_lines(
            [r for r in final_regions if "user" in r.contributing_detectors or "whitelist" in r.contributing_detectors],
            medicine_boxes,
        )
        overlaps_medicine = bool(overlap_fields)

        sanitized = rx.apply_mask(image, final_regions, method=config.method)

        elapsed = time.perf_counter() - start

        masked_fields = []
        seen_labels = set()
        for r in final_regions:
            label = FIELD_LABELS.get(r.field_type, r.field_type)
            key = (label, tuple(r.contributing_detectors))
            if key in seen_labels:
                continue
            seen_labels.add(key)
            masked_fields.append({"field": r.field_type, "field_label": label, "confidence": r.confidence, "sources": r.contributing_detectors})

        report = PrivacyReport(
            enabled=True,
            detectors_used=sorted(set(detectors_used)),
            masked_fields=masked_fields,
            method=config.method,
            processing_time_s=elapsed,
            region_count=len(final_regions),
            coverage_pct=round(coverage * 100, 1),
            high_coverage=high_coverage,
            overlaps_medicine=overlaps_medicine,
            used_fallback_baseline=used_fallback_baseline,
            rx_marker_found=rx_marker_found,
            ocr_low_confidence=ocr_low_confidence,
            low_confidence_regions=low_confidence_regions,
        )
        return sanitized, report
