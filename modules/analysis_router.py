"""
analysis_router.py
---------------------
Single entry point the UI calls to run prescription analysis, regardless
of which engine(s) are configured. Supports three modes:

    "tesseract" - offline-only, for research/comparison. Local Tesseract
                  OCR + rule-based/fuzzy-match extraction. Never touches
                  the network.
    "api"       - cloud-only, for research/comparison. Sends the image
                  straight to whichever provider is configured (Groq,
                  OpenAI, OpenRouter, Together, or a custom endpoint) for
                  from-scratch extraction. Requires an API key + internet.
    "hybrid"    - the recommended default. A real two-stage pipeline:
                    1. Tesseract OCR runs on the processed image and the
                       local fuzzy-matching layer produces a draft list.
                    2. If a cloud API is configured, that draft is sent
                       to the AI *together with the original image* and
                       the AI checks/corrects/completes it (not a
                       from-scratch re-read — a verification pass).
                    3. If Tesseract found nothing usable at all (empty
                       result, or the engine isn't installed), Hybrid
                       escalates straight to full AI extraction from the
                       image instead of trying to "verify" nothing.
                    4. If no API is configured, or the AI call fails,
                       Hybrid gracefully falls back to the raw Tesseract
                       result rather than erroring out.
                  The user still edits the final result by hand afterward
                  regardless of which path was taken.

Every call returns a single `AnalysisResult` (see analysis_types.py),
annotated with `engine`, `fallback_used`, and `fallback_reason` so the UI
can be transparent with the user about what actually ran.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from modules.analysis_types import AnalysisResult
from modules import image_processing as dip
from modules.ocr_engine import (
    TesseractPrescriptionAnalyzer,
    classify_prescription_quality,
    tesseract_is_available,
    build_config,
)
from modules.cloud_ai_engine import CloudVisionAnalyzer
from modules.api_providers import ApiConfig, api_is_configured

MODES = ["tesseract", "api", "hybrid"]

ENGINE_LABELS = {
    "tesseract": "Tesseract OCR (Offline)",
    "api": "Cloud API (direct extraction)",
    "hybrid-verified": "Tesseract + AI Verification (Hybrid)",
    "hybrid-ai-extract": "AI Vision Extraction (Hybrid fallback)",
}


def _apply_local_quality(result: AnalysisResult, original_image: Image.Image) -> AnalysisResult:
    """Tesseract has no AI quality assessment, so fill it in from local OpenCV heuristics."""
    if result.success and (not result.prescription_quality or result.prescription_quality == "Unknown"):
        metrics = dip.estimate_prescription_quality(dip.to_grayscale(dip.pil_to_cv2(original_image)))
        result.prescription_quality = classify_prescription_quality(
            metrics["sharpness"], metrics["brightness"], metrics["contrast"]
        )
    return result


def _run_tesseract(image: Image.Image, original_image: Image.Image, ocr_language: str, ocr_psm: str = "6") -> AnalysisResult:
    try:
        if not tesseract_is_available():
            return AnalysisResult(
                success=False,
                engine="tesseract",
                error_message="Tesseract OCR engine not found on this system. Install it (see "
                              "Settings) or use Cloud API / Hybrid mode instead.",
            )
        analyzer = TesseractPrescriptionAnalyzer(language=ocr_language, config=build_config(ocr_psm))
        result = analyzer.analyze(image)
        return _apply_local_quality(result, original_image)
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(success=False, engine="tesseract", error_message=f"Tesseract engine error: {exc}")


def _run_api_extract(image: Image.Image, api_config: Optional[ApiConfig]) -> AnalysisResult:
    try:
        if not api_is_configured(api_config):
            return AnalysisResult(
                success=False,
                engine="api",
                error_message="No cloud API configured. Pick a provider and add an API key in Settings.",
            )
        analyzer = CloudVisionAnalyzer(api_config)
        return analyzer.extract(image)
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(success=False, engine="api", error_message=f"Cloud API error: {exc}")


def _run_api_verify(image: Image.Image, api_config: Optional[ApiConfig], draft) -> AnalysisResult:
    try:
        if not api_is_configured(api_config):
            return AnalysisResult(
                success=False,
                engine="hybrid-verified",
                error_message="No cloud API configured. Pick a provider and add an API key in Settings.",
            )
        analyzer = CloudVisionAnalyzer(api_config)
        return analyzer.verify_and_correct(image, draft)
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(success=False, engine="hybrid-verified", error_message=f"AI verification error: {exc}")


def _run_hybrid(
    image: Image.Image,
    image_for_api: Image.Image,
    original_image: Image.Image,
    ocr_language: str,
    ocr_psm: str,
    api_config: Optional[ApiConfig],
) -> AnalysisResult:
    tess_result = _run_tesseract(image, original_image, ocr_language, ocr_psm)
    api_ready = api_is_configured(api_config)

    tesseract_found_nothing = (not tess_result.success) or (not tess_result.medicines)

    if tesseract_found_nothing:
        # Nothing useful to "verify" — escalate straight to full AI extraction from the image.
        if not api_ready:
            if not tess_result.success:
                tess_result.fallback_reason = (
                    f"{tess_result.error_message} No cloud API is configured to fall back on either — "
                    "add one in Settings, or fix the Tesseract install."
                )
            else:
                tess_result.fallback_reason = (
                    "Tesseract ran but found no usable medicines, and no cloud API is configured to "
                    "fall back on. Add an API key in Settings, or try a different pipeline stage / PSM."
                )
            return tess_result

        api_result = _run_api_extract(image_for_api, api_config)
        if api_result.success:
            api_result.engine = "hybrid-ai-extract"
            api_result.fallback_used = True
            reason = (
                f"Tesseract failed ({tess_result.error_message})"
                if not tess_result.success
                else "Tesseract found no usable medicines in the OCR + fuzzy-match pass"
            )
            api_result.fallback_reason = f"{reason} — switched to direct AI vision extraction from the image."
            for m in api_result.medicines:
                m.source = m.source or "ai"
            return api_result

        # Both paths came up empty/failed.
        tess_result.fallback_used = True
        tess_result.fallback_reason = (
            "Tesseract found nothing usable, and the AI fallback also failed "
            f"({api_result.error_message})."
        )
        return tess_result

    # Tesseract produced a usable draft — have the AI verify/correct it against the real image.
    if not api_ready:
        tess_result.fallback_reason = (
            "No cloud API is configured, so this is the raw Tesseract + fuzzy-match result. "
            "Add an API key in Settings to have AI double-check it against the image."
        )
        return tess_result

    verified = _run_api_verify(image_for_api, api_config, tess_result.medicines)
    if verified.success:
        if not verified.prescription_quality or verified.prescription_quality == "Unknown":
            verified.prescription_quality = tess_result.prescription_quality
        verified.fallback_used = False
        verified.fallback_reason = (
            "AI reviewed the Tesseract + fuzzy-match draft against the original image and adjusted it "
            "where needed (see the 'Source' column below)."
        )
        return verified

    tess_result.fallback_used = True
    tess_result.fallback_reason = (
        f"AI verification failed ({verified.error_message}); showing the raw Tesseract + fuzzy-match "
        "result instead."
    )
    return tess_result


def analyze_prescription(
    image: Image.Image,
    original_image: Image.Image,
    mode: str,
    ocr_language: str = "eng",
    ocr_psm: str = "6",
    api_config: Optional[ApiConfig] = None,
    image_for_api: Optional[Image.Image] = None,
) -> AnalysisResult:
    """
    Route a single analysis request to the configured engine(s).

    `image` is the (possibly pre-processed) prescription image used for local
    Tesseract OCR — this never leaves the machine. `image_for_api`, if given,
    is a privacy-sanitized copy that gets sent instead whenever a cloud API
    call is made (falls back to `image` if not provided, e.g. Privacy Mode
    is off). `original_image` is the raw upload, used only for local OpenCV
    quality heuristics when Tesseract runs.
    """
    api_image = image_for_api if image_for_api is not None else image

    if mode == "tesseract":
        return _run_tesseract(image, original_image, ocr_language, ocr_psm)

    if mode == "api":
        return _run_api_extract(api_image, api_config)

    if mode == "hybrid":
        return _run_hybrid(image, api_image, original_image, ocr_language, ocr_psm, api_config)

    raise ValueError(f"Unknown analysis mode: {mode!r}")
