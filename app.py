"""
app.py
-------
Offline Prescription Recognition and Medicine Retrieval System
Main Streamlit application entry point.

100% offline: image processing runs locally via OpenCV, and text
recognition runs locally via Tesseract OCR (pytesseract). No API keys,
no cloud calls, no internet connection required at analysis time.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from modules import image_processing as dip
from modules import ui_components as ui
from modules.ocr_engine import tesseract_is_available, set_tesseract_cmd, DEFAULT_LANGUAGE, PSM_PRESETS
from modules.auto_install import attempt_auto_install
from modules.api_providers import PROVIDERS, ApiConfig, api_is_configured
from modules.analysis_router import analyze_prescription, MODES, ENGINE_LABELS
from modules.analysis_types import MedicineItem
from privacy import (
    PrivacyEngine, PrivacyConfig, PII_FIELD_META, REDACTION_METHODS, DEFAULT_ENABLED_FIELDS,
)
from modules.pdf_generator import generate_pdf_report

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

load_dotenv()

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MediScan | Prescription Recognition System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "uploaded_file_bytes": None,
    "original_image": None,
    "file_metadata": None,
    "pipeline_result": None,
    "analysis_result": None,
    "ocr_image_stage": "Final Processed",  # image stage fed to Tesseract (binarized text usually reads best)
    "ai_image_stage": "Original",          # image stage fed to AI (vision models read raw photos better)
    "ocr_language": DEFAULT_LANGUAGE,
    "ocr_psm": "6",
    "current_page": "Dashboard",
    "theme": "light",
    "analysis_mode": "hybrid",   # "tesseract" | "api" | "hybrid" (default — recommended)
    "api_provider": "groq",
    "api_base_url": PROVIDERS["groq"].base_url,
    "api_key": os.getenv("CLOUD_API_KEY", "") or os.getenv("GROQ_API_KEY", ""),  # generic; GROQ_API_KEY kept only for backward compatibility
    "api_model": PROVIDERS["groq"].default_model,
    "tesseract_cmd_path": os.getenv("TESSERACT_CMD", ""),
    "privacy_enabled": True,   # default ON for api/hybrid modes; irrelevant for tesseract-only
    "privacy_fields": {k: (k in DEFAULT_ENABLED_FIELDS) for k in PII_FIELD_META.keys()},
    "privacy_method": "blackbox",
    "privacy_strategy": "targeted",  # "targeted" (redact only detected PII fields) | "whitelist" (redact everything except recognized clinical lines)
    "privacy_auto_mask_threshold": 0.5,
    "privacy_manual_top_pct": 0,
    "privacy_smart_rx_header": False,  # if manual top redaction is on, try to find Rx marker precisely, fallback to privacy_manual_top_pct
    "privacy_manual_bottom_pct": 0,
    "privacy_custom_regions": [],  # list of {"top_pct","left_pct","width_pct","height_pct"}
    "privacy_last_summary": None,  # populated after each analysis for report/UI transparency
    "privacy_last_sanitized_image": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

ui.inject_global_css(st.session_state.theme)


def reset_session():
    persistent = (
        "ocr_language", "ocr_psm", "theme", "analysis_mode",
        "api_provider", "api_base_url", "api_key", "api_model", "tesseract_cmd_path",
        "privacy_enabled", "privacy_fields", "privacy_method", "privacy_strategy", "privacy_auto_mask_threshold",
        "privacy_manual_top_pct", "privacy_manual_bottom_pct", "privacy_custom_regions", "privacy_smart_rx_header",
    )
    for key, value in DEFAULTS.items():
        if key not in persistent:
            st.session_state[key] = value


def current_api_config() -> ApiConfig:
    return ApiConfig(
        provider=st.session_state.api_provider,
        base_url=st.session_state.api_base_url,
        api_key=st.session_state.api_key,
        model=st.session_state.api_model,
    )


def privacy_applies_to_current_mode() -> bool:
    """Privacy filtering is only ever relevant when an image might be sent to a cloud API."""
    return st.session_state.analysis_mode in ("api", "hybrid")


def enabled_privacy_field_keys() -> set:
    return {k for k, v in st.session_state.privacy_fields.items() if v}


def build_sanitized_image(image_to_send):
    """
    Run the confidence-driven Privacy Engine, returning (sanitized_image, summary_dict).
    Only called when Privacy Mode is on and the current mode could call a cloud API.
    The original `image_to_send` object itself is never mutated.

    All the actual detection/fusion/redaction logic now lives in the `privacy/` package
    (layout + OCR + regex detectors, fused by confidence, plus user overrides and safety
    nets) — this function just builds the config from session_state, calls the engine,
    and reshapes the PrivacyReport into the dict the rest of the UI/PDF code expects.
    """
    config = PrivacyConfig(
        enabled=True,
        fields_enabled=dict(st.session_state.privacy_fields),
        method=st.session_state.privacy_method,
        manual_top_pct=st.session_state.privacy_manual_top_pct,
        manual_bottom_pct=st.session_state.privacy_manual_bottom_pct,
        smart_rx_header=st.session_state.privacy_smart_rx_header,
        custom_regions=list(st.session_state.privacy_custom_regions),
        language=st.session_state.ocr_language,
        psm=st.session_state.ocr_psm,
        strategy=st.session_state.privacy_strategy,
        auto_mask_threshold=st.session_state.privacy_auto_mask_threshold,
    )

    sanitized, report = _privacy_engine.process(image_to_send, config)

    summary = {
        "enabled": True,
        "strategy": config.strategy,
        "method": REDACTION_METHODS.get(config.method, config.method),
        "fields_removed": report.fields_removed_labels,
        "region_count": report.region_count,
        "low_confidence": report.ocr_low_confidence,
        "manual_used": bool(config.manual_top_pct or config.manual_bottom_pct or config.custom_regions),
        "used_fallback_baseline": report.used_fallback_baseline,
        "overlaps_medicine": report.overlaps_medicine,
        "coverage_pct": report.coverage_pct,
        "high_coverage": report.high_coverage,
        "rx_marker_found": report.rx_marker_found,
        "detectors_used": report.detectors_used,
        "masked_fields": report.masked_fields,
        "low_confidence_regions": report.low_confidence_regions,
    }
    return sanitized, summary


if st.session_state.tesseract_cmd_path:
    set_tesseract_cmd(st.session_state.tesseract_cmd_path)

TESSERACT_READY = tesseract_is_available()
API_READY = api_is_configured(current_api_config())
_privacy_engine = PrivacyEngine()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="font-size:2rem;">💊</div>
            <div>
                <div style="font-weight:800;font-size:1.05rem;">MediScan</div>
                <div style="font-size:0.72rem;opacity:0.75;">Prescription Recognition System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    pages = ["Dashboard", "Analytics", "Settings", "About"]
    icons = ["speedometer2", "bar-chart-line", "gear", "info-circle"]

    if HAS_OPTION_MENU:
        nav_bg = "#131C2C" if st.session_state.theme == "dark" else "transparent"
        nav_text = "#EAF1F7" if st.session_state.theme == "dark" else "#152331"
        selected_page = option_menu(
            menu_title=None,
            options=pages,
            icons=icons,
            default_index=pages.index(st.session_state.current_page),
            styles={
                "container": {"padding": "0", "background-color": nav_bg},
                "icon": {"color": "#0E9488", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "font-weight": "600",
                    "text-align": "left",
                    "margin": "3px 0",
                    "border-radius": "10px",
                    "padding": "10px 12px",
                    "color": nav_text,
                },
                "nav-link-selected": {"background-color": "#0E9488", "color": "white"},
            },
        )
    else:
        selected_page = st.radio("Navigation", pages, index=pages.index(st.session_state.current_page))

    st.session_state.current_page = selected_page

    st.markdown("---")
    ui.render_theme_toggle()

    st.markdown("---")
    st.markdown("##### 🔌 System Status")

    mode_labels = {
        "tesseract": "🖥️ Tesseract (Offline, research)",
        "api": "☁️ Cloud API (direct, research)",
        "hybrid": "🔀 Hybrid (OCR + AI verify) — default",
    }
    st.markdown(f"**Mode:** {mode_labels.get(st.session_state.analysis_mode, st.session_state.analysis_mode)}")

    if TESSERACT_READY:
        st.markdown("🟢 **Tesseract OCR:** Ready (offline)")
    else:
        st.markdown("🔴 **Tesseract OCR:** Not found")

    provider_label = PROVIDERS.get(st.session_state.api_provider, PROVIDERS["custom"]).label
    if API_READY:
        st.markdown(f"🟢 **Cloud API ({provider_label}):** Connected")
    else:
        st.markdown(f"🟡 **Cloud API ({provider_label}):** Not configured")

    if st.session_state.analysis_mode == "hybrid":
        if not TESSERACT_READY and not API_READY:
            st.caption("⚠️ Neither engine is ready — Hybrid mode needs at least one configured.")
        elif not API_READY:
            st.caption("Hybrid will use raw Tesseract results only until a Cloud API key is added.")
        elif not TESSERACT_READY:
            st.caption("Hybrid will use direct AI extraction only until Tesseract is installed.")

    if st.session_state.pipeline_result:
        st.markdown("🟢 **Image Pipeline:** Processed")
    else:
        st.markdown("⚪ **Image Pipeline:** Idle")

    if st.session_state.analysis_result and st.session_state.analysis_result.success:
        st.markdown("🟢 **Analysis:** Complete")
    else:
        st.markdown("⚪ **Analysis:** Idle")

    st.markdown("---")
    if st.button("🔄 Start New Prescription", use_container_width=True):
        reset_session()
        st.rerun()

    st.caption("MediScan v4.0 · Tesseract + Any API + Hybrid · by Malik Usman")


# --------------------------------------------------------------------------- #
# DASHBOARD PAGE
# --------------------------------------------------------------------------- #
def render_dashboard():
    mode_note = {
        "tesseract": "Currently running in Tesseract-only research mode (100% offline).",
        "api": "Currently running in direct Cloud API research mode.",
        "hybrid": "Currently running in Hybrid mode — Tesseract extracts first, then AI verifies "
                  "the result against the image before you review it.",
    }[st.session_state.analysis_mode]
    ui.render_hero(
        "Prescription Recognition System",
        "Upload a prescription photo to automatically extract medicines, dosages, and usage "
        f"instructions. {mode_note} Change the engine anytime in Settings.",
        chips=["OpenCV DIP Pipeline", "Tesseract OCR", "Fuzzy Matching", "AI Verification", "PDF Reports"],
        show_offline_pill=(st.session_state.analysis_mode == "tesseract"),
    )

    # ---------------- Step 1: Upload ---------------- #
    ui.render_section_header("📤", "Step 1 — Upload Prescription Image", "Supported formats: JPG, PNG, JPEG")

    if st.session_state.analysis_mode in ("api", "hybrid"):
        st.markdown(
            '<div class="pr-card" style="border-left: 4px solid #E2A33D;">'
            '<h4>💡 Best practice: cover or crop out patient details before uploading</h4>'
            '<div class="section-sub" style="margin-left:0;">'
            "This app's automatic Privacy Protection is a local, best-effort safety net — it "
            "helps, but it isn't perfect, especially on handwritten prescriptions. "
            "<b>The most reliable way to keep patient information private is not to include it "
            "in the photo at all.</b> Before uploading, if you're using Cloud API or Hybrid mode:"
            "<ul style='margin:8px 0 4px 0;'>"
            "<li>Physically cover the name/address/date section with a piece of paper before taking the photo, or</li>"
            "<li>Crop the image (in your phone's gallery or any photo editor) to exclude that section, or</li>"
            "<li>Ask whoever wrote the prescription to omit identifying details if this is for demo/testing</li>"
            "</ul>"
            "Automatic and manual redaction below are a second layer of protection, not a substitute "
            "for this."
            "</div></div>",
            unsafe_allow_html=True,
        )

    upload_col, preview_col = st.columns([1.1, 1])

    with upload_col:
        st.markdown('<div class="pr-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or browse a prescription image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if file_bytes != st.session_state.uploaded_file_bytes:
                # New file uploaded -> reset downstream state
                st.session_state.uploaded_file_bytes = file_bytes
                st.session_state.original_image = Image.open(uploaded_file)
                st.session_state.file_metadata = dip.get_image_metadata(
                    file_bytes, st.session_state.original_image
                )
                st.session_state.pipeline_result = None
                st.session_state.analysis_result = None
                st.success("Prescription image uploaded successfully.")
        elif st.session_state.original_image is None:
            st.info("Upload a prescription image to begin the analysis pipeline.")
        st.markdown("</div>", unsafe_allow_html=True)

    with preview_col:
        if st.session_state.original_image is not None:
            st.markdown('<div class="pr-card"><h4>📋 Image Preview & Metadata</h4>', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            meta = st.session_state.file_metadata
            m1, m2 = st.columns(2)
            m1.metric("Dimensions", meta["Dimensions"])
            m2.metric("File Size", meta["File Size"])
            m3, m4 = st.columns(2)
            m3.metric("Color Mode", meta["Color Mode"])
            m4.metric("Megapixels", meta["Megapixels"])
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.original_image is None:
        return  # Nothing further to show until an image exists

    # ---------------- Step 2: DIP Pipeline ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header(
        "🧪", "Step 2 — Digital Image Processing Pipeline",
        "Grayscale → Median Filter → CLAHE Contrast Enhancement → Sharpening → Adaptive Thresholding"
    )

    run_pipeline_clicked = st.button("▶️ Run Image Processing Pipeline", type="primary")
    if run_pipeline_clicked:
        with st.spinner("Processing prescription image through the DIP pipeline..."):
            time.sleep(0.3)  # tiny UX pause so the spinner is visible even on fast machines
            st.session_state.pipeline_result = dip.run_pipeline(st.session_state.original_image)
        st.success(
            f"Image processing complete in {st.session_state.pipeline_result.total_time_ms:.1f} ms."
        )

    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result
        captions = {
            "Original": "Raw uploaded prescription",
            "Grayscale": "Single-channel conversion",
            "Noise Reduced": "Median filter denoising",
            "Enhanced": "CLAHE contrast enhancement",
            "Final Processed": "Sharpened + adaptive threshold",
        }
        cols = st.columns(5)
        for col, (stage_name, stage_img) in zip(cols, result.stages.items()):
            with col:
                st.markdown('<div class="pr-card">', unsafe_allow_html=True)
                ui.render_stage_image_card(stage_img, stage_name, captions.get(stage_name, ""))
                st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("⏱️ View per-stage processing time"):
            timing_df = pd.DataFrame(
                {"Stage": list(result.timings_ms.keys()), "Time (ms)": [round(v, 3) for v in result.timings_ms.values()]}
            )
            st.dataframe(timing_df, use_container_width=True, hide_index=True)

    else:
        return  # Don't show OCR step until pipeline has run

    # ---------------- Step 3: Analysis ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    mode_titles = {
        "tesseract": "Step 3 — Offline OCR Analysis (Tesseract, research mode)",
        "api": "Step 3 — Direct Cloud AI Analysis (research mode)",
        "hybrid": "Step 3 — Hybrid Analysis (Tesseract → AI Verification)",
    }
    ui.render_section_header("🔎", mode_titles.get(st.session_state.analysis_mode, "Step 3 — Analysis"),
                              "Choose which processed image version to analyze — engine is set in Settings")

    st.markdown('<div class="pr-card">', unsafe_allow_html=True)
    stage_options = ["Original", "Enhanced", "Final Processed"]
    if st.session_state.analysis_mode == "hybrid":
        col_ocr, col_ai = st.columns(2)
        with col_ocr:
            ocr_stage_choice = st.selectbox(
                "🖥️ Image for Tesseract OCR",
                options=stage_options,
                index=stage_options.index(st.session_state.ocr_image_stage),
                help="Tesseract reads binarized/high-contrast text best — 'Final Processed' is usually the right choice.",
            )
        with col_ai:
            ai_stage_choice = st.selectbox(
                "☁️ Image for AI Verification",
                options=stage_options,
                index=stage_options.index(st.session_state.ai_image_stage),
                help="Vision AI models generally read a natural, less-processed photo better than a "
                     "binarized one — 'Original' or 'Enhanced' usually work better here than 'Final Processed'.",
            )
        st.caption(
            "Hybrid mode intentionally uses two different image versions — Tesseract and the AI "
            "verification step each get whichever version they individually perform best on."
        )
    elif st.session_state.analysis_mode == "tesseract":
        ocr_stage_choice = st.selectbox(
            "Image to analyze (Tesseract OCR)",
            options=stage_options,
            index=stage_options.index(st.session_state.ocr_image_stage),
            help="Tesseract reads binarized/high-contrast text best — 'Final Processed' is usually the right choice.",
        )
        ai_stage_choice = st.session_state.ai_image_stage
    else:  # api mode
        ai_stage_choice = st.selectbox(
            "Image to send to the AI provider",
            options=stage_options,
            index=stage_options.index(st.session_state.ai_image_stage),
            help="Vision AI models generally read a natural, less-processed photo better — try "
                 "'Original' or 'Enhanced' rather than the binarized 'Final Processed' version.",
        )
        ocr_stage_choice = st.session_state.ocr_image_stage

    st.session_state.ocr_image_stage = ocr_stage_choice
    st.session_state.ai_image_stage = ai_stage_choice
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Privacy status panel (only relevant when a cloud API might be called) ----
    if privacy_applies_to_current_mode():
        if st.session_state.privacy_enabled:
            checked_labels = [PII_FIELD_META[k]["label"] for k in enabled_privacy_field_keys()]
            manual_bits = []
            if st.session_state.privacy_manual_top_pct > 0:
                manual_bits.append(f"top {st.session_state.privacy_manual_top_pct}%")
            if st.session_state.privacy_manual_bottom_pct > 0:
                manual_bits.append(f"bottom {st.session_state.privacy_manual_bottom_pct}%")
            if st.session_state.privacy_custom_regions:
                manual_bits.append(f"{len(st.session_state.privacy_custom_regions)} custom region(s)")
            manual_note = f" Manual redaction is also active ({' + '.join(manual_bits)} of the image)." if manual_bits else \
                " No manual region redaction is set — for handwritten prescriptions, consider enabling it in Settings as automatic detection can miss cursive text."
            st.markdown(
                '<div class="pr-card">'
                '<h4>🛡️ Privacy Protection: <span style="color:#22B378;">Enabled</span></h4>'
                f'<div class="section-sub" style="margin-left:0;">Before anything is sent to the cloud AI, '
                f'these fields will be detected and redacted locally: <b>{", ".join(checked_labels) or "none selected"}</b>.'
                f'{manual_note} The original, unmodified image never leaves this machine — only the sanitized '
                'copy is sent. Configure this in Settings.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pr-card">'
                '<h4>🛡️ Privacy Protection: <span style="color:#E85D4A;">Disabled</span></h4>'
                '<div class="section-sub" style="margin-left:0;">The image will be sent to the cloud AI '
                'as-is, with no local redaction. Enable Privacy Protection in Settings if the prescription '
                'contains real patient information.</div></div>',
                unsafe_allow_html=True,
            )

    analyze_clicked = st.button("🔎 Analyze Prescription", type="primary")

    if analyze_clicked:
        if st.session_state.analysis_mode == "tesseract" and not TESSERACT_READY:
            st.error("⚠️ Tesseract OCR engine not found on this system. See the **Settings** page.")
        elif st.session_state.analysis_mode == "api" and not API_READY:
            st.error("⚠️ No cloud API configured. Pick a provider and add a key in the **Settings** page.")
        elif st.session_state.analysis_mode == "hybrid" and not TESSERACT_READY and not API_READY:
            st.error("⚠️ Neither engine is available. Configure Tesseract and/or a Cloud API key in **Settings**.")
        else:
            ocr_image = st.session_state.pipeline_result.stages[st.session_state.ocr_image_stage]
            ai_image_raw = st.session_state.pipeline_result.stages[st.session_state.ai_image_stage]

            image_for_api = None
            privacy_summary = None
            if privacy_applies_to_current_mode() and st.session_state.privacy_enabled:
                with st.spinner("Scanning locally for personal information before any cloud call..."):
                    image_for_api, privacy_summary = build_sanitized_image(ai_image_raw)
                st.session_state.privacy_last_sanitized_image = image_for_api
            elif privacy_applies_to_current_mode():
                image_for_api = ai_image_raw
                privacy_summary = {"enabled": False, "method": None, "fields_removed": [], "region_count": 0}
                st.session_state.privacy_last_sanitized_image = None
            else:
                st.session_state.privacy_last_sanitized_image = None
            st.session_state.privacy_last_summary = privacy_summary

            spinner_text = {
                "tesseract": "Running local OCR + rule-based extraction...",
                "api": "Sending the image to the cloud AI provider... this can take a few seconds.",
                "hybrid": "Running Tesseract OCR + fuzzy matching, then AI verification against the image...",
            }[st.session_state.analysis_mode]
            with st.spinner(spinner_text):
                result = analyze_prescription(
                    image=ocr_image,
                    image_for_api=image_for_api,
                    original_image=st.session_state.original_image,
                    mode=st.session_state.analysis_mode,
                    ocr_language=st.session_state.ocr_language,
                    ocr_psm=st.session_state.ocr_psm,
                    api_config=current_api_config(),
                )
                st.session_state.analysis_result = result

    analysis = st.session_state.analysis_result
    if analysis is None:
        return

    if analysis.fallback_reason:
        st.info(f"🔀 {analysis.fallback_reason}")

    priv = st.session_state.privacy_last_summary
    if priv and priv.get("enabled"):
        removed = priv["fields_removed"]
        st.success(
            f"🛡️ Privacy Protection applied ({priv['method']}) — redacted: "
            f"{', '.join(removed) if removed else 'no PII detected'}. Only the sanitized image was sent to the cloud API."
        )
        if priv.get("detectors_used"):
            st.caption(f"🔬 Detectors that contributed evidence: {', '.join(priv['detectors_used'])}")
        if priv.get("masked_fields"):
            with st.expander("📊 Confidence breakdown (why each region was redacted)"):
                conf_df = pd.DataFrame(
                    [
                        {
                            "Field": m["field_label"],
                            "Fused Confidence": f"{m['confidence']*100:.0f}%",
                            "Detected By": " + ".join(m["sources"]),
                        }
                        for m in priv["masked_fields"]
                    ]
                )
                st.dataframe(conf_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Fused confidence combines independent evidence (layout structure, OCR labels, "
                    "regex patterns, or your manual/custom overrides) rather than trusting any single "
                    "signal alone."
                )
        if priv.get("low_confidence_regions"):
            with st.expander(f"🔎 {len(priv['low_confidence_regions'])} low-confidence region(s) NOT auto-redacted (below threshold)"):
                for r in priv["low_confidence_regions"]:
                    st.caption(f"• {r['field_label']} — confidence {r['confidence']*100:.0f}% (below auto-mask threshold)")
                st.caption(
                    "These were flagged as possible PII but not confident enough to auto-redact. "
                    "Review the sanitized preview and consider adding a Custom Region in Settings if "
                    "one of these turns out to be real."
                )
        if priv.get("used_fallback_baseline"):
            st.warning(
                "⚠️ No PII was detected by any active method (targeted/whitelist detection, manual "
                "top/bottom, custom regions) — applied a **guaranteed top-10% fallback redaction** "
                "instead of sending the image completely unredacted. Check the preview: this default "
                "band may not fully cover this specific layout — configure Manual/Custom regions in "
                "Settings for a more reliable result on this prescription pad."
            )
        if priv.get("rx_marker_found") is True:
            st.caption("🎯 Found the printed 'Rx' marker — header was redacted precisely up to it.")
        elif priv.get("rx_marker_found") is False:
            st.caption(
                f"🎯 Could not locate an 'Rx' marker on this image (common on noisy/handwritten scans) "
                f"— used the fallback top {st.session_state.privacy_manual_top_pct}% instead."
            )
        if priv.get("low_confidence") and not priv.get("manual_used"):
            st.warning(
                "⚠️ Automatic detection read very little text from this image — common on handwritten "
                "or heavily degraded scans. It may have missed real PII. **Check the preview below "
                "carefully**, and consider enabling Manual Region Redaction in Settings as a more "
                "dependable fallback for this kind of image."
            )
        if priv.get("high_coverage"):
            st.error(
                f"🚨 Redaction covered ~{priv.get('coverage_pct', '?')}% of this image. That's high enough "
                "that clinical content (medicine names, dosages) may have been blacked out along with — or "
                "instead of — PII. This usually means OCR couldn't read this image well enough for automatic "
                "detection to work reliably. **Check the preview below.** For heavily degraded/handwritten "
                "scans, Manual Region Redaction (position-based, doesn't depend on OCR) is usually more "
                "reliable — configure it in Settings."
            )
        if priv.get("overlaps_medicine"):
            st.error(
                "🚨 Your manual/custom redaction region(s) appear to overlap text that looks like a "
                "medicine or dosage entry. Manual regions are pure geometry — they don't know what's "
                "underneath — so this may mean the AI won't be able to see part of the prescription "
                "itself. **Check the preview below and narrow your region in Settings if it's covering "
                "clinical content.**"
            )
        if st.session_state.privacy_last_sanitized_image is not None:
            preview_expanded = bool(
                (priv.get("low_confidence") and not priv.get("manual_used"))
                or priv.get("overlaps_medicine")
                or priv.get("high_coverage")
            )
            with st.expander("👁️ Preview sanitized image (exactly what the cloud AI received)", expanded=preview_expanded):
                st.image(st.session_state.privacy_last_sanitized_image, use_container_width=True)

    if not analysis.success:
        st.error(f"⚠️ Analysis failed: {analysis.error_message}")
        if analysis.raw_response:
            with st.expander("Show raw engine response"):
                st.code(analysis.raw_response)
        return

    engine_label = ENGINE_LABELS.get(analysis.engine, analysis.engine)
    st.success(f"✅ Analysis complete via **{engine_label}** in {analysis.processing_time_s:.2f} seconds.")
    if analysis.prescription_quality.lower() == "poor":
        st.warning("⚠️ The prescription image quality was assessed as **Poor** — results may be unreliable. "
                   "Consider re-uploading a clearer, better-lit photo.")

    # ---------------- Step 4: Medicine table ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header("💊", "Step 4 — Extracted Medicines", "Search below, then fix any mistakes in the editable table")

    SOURCE_LABELS = {
        "ocr": "🖥️ OCR",
        "ocr-unverified": "⚠️ OCR (unverified)",
        "kept": "✅ AI: kept as-is",
        "corrected": "✏️ AI: corrected",
        "added": "➕ AI: added",
        "ai": "🤖 AI",
    }

    med_df = pd.DataFrame(
        [
            {
                "Medicine Name": m.name,
                "Strength": m.strength or "",
                "Frequency": m.frequency or "",
                "Confidence (%)": m.confidence,
                "Source": SOURCE_LABELS.get(m.source, m.source or ""),
            }
            for m in analysis.medicines
        ]
    ) if analysis.medicines else pd.DataFrame(columns=["Medicine Name", "Strength", "Frequency", "Confidence (%)", "Source"])

    if analysis.medicines:
        search_col, filter_col = st.columns([2, 1])
        with search_col:
            search_term = st.text_input("🔍 Search medicine name", placeholder="e.g. Panadol, Augmentin...")
        with filter_col:
            min_conf = st.slider("Minimum confidence", 0, 100, 0)

        filtered_df = med_df.copy()
        if search_term:
            filtered_df = filtered_df[filtered_df["Medicine Name"].str.contains(search_term, case=False, na=False)]
        filtered_df = filtered_df[filtered_df["Confidence (%)"] >= min_conf]

        st.markdown('<div class="pr-card"><h4>👀 Preview (read-only, respects search/filter above)</h4>', unsafe_allow_html=True)
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence (%)", min_value=0, max_value=100, format="%d%%"
                )
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No medicines were confidently identified in this prescription. Try a clearer image, "
                   "a different pipeline stage in Step 3, or add them manually below.")

    st.markdown(
        '<div class="pr-card"><h4>✏️ Edit / Correct Extracted Medicines</h4>'
        '<div class="section-sub" style="margin-left:0;">'
        "OCR and AI both make mistakes — fix a name, strength, or frequency here, delete a wrong "
        "row, or add a medicine the engine missed. The <b>Source</b> column shows what OCR found "
        "versus what AI kept/corrected/added during verification. Use the ⊕ row at the bottom to "
        "add a new entry. Click <b>Apply Edits</b> to save changes before generating KPIs or the PDF report."
        "</div></div>",
        unsafe_allow_html=True,
    )
    edited_df = st.data_editor(
        med_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="medicine_editor",
        column_config={
            "Medicine Name": st.column_config.TextColumn("Medicine Name", required=True),
            "Strength": st.column_config.TextColumn("Strength", help="e.g. 500mg"),
            "Frequency": st.column_config.TextColumn("Frequency", help="e.g. 1-0-1, BD, twice daily"),
            "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", min_value=0, max_value=100, step=1),
            "Source": st.column_config.TextColumn("Source", help="What produced/adjusted this row", disabled=True),
        },
    )

    if st.button("💾 Apply Edits", type="primary"):
        cleaned = edited_df.copy()
        cleaned = cleaned[cleaned["Medicine Name"].astype(str).str.strip() != ""]
        new_medicines = []
        for _, row in cleaned.iterrows():
            conf_val = row.get("Confidence (%)", 0)
            try:
                conf_int = int(conf_val) if pd.notna(conf_val) else 0
            except (ValueError, TypeError):
                conf_int = 0
            new_medicines.append(
                MedicineItem(
                    name=str(row["Medicine Name"]).strip(),
                    strength=(str(row["Strength"]).strip() or None) if pd.notna(row.get("Strength")) else None,
                    frequency=(str(row["Frequency"]).strip() or None) if pd.notna(row.get("Frequency")) else None,
                    confidence=max(0, min(100, conf_int)),
                )
            )
        analysis.medicines = new_medicines
        if new_medicines:
            analysis.overall_confidence = round(sum(m.confidence for m in new_medicines) / len(new_medicines))
        st.session_state.analysis_result = analysis
        st.success(f"✅ Saved {len(new_medicines)} medicine row(s). KPIs and the PDF report will use these edited values.")
        st.rerun()

    if analysis.doctor_notes:
        st.markdown('<div class="pr-card"><h4>📝 Additional Notes (raw OCR lines)</h4>', unsafe_allow_html=True)
        st.write(analysis.doctor_notes)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🧾 View full raw engine response / OCR text"):
        st.code(analysis.raw_response or "(no text detected)")

    # ---------------- Step 5: KPIs ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header("📈", "Step 5 — Analysis Summary")

    quality_metrics = dip.estimate_prescription_quality(
        dip.to_grayscale(dip.pil_to_cv2(st.session_state.original_image))
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.render_kpi_card("💊", str(len(analysis.medicines)), "Medicines Detected")
    with k2:
        total_time = st.session_state.pipeline_result.total_time_ms / 1000 + analysis.processing_time_s
        ui.render_kpi_card("⏱️", f"{total_time:.2f}s", "Total Processing Time")
    with k3:
        ui.render_kpi_card("🎯", f"{analysis.overall_confidence}%", "Recognition Confidence")
    with k4:
        ui.render_kpi_card("📋", analysis.prescription_quality, "Prescription Quality")

    # ---------------- Step 6: Report generation ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header("📄", "Step 6 — Download Report", "Export a complete PDF summary of this analysis")

    st.markdown('<div class="pr-card">', unsafe_allow_html=True)
    patient_ref = st.text_input("Reference / Patient ID (optional)", placeholder="e.g. OPD-2026-0142")
    if st.button("📥 Generate PDF Report", type="primary"):
        with st.spinner("Compiling PDF report..."):
            pdf_bytes = generate_pdf_report(
                original_image=st.session_state.original_image,
                processed_image=st.session_state.pipeline_result.stages["Final Processed"],
                medicines=[
                    {
                        "name": m.name,
                        "strength": m.strength,
                        "frequency": m.frequency,
                        "confidence": m.confidence,
                    }
                    for m in analysis.medicines
                ],
                overall_confidence=analysis.overall_confidence,
                prescription_quality=analysis.prescription_quality,
                processing_time_s=total_time,
                doctor_notes=analysis.doctor_notes,
                patient_reference=patient_ref or None,
                engine=analysis.engine,
                privacy_info=(
                    {"mode": "not_applicable"} if st.session_state.analysis_mode == "tesseract"
                    else (st.session_state.privacy_last_summary or {"enabled": False, "method": None, "fields_removed": []})
                ),
            )
        st.download_button(
            "⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"prescription_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# ANALYTICS PAGE
# --------------------------------------------------------------------------- #
def render_analytics():
    ui.render_hero(
        "Analytics Dashboard",
        "Visual insights into the most recent prescription analysis session.",
        chips=["Confidence Breakdown", "Processing Time", "Quality Assessment"],
    )

    analysis = st.session_state.analysis_result
    pipeline = st.session_state.pipeline_result

    if not analysis or not analysis.success or not pipeline:
        st.info("ℹ️ Run a full analysis from the **Dashboard** page first to populate analytics.")
        return

    plot_bg = "#131C2C" if st.session_state.theme == "dark" else "white"
    font_color = "#EAF1F7" if st.session_state.theme == "dark" else "#152331"

    k1, k2, k3, k4 = st.columns(4)
    total_time = pipeline.total_time_ms / 1000 + analysis.processing_time_s
    with k1:
        ui.render_kpi_card("💊", str(len(analysis.medicines)), "Medicines Detected")
    with k2:
        ui.render_kpi_card("⏱️", f"{total_time:.2f}s", "Total Processing Time")
    with k3:
        ui.render_kpi_card("🎯", f"{analysis.overall_confidence}%", "Overall Confidence")
    with k4:
        ui.render_kpi_card("📋", analysis.prescription_quality, "Prescription Quality")

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="pr-card"><h4>💊 Per-Medicine Confidence</h4>', unsafe_allow_html=True)
        if analysis.medicines:
            names = [m.name for m in analysis.medicines]
            confs = [m.confidence for m in analysis.medicines]
            colors = ["#22B378" if c >= 80 else "#E2A33D" if c >= 50 else "#E85D4A" for c in confs]
            fig = go.Figure(go.Bar(x=confs, y=names, orientation="h", marker_color=colors))
            fig.update_layout(
                height=max(260, 40 * len(names)),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Confidence (%)",
                xaxis_range=[0, 100],
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=font_color),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No medicines to chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="pr-card"><h4>🎯 Overall Recognition Confidence</h4>', unsafe_allow_html=True)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=analysis.overall_confidence,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0E9488"},
                    "steps": [
                        {"range": [0, 50], "color": "#3A2A26" if st.session_state.theme == "dark" else "#FBEAE6"},
                        {"range": [50, 80], "color": "#3A3222" if st.session_state.theme == "dark" else "#FCF2DE"},
                        {"range": [80, 100], "color": "#1E3A30" if st.session_state.theme == "dark" else "#E4F5EE"},
                    ],
                },
            )
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor=plot_bg, font=dict(color=font_color))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pr-card"><h4>⏱️ Processing Time Breakdown</h4>', unsafe_allow_html=True)
    timing_labels = list(pipeline.timings_ms.keys()) + [f"{analysis.engine.capitalize()} Analysis"]
    timing_values = [round(v, 1) for v in pipeline.timings_ms.values()] + [round(analysis.processing_time_s * 1000, 1)]
    fig = go.Figure(go.Bar(x=timing_labels, y=timing_values, marker_color="#3B82F6"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Time (ms)",
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        font=dict(color=font_color),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pr-card"><h4>🖼️ Local Image Quality Heuristics</h4>', unsafe_allow_html=True)
    quality_metrics = dip.estimate_prescription_quality(
        dip.to_grayscale(dip.pil_to_cv2(st.session_state.original_image))
    )
    q1, q2, q3 = st.columns(3)
    q1.metric("Sharpness (Laplacian Var.)", quality_metrics["sharpness"])
    q2.metric("Brightness (Mean Pixel)", quality_metrics["brightness"])
    q3.metric("Contrast (Std. Dev.)", quality_metrics["contrast"])
    st.caption(
        "These are computed locally via OpenCV and drive the offline Prescription Quality "
        "classification used above — no cloud AI call involved."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# SETTINGS PAGE
# --------------------------------------------------------------------------- #
def render_settings():
    ui.render_hero("Settings", "Choose your analysis engine, connect any cloud AI provider, and control privacy.",
                    chips=["Tesseract", "Any API Provider", "Hybrid", "Privacy Protection"])

    st.markdown('<div class="pr-card"><h4>⚙️ Analysis Mode</h4>', unsafe_allow_html=True)
    mode_options = {
        "hybrid": "🔀 Hybrid — Tesseract extracts, then AI verifies it against the image (recommended default)",
        "tesseract": "🖥️ Tesseract OCR only — fully offline, free, no API key (research/comparison)",
        "api": "☁️ Cloud API only — sends the image straight to the AI, no OCR pass (research/comparison)",
    }
    mode_choice = st.radio(
        "Choose how prescriptions are analyzed",
        options=list(mode_options.keys()),
        format_func=lambda k: mode_options[k],
        index=list(mode_options.keys()).index(st.session_state.analysis_mode) if st.session_state.analysis_mode in mode_options else 0,
    )
    if mode_choice == "hybrid":
        st.caption(
            "Hybrid pipeline: Tesseract OCR + local fuzzy medicine-name matching runs first. If it "
            "finds a usable draft, the configured cloud API checks it against the actual image and "
            "corrects/completes it. If Tesseract finds nothing (or isn't installed), Hybrid escalates "
            "straight to full AI extraction instead. If no API is configured, you still get the raw "
            "Tesseract result — Hybrid never hard-fails as long as at least one engine works."
        )

    if st.button("💾 Save Analysis Mode", type="primary"):
        st.session_state.analysis_mode = mode_choice
        st.success("✅ Analysis mode saved.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Tesseract config ---------------- #
    if st.session_state.analysis_mode in ("tesseract", "hybrid"):
        st.markdown('<div class="pr-card"><h4>🖥️ Tesseract OCR Engine</h4>', unsafe_allow_html=True)
        if TESSERACT_READY:
            st.markdown("🟢 **Status:** Tesseract is installed and ready.")
        else:
            st.markdown("🔴 **Status:** Tesseract was not detected on this system.")

            if st.button("🔧 Try Automatic Install"):
                with st.spinner("Attempting to install Tesseract using your OS's package manager..."):
                    result = attempt_auto_install()
                st.session_state["_auto_install_result"] = result
                if result.success:
                    st.success("✅ Install command completed. Checking again...")
                    st.rerun()

            last_result = st.session_state.get("_auto_install_result")
            if last_result is not None and not last_result.success:
                st.warning("⚠️ Automatic install didn't complete. See the log and manual command below.")
                with st.expander("📜 Install log"):
                    st.code("\n".join(last_result.log) or "(no output)")
                if last_result.manual_command:
                    st.code(last_result.manual_command, language="bash")
                    st.caption("Run this command yourself in a terminal (as Administrator/with sudo if needed), then reopen this app.")

            st.caption(
                "'Try Automatic Install' works when winget (Windows), Homebrew (macOS), or apt/dnf "
                "(Linux, if running as root) is available. If it can't complete silently, copy the "
                "manual command shown and run it yourself in a terminal."
            )
            st.caption(
                "If you already installed it, Windows often doesn't add it to PATH. "
                "Paste the full path to tesseract.exe below (default install location is "
                "usually `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`)."
            )

        path_input = st.text_input(
            "Tesseract executable path (optional — only needed if not auto-detected)",
            value=st.session_state.tesseract_cmd_path,
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )
        if st.button("💾 Save Tesseract Path"):
            st.session_state.tesseract_cmd_path = path_input
            if path_input:
                set_tesseract_cmd(path_input)
            st.rerun()

        lang_options = ["eng", "eng+urd", "eng+ara", "eng+fra", "eng+spa"]
        lang_choice = st.selectbox(
            "OCR Language",
            options=lang_options,
            index=lang_options.index(st.session_state.ocr_language) if st.session_state.ocr_language in lang_options else 0,
            help="Additional languages require the matching Tesseract language pack to be installed "
                 "(e.g. 'tesseract-ocr-urd' on Debian/Ubuntu).",
        )

        psm_keys = list(PSM_PRESETS.keys())
        psm_choice = st.selectbox(
            "Page Segmentation Mode (PSM)",
            options=psm_keys,
            index=psm_keys.index(st.session_state.ocr_psm) if st.session_state.ocr_psm in psm_keys else 0,
            format_func=lambda k: f"{k} — {PSM_PRESETS[k]}",
            help="If OCR is extracting little or nothing, try '11 — Sparse text' or '4 — Single "
                 "column' instead of the default.",
        )

        if st.button("💾 Save OCR Settings"):
            st.session_state.ocr_language = lang_choice
            st.session_state.ocr_psm = psm_choice
            st.success("✅ OCR settings saved.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="pr-card"><h4>⚠️ A note on handwriting</h4>', unsafe_allow_html=True)
        st.markdown(
            """
            **Tesseract is a printed-text OCR engine** — it was not built to read cursive
            handwriting, and most real prescriptions are handwritten. On cursive handwriting it will
            often extract little or nothing usable no matter how the settings above are tuned —
            that's why **Hybrid mode** (the default) escalates straight to AI vision extraction
            whenever Tesseract comes up empty.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Cloud API provider config ---------------- #
    if st.session_state.analysis_mode in ("api", "hybrid"):
        st.markdown('<div class="pr-card"><h4>🔑 Cloud AI Provider</h4>', unsafe_allow_html=True)
        st.caption("Pick any OpenAI-compatible provider — not locked to one vendor. Groq, OpenAI, OpenRouter, "
                   "Together AI, or point 'Custom' at your own endpoint.")

        provider_keys = list(PROVIDERS.keys())
        provider_choice = st.selectbox(
            "Provider",
            options=provider_keys,
            index=provider_keys.index(st.session_state.api_provider) if st.session_state.api_provider in provider_keys else 0,
            format_func=lambda k: PROVIDERS[k].label,
        )
        provider_info = PROVIDERS[provider_choice]

        base_url_input = st.text_input(
            "API Base URL",
            value=st.session_state.api_base_url if st.session_state.api_provider == provider_choice else provider_info.base_url,
            placeholder="https://api.example.com/v1",
            help="Auto-filled per provider; edit for 'Custom' or if a provider changes its endpoint.",
        )

        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="Paste your API key here",
            help=provider_info.key_help,
        )

        model_default = st.session_state.api_model if st.session_state.api_provider == provider_choice else provider_info.default_model
        model_input = st.text_input(
            "Model name",
            value=model_default,
            placeholder=provider_info.default_model or "model-id",
            help=("Suggested models: " + ", ".join(provider_info.suggested_models)) if provider_info.suggested_models else
                 "Enter the exact model ID your endpoint expects.",
        )

        if st.button("💾 Save Cloud API Settings", type="primary"):
            st.session_state.api_provider = provider_choice
            st.session_state.api_base_url = base_url_input
            st.session_state.api_key = api_key_input
            st.session_state.api_model = model_input
            if api_key_input and base_url_input and model_input:
                st.success("✅ Cloud API settings saved.")
            else:
                st.warning("⚠️ Missing base URL, API key, or model — Cloud API / Hybrid analysis will be unavailable until all three are set.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Privacy Protection ---------------- #
    if st.session_state.analysis_mode in ("api", "hybrid"):
        st.markdown('<div class="pr-card"><h4>🛡️ Privacy Protection</h4>', unsafe_allow_html=True)
        st.markdown(
            "Runs **entirely locally**, before anything is sent to a cloud AI provider. It scans the "
            "image (using the same local Tesseract engine — no network call) for likely personal "
            "information, then redacts only those regions. The clinical body of the prescription — "
            "medicine names, dosages, frequencies, diagnosis, instructions — is left untouched so the "
            "AI can still read it. **Offline Tesseract-only mode never needs this**, since nothing "
            "ever leaves your machine in that mode."
        )

        privacy_on = st.toggle(
            "Enable Privacy Protection before sending images to a cloud AI provider",
            value=st.session_state.privacy_enabled,
        )

        st.markdown("**Redaction strategy:**")
        strategy_choice = st.radio(
            "How should the system decide what to redact?",
            options=["targeted", "whitelist"],
            format_func=lambda k: (
                "Targeted — redact only detected PII fields (default, keeps most of the image visible)"
                if k == "targeted" else
                "Whitelist — redact everything EXCEPT lines recognized as medicine/dosage/instructions"
            ),
            index=["targeted", "whitelist"].index(st.session_state.privacy_strategy),
        )
        threshold = st.session_state.privacy_auto_mask_threshold
        if strategy_choice == "whitelist":
            st.warning(
                "⚠️ **Tested trade-off:** Whitelist mode is safer against missing unrecognized PII on "
                "readable text, since it redacts by default rather than only on a match. But on heavily "
                "degraded or very cursive scans, we found it can end up redacting 90%+ of the image — "
                "**including the medicine/dosage lines themselves** — because OCR can't reliably tell "
                "clinical text from anything else either. A coverage check will warn you after analysis "
                "if this happens, but always check the sanitized preview."
            )
        else:
            threshold = st.slider(
                "Auto-mask confidence threshold",
                min_value=0.2, max_value=0.9, value=float(st.session_state.privacy_auto_mask_threshold), step=0.05,
                help="In Targeted mode, independent detectors (layout structure, OCR labels, regex "
                     "patterns) each vote on whether a region is sensitive, and their combined "
                     "confidence is compared against this threshold. Lower = redact more aggressively "
                     "(fewer missed PII, more false positives). Higher = redact more conservatively "
                     "(less clinical content at risk, more chance of missing something).",
            )

        st.markdown("**Fields to detect and redact (used in Targeted mode):**")
        field_cols = st.columns(2)
        new_field_values = {}
        field_keys = list(PII_FIELD_META.keys())
        half = (len(field_keys) + 1) // 2
        for col, keys_subset in zip(field_cols, [field_keys[:half], field_keys[half:]]):
            with col:
                for key in keys_subset:
                    new_field_values[key] = st.checkbox(
                        PII_FIELD_META[key]["label"],
                        value=st.session_state.privacy_fields.get(key, PII_FIELD_META[key]["default"]),
                        key=f"privacy_field_{key}",
                    )

        method_keys = list(REDACTION_METHODS.keys())
        method_choice = st.radio(
            "Redaction method",
            options=method_keys,
            format_func=lambda k: REDACTION_METHODS[k],
            index=method_keys.index(st.session_state.privacy_method) if st.session_state.privacy_method in method_keys else 0,
            horizontal=True,
        )

        st.markdown("---")
        st.markdown("**🖐️ Manual Region Redaction (recommended for handwritten prescriptions)**")
        st.warning(
            "⚠️ **Tested limitation:** on handwritten/cursive or heavily degraded scans, automatic "
            "OCR-based detection above can fail to properly locate PII — Tesseract may read cursive "
            "text as scattered garbage fragments that don't actually cover the real name/address/date. "
            "Manual redaction below is **OCR-independent** — it blacks out a fixed top/bottom band by "
            "position alone, so it works even when OCR can't read the handwriting at all. Most "
            "prescription pads put patient info in a header block and physician license/signature in "
            "a footer block. Check the sanitized preview after analysis to confirm the bands actually "
            "cover the right areas for your specific prescription layout — layouts vary."
        )
        manual_col1, manual_col2 = st.columns(2)
        with manual_col1:
            top_pct = st.slider(
                "Redact top __% of image (used as fallback if Rx marker isn't found — see below)",
                min_value=0, max_value=45, value=int(st.session_state.privacy_manual_top_pct), step=1,
            )
        with manual_col2:
            bottom_pct = st.slider(
                "Redact bottom __% of image (footer: signature/license)",
                min_value=0, max_value=30, value=int(st.session_state.privacy_manual_bottom_pct), step=1,
            )

        smart_rx = st.checkbox(
            "🎯 Smart header redaction: redact up to the printed 'Rx' marker if it can be located, "
            "otherwise use the top % above as a fallback",
            value=st.session_state.privacy_smart_rx_header,
            help="Most prescription pads print 'Rx' between the patient-info header and the medication "
                 "body. If OCR can find that marker, this redacts precisely up to it instead of a blind "
                 "percentage — tested to work on clean/printed markers. On heavily degraded scans where "
                 "the marker itself is illegible, it automatically falls back to the top % slider instead "
                 "(also tested — confirmed graceful fallback rather than silently redacting nothing).",
        )

        if st.button("💾 Save Privacy Settings", type="primary"):
            st.session_state.privacy_enabled = privacy_on
            st.session_state.privacy_strategy = strategy_choice
            st.session_state.privacy_auto_mask_threshold = threshold
            st.session_state.privacy_fields = new_field_values
            st.session_state.privacy_method = method_choice
            st.session_state.privacy_manual_top_pct = top_pct
            st.session_state.privacy_manual_bottom_pct = bottom_pct
            st.session_state.privacy_smart_rx_header = smart_rx
            st.success("✅ Privacy settings saved.")
            st.rerun()

        st.markdown("---")
        st.markdown("**📐 Custom Region (for PII anywhere else — bottom, middle, side margin, etc.)**")
        st.caption(
            "Top/bottom bands assume patient info sits at the top or bottom of the page. If a "
            "prescription pad puts the name/address elsewhere — e.g. at the very bottom, or in a "
            "side margin — add a custom rectangle here instead. Values are percentages of the image "
            "so they scale to any photo resolution. Check the sanitized preview after analysis to "
            "confirm it lines up before trusting it."
        )

        if st.session_state.privacy_custom_regions:
            for i, region in enumerate(st.session_state.privacy_custom_regions):
                rcol1, rcol2 = st.columns([5, 1])
                with rcol1:
                    st.caption(
                        f"Region {i+1}: top {region['top_pct']}%, left {region['left_pct']}%, "
                        f"width {region['width_pct']}%, height {region['height_pct']}%"
                    )
                with rcol2:
                    if st.button("🗑️ Remove", key=f"remove_region_{i}"):
                        st.session_state.privacy_custom_regions.pop(i)
                        st.rerun()

        with st.expander("➕ Add a custom redaction region"):
            c1, c2, c3, c4 = st.columns(4)
            new_top = c1.number_input("Top %", min_value=0, max_value=100, value=0, key="new_region_top")
            new_left = c2.number_input("Left %", min_value=0, max_value=100, value=0, key="new_region_left")
            new_width = c3.number_input("Width %", min_value=1, max_value=100, value=100, key="new_region_width")
            new_height = c4.number_input("Height %", min_value=1, max_value=100, value=15, key="new_region_height")
            if st.button("Add region"):
                st.session_state.privacy_custom_regions.append(
                    {"top_pct": new_top, "left_pct": new_left, "width_pct": new_width, "height_pct": new_height}
                )
                st.rerun()

        st.caption(
            "Automatic detection is a local, heuristic keyword/pattern matcher — like any PII scrubber "
            "it can occasionally miss something, and (as tested) can fail more broadly on handwriting. "
            "Manual top/bottom redaction is a blunter but more dependable safety net for those cases. "
            "Always check the sanitized preview shown after analysis before relying on either for real "
            "patient data. The original, unmodified image is never sent anywhere — only the sanitized copy is."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# ABOUT PAGE
# --------------------------------------------------------------------------- #
def render_about():
    ui.render_hero("About MediScan", "Hybrid Prescription Recognition System · by Malik Usman",
                    chips=["Python", "Streamlit", "OpenCV", "Tesseract OCR", "Any Cloud AI", "Privacy Protection"],
                    show_offline_pill=False)

    st.markdown(
        """
        <div class="pr-card">
        <h4>🎯 Project Overview</h4>
        This system automates the error-prone process of manually reading handwritten or printed
        medical prescriptions. It combines classical <b>Digital Image Processing</b> techniques with
        a choice of two recognition engines — the fully offline <b>Tesseract OCR</b> engine with a local
        rule-based/fuzzy-matching layer, or any OpenAI-compatible <b>cloud vision AI</b> provider — to identify
        medicine names, dosage strengths, and usage instructions in a structured, searchable format.
        A third <b>Hybrid mode</b> runs one engine and automatically falls back to the other if it
        fails, combining offline reliability with cloud-grade accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="pr-card">
            <h4>🧪 Image Processing Pipeline</h4>
            <ol>
            <li>Grayscale conversion</li>
            <li>Median filter noise reduction</li>
            <li>CLAHE contrast enhancement</li>
            <li>Unsharp-mask sharpening</li>
            <li>Adaptive Gaussian thresholding</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="pr-card">
            <h4>🛠️ Technology Stack</h4>
            <ul>
            <li>Streamlit — UI / dashboard</li>
            <li>OpenCV + NumPy + Pillow — image processing</li>
            <li>Tesseract OCR + pytesseract — offline text recognition</li>
            <li>Generic OpenAI-compatible HTTP client — works with Groq, OpenAI, OpenRouter, Together, or a custom endpoint</li>
            <li>Local rule-based / fuzzy-matching engine — medicine, dosage & frequency extraction</li>
            <li>Local PII detector + redactor — privacy protection before any cloud call</li>
            <li>Pandas + Plotly — data & analytics</li>
            <li>ReportLab — PDF report generation</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="pr-card">
        <h4>🔀 Three Analysis Modes</h4>
        <ul>
        <li><b>Hybrid (default):</b> Tesseract OCR + local fuzzy matching extracts a draft first, then the configured cloud AI checks that draft against the actual image and corrects/completes it. If Tesseract finds nothing at all, Hybrid escalates straight to full AI extraction instead. If no cloud API is configured, you still get the raw Tesseract result — nothing hard-fails.</li>
        <li><b>Tesseract (Offline, research):</b> OCR + fuzzy matching only, runs entirely on-device, no cost, no internet, no data leaves the machine.</li>
        <li><b>Cloud API (research):</b> sends the image straight to whichever provider is configured for direct from-scratch extraction, for comparing against the Hybrid/OCR results.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pr-card">
        <h4>🛡️ Privacy Protection</h4>
        Whenever Cloud API or Hybrid mode might send an image to an external provider, a local,
        offline PII detector scans it first (using the same on-device Tesseract engine — no network
        call) and redacts only the fields you've selected — patient name, ID, phone, address, DOB/age,
        email, national ID — while leaving the medicines, dosages, and clinical instructions fully
        visible. Only the sanitized copy is ever transmitted; the original image never leaves the
        machine. This is a heuristic, best-effort detector, not a certified compliance tool — always
        check the sanitized preview before relying on it with real patient data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pr-card">
        <h4>⚠️ Disclaimer</h4>
        This application is built for academic / portfolio demonstration purposes. Extracted
        medicine information — whether from OCR or cloud AI — must always be verified by a
        licensed pharmacist or physician before any clinical or dispensing decision is made.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
PAGE_RENDERERS = {
    "Dashboard": render_dashboard,
    "Analytics": render_analytics,
    "Settings": render_settings,
    "About": render_about,
}

PAGE_RENDERERS[st.session_state.current_page]()
