"""
ui_components.py
------------------
Reusable, premium healthcare-themed UI building blocks for the Streamlit
app: global CSS injection (with a full light/dark theme system driven by
CSS custom properties), header banner, KPI cards, stage cards, confidence
badges, and styled alert boxes.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Theme tokens — everything below is expressed as CSS variables so a single
# toggle can re-skin the whole app without touching component markup.
# --------------------------------------------------------------------------- #
THEMES = {
    "light": {
        "bg": "#F4F7FA",
        "bg_grad_to": "#EAF1F3",
        "card_bg": "#FFFFFF",
        "card_border": "#E3EAEE",
        "ink": "#152331",
        "muted": "#64748B",
        "sidebar_bg_from": "#FFFFFF",
        "sidebar_bg_to": "#F1F6F7",
        "sidebar_border": "#E3EAEE",
        "shadow": "rgba(21, 35, 49, 0.07)",
        "input_bg": "#FFFFFF",
        "table_stripe": "#F7FAFB",
    },
    "dark": {
        "bg": "#0B1220",
        "bg_grad_to": "#0F1B2B",
        "card_bg": "#131C2C",
        "card_border": "#223049",
        "ink": "#EAF1F7",
        "muted": "#93A4BC",
        "sidebar_bg_from": "#0D1524",
        "sidebar_bg_to": "#0A101C",
        "sidebar_border": "#1E293D",
        "shadow": "rgba(0, 0, 0, 0.45)",
        "input_bg": "#0F1929",
        "table_stripe": "#111A29",
    },
}

# Brand accents stay constant across themes for consistent identity.
PRIMARY = "#0E9488"
PRIMARY_DARK = "#0B6F66"
ACCENT = "#3B82F6"
SUCCESS = "#22B378"
WARNING = "#E2A33D"
DANGER = "#E85D4A"


def inject_global_css(theme: str = "light") -> None:
    t = THEMES.get(theme, THEMES["light"])
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        :root {{
            --pr-bg: {t['bg']};
            --pr-bg-grad-to: {t['bg_grad_to']};
            --pr-card: {t['card_bg']};
            --pr-border: {t['card_border']};
            --pr-ink: {t['ink']};
            --pr-muted: {t['muted']};
            --pr-shadow: {t['shadow']};
            --pr-primary: {PRIMARY};
            --pr-primary-dark: {PRIMARY_DARK};
            --pr-accent: {ACCENT};
            --pr-success: {SUCCESS};
            --pr-warning: {WARNING};
            --pr-danger: {DANGER};
        }}

        .stApp {{
            background: linear-gradient(180deg, var(--pr-bg) 0%, var(--pr-bg-grad-to) 100%);
        }}

        /* Hide default Streamlit chrome for a cleaner product feel */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{background: transparent;}}

        h1, h2, h3, h4, h5, p, span, label, div {{ color: var(--pr-ink); }}
        .stMarkdown, .stCaption, small {{ color: var(--pr-muted); }}

        /* ---------- Hero header ---------- */
        .app-hero {{
            background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 55%, {ACCENT} 140%);
            border-radius: 20px;
            padding: 30px 36px;
            margin-bottom: 24px;
            box-shadow: 0 14px 34px var(--pr-shadow);
            color: white !important;
            animation: fadeIn 0.5s ease-in-out;
        }}
        .app-hero * {{ color: white !important; }}
        .app-hero h1 {{
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0 0 6px 0;
            letter-spacing: -0.02em;
        }}
        .app-hero p {{
            margin: 0;
            opacity: 0.94;
            font-size: 0.96rem;
            font-weight: 400;
            max-width: 760px;
        }}
        .app-hero .badge-row {{
            margin-top: 16px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .hero-chip {{
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 999px;
            padding: 4px 13px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            backdrop-filter: blur(6px);
        }}
        .offline-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-top: 14px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.3);
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
        }}

        /* ---------- Generic card ---------- */
        .pr-card {{
            background: var(--pr-card);
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 2px 16px var(--pr-shadow);
            border: 1px solid var(--pr-border);
            margin-bottom: 16px;
            animation: fadeIn 0.4s ease-in-out;
        }}
        .pr-card h4 {{
            margin: 0 0 10px 0;
            color: var(--pr-ink);
            font-size: 1rem;
            font-weight: 700;
        }}
        .section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 6px 0 14px 0;
        }}
        .section-header .icon {{
            font-size: 1.3rem;
        }}
        .section-header h2 {{
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--pr-ink);
            margin: 0;
        }}
        .section-sub {{
            color: var(--pr-muted);
            font-size: 0.86rem;
            margin: -8px 0 16px 32px;
        }}

        /* ---------- KPI cards ---------- */
        .kpi-card {{
            background: var(--pr-card);
            border-radius: 16px;
            padding: 18px 20px;
            border: 1px solid var(--pr-border);
            box-shadow: 0 2px 16px var(--pr-shadow);
            text-align: left;
            height: 100%;
            transition: transform 0.15s ease;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); }}
        .kpi-card .kpi-icon {{
            font-size: 1.4rem;
            margin-bottom: 6px;
        }}
        .kpi-card .kpi-value {{
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--pr-ink);
            line-height: 1.1;
        }}
        .kpi-card .kpi-label {{
            font-size: 0.8rem;
            color: var(--pr-muted);
            font-weight: 600;
            margin-top: 3px;
        }}

        /* ---------- Stage image cards ---------- */
        .stage-caption {{
            text-align: center;
            font-weight: 700;
            color: var(--pr-ink);
            font-size: 0.86rem;
            margin-top: 8px;
        }}
        .stage-sub {{
            text-align: center;
            color: var(--pr-muted);
            font-size: 0.72rem;
            margin-top: 2px;
        }}

        /* ---------- Confidence badges ---------- */
        .conf-badge {{
            display: inline-block;
            padding: 3px 11px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }}
        .conf-high {{ background: rgba(34,179,120,0.16); color: {SUCCESS}; }}
        .conf-mid  {{ background: rgba(226,163,61,0.18); color: {WARNING}; }}
        .conf-low  {{ background: rgba(232,93,74,0.16); color: {DANGER}; }}

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.3rem;
            font-weight: 700;
            font-size: 0.92rem;
            transition: all 0.15s ease-in-out;
            box-shadow: 0 4px 14px rgba(14,148,136,0.28);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(14,148,136,0.4);
        }}
        .stButton > button p, .stDownloadButton > button p {{ color: white !important; }}

        /* Secondary / theme-toggle style buttons */
        button[kind="secondary"] {{
            background: var(--pr-card) !important;
            color: var(--pr-ink) !important;
            border: 1px solid var(--pr-border) !important;
            box-shadow: none !important;
        }}
        button[kind="secondary"] p {{ color: var(--pr-ink) !important; }}

        /* ---------- Misc ---------- */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {t['sidebar_bg_from']} 0%, {t['sidebar_bg_to']} 100%);
            border-right: 1px solid {t['sidebar_border']};
        }}
        [data-testid="stSidebar"] * {{ color: var(--pr-ink); }}

        /* ---------- Alerts ---------- */
        div[data-testid="stNotification"], .stAlert {{
            border-radius: 12px !important;
            box-shadow: 0 2px 10px var(--pr-shadow);
        }}

        /* ---------- Dataframe / table polish ---------- */
        [data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--pr-border);
        }}

        /* ---------- Metric polish (native st.metric) ---------- */
        [data-testid="stMetric"] {{
            background: var(--pr-card);
            border: 1px solid var(--pr-border);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 2px 16px var(--pr-shadow);
        }}

        /* ---------- Inputs ---------- */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
            background: {t['input_bg']} !important;
            color: var(--pr-ink) !important;
            border-color: var(--pr-border) !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: {t['input_bg']};
            border-radius: 12px;
        }}

        hr {{ border-color: var(--pr-border) !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle() -> None:
    """Sidebar dark/light mode switch — flips session_state.theme and reruns."""
    current = st.session_state.get("theme", "light")
    label = "☀️ Light mode" if current == "dark" else "🌙 Dark mode"
    if st.button(label, use_container_width=True, key="theme_toggle_btn"):
        st.session_state.theme = "dark" if current == "light" else "light"
        st.rerun()


def render_hero(title: str, subtitle: str, chips: Optional[list] = None, show_offline_pill: bool = True) -> None:
    chips = chips or []
    chip_html = "".join(f'<span class="hero-chip">{c}</span>' for c in chips)
    offline_html = (
        '<div class="offline-pill">🔒 100% Offline — No Cloud AI, No Internet Required</div>'
        if show_offline_pill else ""
    )
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>🩺 {title}</h1>
            <p>{subtitle}</p>
            <div class="badge-row">{chip_html}</div>
            {offline_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-header"><span class="icon">{icon}</span><h2>{title}</h2></div>
        {f'<div class="section-sub">{subtitle}</div>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(icon: str, value: str, label: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_image_card(image: Image.Image, title: str, subtitle: str) -> None:
    st.image(image, use_container_width=True)
    st.markdown(
        f'<div class="stage-caption">{title}</div><div class="stage-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def confidence_badge_html(confidence: int) -> str:
    if confidence >= 80:
        css_class = "conf-high"
    elif confidence >= 50:
        css_class = "conf-mid"
    else:
        css_class = "conf-low"
    return f'<span class="conf-badge {css_class}">{confidence}%</span>'


def quality_badge_html(quality: str) -> str:
    mapping = {
        "Excellent": "conf-high",
        "Good": "conf-high",
        "Fair": "conf-mid",
        "Poor": "conf-low",
    }
    css_class = mapping.get(quality, "conf-mid")
    return f'<span class="conf-badge {css_class}">{quality}</span>'
