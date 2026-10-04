"""Reusable presentation components for Customer Behavior Lab."""

from __future__ import annotations

from html import escape

import streamlit as st


def inject_styles() -> None:
    """Load the small set of app-level layout and motion styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap');
        :root {
          --cbl-ink: #17243a;
          --cbl-muted: #66758b;
          --cbl-line: #e3e8f0;
          --cbl-blue: #4355d8;
          --cbl-blue-soft: #eef1ff;
          --cbl-green: #16845b;
          --cbl-amber: #a96308;
          --cbl-surface: #ffffff;
        }
        html, body, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], button, input, textarea {
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        [data-testid="stIconMaterial"], .material-symbols-rounded {
          font-family: "Material Symbols Rounded" !important;
        }
        .stApp { background: #f6f8fb; color: var(--cbl-ink); }
        [data-testid="stHeader"] { background: rgba(246, 248, 251, .9); }
        [data-testid="stSidebar"] {
          background: #fff;
          border-right: 1px solid var(--cbl-line);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .65rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
          border-radius: 9px; padding: .48rem .62rem; transition: background .18s ease, color .18s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #f3f5fa; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
          color: var(--cbl-blue); background: var(--cbl-blue-soft); font-weight: 650;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--cbl-line) !important; border-radius: 12px !important; background: #fff;
        }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
        .cbl-brand { display: flex; align-items: center; gap: .7rem; padding: .25rem .15rem 1rem; }
        .cbl-brand-mark {
          display: grid; place-items: center; width: 36px; height: 36px; border-radius: 11px;
          color: #fff; background: #263b86; font-weight: 700; font-size: 1rem;
          box-shadow: 0 5px 12px rgba(38,59,134,.16);
        }
        .cbl-brand-name { color: var(--cbl-ink); font-size: .98rem; font-weight: 700; line-height: 1.15; }
        .cbl-brand-caption { color: var(--cbl-muted); font-size: .68rem; margin-top: .22rem; }
        .cbl-page-head { margin: .2rem 0 1.2rem; animation: cbl-enter .32s ease-out both; }
        .cbl-eyebrow { color: var(--cbl-blue); text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; font-weight: 700; margin-bottom: .45rem; }
        .cbl-page-title { color: var(--cbl-ink); font-size: clamp(1.65rem, 2.8vw, 2.15rem); font-weight: 650; letter-spacing: -.045em; line-height: 1.18; margin: 0; }
        .cbl-page-description { max-width: 900px; color: var(--cbl-muted); font-size: .91rem; margin: .45rem 0 0; line-height: 1.55; }
        .cbl-meta-line { display: flex; flex-wrap: wrap; gap: .48rem 1.2rem; color: var(--cbl-muted); font-size: .76rem; margin-top: .9rem; }
        .cbl-section-head { margin: .6rem 0 .8rem; }
        .cbl-section-title { color: var(--cbl-ink); font-size: 1.04rem; font-weight: 650; margin: 0; letter-spacing: -.018em; }
        .cbl-section-copy { color: var(--cbl-muted); font-size: .8rem; margin: .28rem 0 0; line-height: 1.5; }
        .cbl-metric-card, .cbl-segment-card, .cbl-highlight-card, .cbl-pipeline-card {
          background: var(--cbl-surface); border: 1px solid var(--cbl-line); border-radius: 12px;
          padding: 1rem 1.05rem; height: 100%; transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
          animation: cbl-enter .3s ease-out both;
        }
        .cbl-metric-card:hover, .cbl-segment-card:hover, .cbl-highlight-card:hover {
          transform: translateY(-2px); border-color: #cbd3e4; box-shadow: 0 8px 22px rgba(31,47,79,.06);
        }
        .cbl-metric-top { display: flex; justify-content: space-between; align-items: center; gap: .5rem; }
        .cbl-metric-label { color: var(--cbl-muted); font-size: .77rem; font-weight: 550; }
        .cbl-metric-icon { color: var(--cbl-blue); font-size: .9rem; }
        .cbl-metric-value { color: var(--cbl-ink); font-size: clamp(1.45rem, 2.4vw, 1.9rem); font-weight: 650; letter-spacing: -.045em; line-height: 1.2; margin-top: .55rem; }
        .cbl-metric-detail { color: var(--cbl-muted); font-size: .72rem; margin-top: .35rem; line-height: 1.4; }
        .cbl-chart-card { background: #fff; border: 1px solid var(--cbl-line); border-radius: 12px; padding: .9rem 1rem .25rem; height: 100%; }
        .cbl-chart-title { color: var(--cbl-ink); font-size: .93rem; font-weight: 650; margin: 0; }
        .cbl-chart-copy { color: var(--cbl-muted); font-size: .74rem; margin: .22rem 0 .25rem; }
        .cbl-segment-kicker { color: var(--cbl-blue); font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
        .cbl-segment-title { color: var(--cbl-ink); font-size: .94rem; font-weight: 650; margin: .3rem 0 .6rem; }
        .cbl-segment-summary { color: var(--cbl-muted); font-size: .76rem; line-height: 1.5; margin-top: .7rem; }
        .cbl-pipeline-number { color: #9aa6b7; font-size: .7rem; font-weight: 700; letter-spacing: .09em; }
        .cbl-pipeline-title { color: var(--cbl-ink); font-size: .82rem; font-weight: 650; margin: .32rem 0 .75rem; min-height: 2.1rem; }
        .cbl-pipeline-value { color: var(--cbl-ink); font-size: 1.15rem; font-weight: 650; letter-spacing: -.025em; }
        .cbl-pipeline-label { color: var(--cbl-muted); font-size: .68rem; margin-top: .12rem; }
        .cbl-status { display: inline-flex; align-items: center; gap: .35rem; border: 1px solid #d9e9e1; background: #f0f8f4; color: var(--cbl-green); border-radius: 999px; padding: .25rem .55rem; font-size: .69rem; font-weight: 600; }
        .cbl-callout { border-left: 3px solid var(--cbl-blue); background: #f0f3ff; color: #344267; padding: .8rem 1rem; border-radius: 0 9px 9px 0; font-size: .82rem; line-height: 1.5; }
        @keyframes cbl-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 760px) {
          .block-container { padding: 1.15rem 1rem 2rem; }
          .cbl-page-title { font-size: 1.55rem; }
          .cbl-metric-card, .cbl-segment-card, .cbl-pipeline-card { padding: .85rem; }
        }
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.markdown(
        '<div class="cbl-brand"><div class="cbl-brand-mark">C</div><div><div class="cbl-brand-name">Customer Behavior Lab</div><div class="cbl-brand-caption">E Commerce Customer Behavior Analytics</div></div></div>',
        unsafe_allow_html=True,
    )


def render_page_header(category: str, title: str, description: str, metadata: list[str] | None = None) -> None:
    meta = "".join(f"<span>{escape(item)}</span>" for item in (metadata or []))
    meta_html = f'<div class="cbl-meta-line">{meta}</div>' if meta else ""
    st.markdown(
        f'<header class="cbl-page-head"><div class="cbl-eyebrow">{escape(category)}</div><h1 class="cbl-page-title">{escape(title)}</h1><p class="cbl-page-description">{escape(description)}</p>{meta_html}</header>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = "") -> None:
    copy = f'<p class="cbl-section-copy">{escape(description)}</p>' if description else ""
    st.markdown(
        f'<div class="cbl-section-head"><h2 class="cbl-section-title">{escape(title)}</h2>{copy}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, detail: str = "", icon: str = "") -> None:
    icon_html = f'<span class="cbl-metric-icon">{escape(icon)}</span>' if icon else ""
    detail_html = f'<div class="cbl-metric-detail">{escape(detail)}</div>' if detail else ""
    st.markdown(
        f'<div class="cbl-metric-card"><div class="cbl-metric-top"><span class="cbl-metric-label">{escape(label)}</span>{icon_html}</div><div class="cbl-metric-value">{escape(value)}</div>{detail_html}</div>',
        unsafe_allow_html=True,
    )


def render_chart_card(title: str, description: str, fig, *, height: int | None = None) -> None:
    if height is not None:
        fig.update_layout(height=height)
    with st.container(border=True):
        st.markdown(
            f'<h3 class="cbl-chart-title">{escape(title)}</h3><p class="cbl-chart-copy">{escape(description)}</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )


def render_pipeline_card(index: int, title: str, events: str, customers: str, retention: str) -> None:
    st.markdown(
        f'<div class="cbl-pipeline-card"><div class="cbl-pipeline-number">BƯỚC {index:02d}</div><div class="cbl-pipeline-title">{escape(title)}</div><div class="cbl-pipeline-value">{escape(events)}</div><div class="cbl-pipeline-label">sự kiện còn lại</div><div class="cbl-pipeline-value" style="margin-top:.55rem">{escape(customers)}</div><div class="cbl-pipeline-label">khách hàng còn lại · giữ lại {escape(retention)}</div></div>',
        unsafe_allow_html=True,
    )


def render_segment_card(cluster: int, name: str, customers: str, share: str, purchase_rate: str, spend: str, recency: str, action: str) -> None:
    st.markdown(
        f'<div class="cbl-segment-card"><div class="cbl-segment-kicker">Cụm C1 · {cluster}</div><div class="cbl-segment-title">{escape(name)}</div><div class="cbl-metric-top"><span class="cbl-metric-label">Khách hàng</span><span class="cbl-metric-label">Tỷ trọng</span></div><div class="cbl-metric-top"><strong>{escape(customers)}</strong><strong>{escape(share)}</strong></div><div class="cbl-metric-top" style="margin-top:.6rem"><span class="cbl-metric-label">Từng mua</span><span class="cbl-metric-label">Chi tiêu TB</span></div><div class="cbl-metric-top"><strong>{escape(purchase_rate)}</strong><strong>{escape(spend)}</strong></div><div class="cbl-segment-summary">Recency trung bình: {escape(recency)} ngày<br>{escape(action)}</div></div>',
        unsafe_allow_html=True,
    )


def render_highlight(label: str, value: str, metric: str, detail: str) -> None:
    st.markdown(
        f'<div class="cbl-highlight-card"><div class="cbl-segment-kicker">{escape(label)}</div><div class="cbl-segment-title" style="margin:.5rem 0 .25rem">{escape(value)}</div><div class="cbl-metric-value" style="font-size:1.35rem;margin-top:.15rem">{escape(metric)}</div><div class="cbl-metric-detail">{escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )
