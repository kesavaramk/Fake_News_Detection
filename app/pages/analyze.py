"""Analyze page for the AI Fake News Detector application."""

import html
from typing import Any, Dict

import streamlit as st


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_url(url: str) -> tuple[str, str]:
    """Fetch a web page and extract the title + main text.

    Returns
    -------
    (title, text) – both are empty strings on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Prefer <article> element; fall back to <body>
        article = soup.find("article") or soup.find("body")
        text = article.get_text(separator=" ", strip=True) if article else ""
        # Collapse whitespace
        text = " ".join(text.split())

        return title, text

    except Exception as exc:  # noqa: BLE001
        return "", f"ERROR: {exc}"


def _build_highlight_html(result: Dict[str, Any], raw_text: str) -> str:
    """Return highlighted HTML for the article text.

    Uses result['highlighted_html'] when available; otherwise falls back to
    plain-escaped text.
    """
    highlighted = result.get("highlighted_html", "")
    if highlighted and highlighted != raw_text:
        # Pipeline already produced annotated HTML – trust it
        return highlighted

    # Minimal fallback: escape & return plain text in a <p>
    return f"<p style='line-height:1.7'>{html.escape(raw_text)}</p>"


def _render_verdict(result: Dict[str, Any]) -> None:
    """Render the big verdict badge, confidence bar, and uncertainty warning."""
    label = result.get("label", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    is_uncertain = result.get("is_uncertain", False)

    # Choose CSS class
    if is_uncertain:
        css_class = "verdict-uncertain"
        display_label = "⚠️ UNCERTAIN"
    elif label == "FAKE":
        css_class = "verdict-fake"
        display_label = "❌ FAKE NEWS"
    else:
        css_class = "verdict-real"
        display_label = "✅ LIKELY REAL"

    st.markdown(
        f'<div class="{css_class}">{display_label}</div>',
        unsafe_allow_html=True,
    )

    # Confidence bar
    confidence_fraction = min(max(confidence / 100.0, 0.0), 1.0)
    st.markdown(f"**Confidence: {confidence:.1f}%**")
    st.progress(confidence_fraction)

    if is_uncertain:
        st.warning(
            "⚠️ Borderline case — AI is uncertain. "
            "Please verify with additional sources."
        )


def _render_explanation(result: Dict[str, Any], raw_text: str) -> None:
    """Render the expandable explanation section."""
    with st.expander("🔍 Why did the AI flag this?", expanded=False):
        top_words: Dict[str, float] = result.get("top_words", {})
        label = result.get("label", "UNKNOWN")
        tag_class = "tag-fake" if label == "FAKE" else "tag-real"

        if top_words:
            st.markdown("**Top contributing words:**")
            tags_html = " ".join(
                f'<span class="{tag_class}">{html.escape(word)}</span>'
                for word in list(top_words.keys())[:20]
            )
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("")

        st.markdown("**Article with highlighted signals:**")
        highlighted = _build_highlight_html(result, raw_text)
        st.markdown(
            f"""
            <div style="
                background:rgba(255,255,255,0.04);
                border:1px solid rgba(255,255,255,0.1);
                border-radius:8px;
                padding:1rem 1.2rem;
                max-height:320px;
                overflow-y:auto;
                line-height:1.75;
                font-size:0.92rem;
            ">
                {highlighted}
            </div>
            """,
            unsafe_allow_html=True,
        )

        explanation_text = result.get("explanation", "")
        if explanation_text:
            st.markdown(f"\n> {explanation_text}")


def _render_linguistic_analysis(result: Dict[str, Any]) -> None:
    """Render the expandable linguistic analysis section."""
    linguistic_stats: Dict[str, Any] = result.get("linguistic_stats", {})
    if not linguistic_stats:
        return

    with st.expander("📊 Linguistic Analysis", expanded=False):
        items = list(linguistic_stats.items())
        # Render in rows of 3 columns
        for row_start in range(0, len(items), 3):
            chunk = items[row_start : row_start + 3]
            cols = st.columns(3)
            for col, (stat_name, stat_value) in zip(cols, chunk):
                with col:
                    st.metric(stat_name, stat_value)


# ── Main page ─────────────────────────────────────────────────────────────────

def show() -> None:
    """Render the analyze page."""

    st.markdown("# 🔍 Analyze an Article")
    st.markdown(
        "Paste article text or enter a URL below, then click **Analyze**."
    )

    pipeline = st.session_state.get("pipeline")

    # ── Input Tabs ────────────────────────────────────────────────────────────
    tab_text, tab_url = st.tabs(["📄 Paste Text", "🌐 Fetch from URL"])

    title_input: str = ""
    text_input: str = ""

    with tab_text:
        title_input = st.text_input(
            "Article Title (optional)",
            placeholder="e.g. Scientists Discover New Species of Bird",
            key="text_tab_title",
        )
        text_input = st.text_area(
            "Article Content",
            placeholder="Paste the full article text here…",
            height=260,
            key="text_tab_content",
        )

    with tab_url:
        url_input = st.text_input(
            "Article URL",
            placeholder="https://example.com/news/article",
            key="url_input",
        )
        if st.button("📥 Fetch Article", key="fetch_btn"):
            if not url_input.strip():
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching article from URL…"):
                    fetched_title, fetched_text = _fetch_url(url_input.strip())

                if fetched_text.startswith("ERROR:"):
                    st.error(f"Could not fetch the URL: {fetched_text}")
                else:
                    st.session_state["fetched_title"] = fetched_title
                    st.session_state["fetched_text"] = fetched_text
                    st.success("Article fetched successfully!")

        if st.session_state.get("fetched_title"):
            st.markdown(f"**Title:** {st.session_state['fetched_title']}")
        if st.session_state.get("fetched_text"):
            st.text_area(
                "Fetched Text (preview)",
                value=st.session_state["fetched_text"][:2000],
                height=180,
                disabled=True,
                key="fetched_preview",
            )

    st.divider()

    # ── Analyze Button ────────────────────────────────────────────────────────
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_clicked:
        # Decide which source to use
        fetched_text = st.session_state.get("fetched_text", "")
        fetched_title = st.session_state.get("fetched_title", "")

        # Prefer fetched content if available, otherwise use pasted text
        if fetched_text:
            final_title = fetched_title
            final_text = fetched_text
        else:
            final_title = title_input.strip()
            final_text = text_input.strip()

        if not final_text:
            st.warning("Please enter some article text or fetch a URL before analyzing.")
            return

        # Combine title + text for richer features
        combined = f"{final_title}. {final_text}" if final_title else final_text

        if pipeline is None:
            st.error(
                "Pipeline is not loaded. Please reload the page and wait for "
                "the model to initialise."
            )
            return

        with st.spinner("Analyzing article…"):
            result = pipeline.predict(combined)

        st.divider()
        st.markdown("## 📋 Analysis Results")

        # Demo mode notice
        if result.get("mode") == "demo":
            st.info(
                "ℹ️ Note: AI model not trained yet — using rule-based demo mode. "
                "Restart the app to trigger training."
            )

        _render_verdict(result)
        _render_explanation(result, combined)
        _render_linguistic_analysis(result)
