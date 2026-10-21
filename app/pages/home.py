"""Home page for the AI Fake News Detector application."""

import json
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_metadata() -> dict:
    """Load model metadata from disk, returning an empty dict if unavailable."""
    metadata_path = _PROJECT_ROOT / "models" / "model_metadata.json"
    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def show() -> None:
    """Render the home page."""

    # ── Hero Section ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 2.5rem 1rem 1.5rem 1rem;">
            <h1 style="font-size:3rem; font-weight:900; margin-bottom:0.4rem;">
                🔍 AI Fake News Detector
            </h1>
            <p style="font-size:1.25rem; opacity:0.75; margin-top:0;">
                Detect misinformation with AI-powered analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── How It Works ─────────────────────────────────────────────────────────
    st.markdown("## How It Works")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.5rem;">📝</div>
                <h3>Enter Article</h3>
                <p style="opacity:0.8;">
                    Paste the article text directly, or provide a URL and let
                    the app fetch the content for you.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.5rem;">🤖</div>
                <h3>AI Analysis</h3>
                <p style="opacity:0.8;">
                    Our NLP + ML pipeline preprocesses the text, extracts
                    TF-IDF features, and runs an ensemble of classifiers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.5rem;">📊</div>
                <h3>Get Results</h3>
                <p style="opacity:0.8;">
                    Receive a clear verdict, a confidence score, and a word-level
                    explanation of what the AI found suspicious.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Feature Highlights ────────────────────────────────────────────────────
    st.markdown("## What This Tool Offers")

    feat_left, feat_right = st.columns(2, gap="large")

    with feat_left:
        st.markdown(
            """
            - 🧠 **Ensemble ML model** — Logistic Regression, Random Forest,
              and SVM voting together for robust predictions
            - 🔤 **Word & character TF-IDF** — captures both semantic patterns
              and punctuation / casing artefacts
            - 🖊️ **Explainability** — highlighted text and top contributing words
              show exactly why the AI decided what it did
            """
        )

    with feat_right:
        st.markdown(
            """
            - 📂 **Batch analysis** — upload a CSV file and analyse hundreds of
              articles at once with a progress bar and downloadable results
            - 🌐 **URL fetching** — paste a news URL and the app scrapes the
              article text automatically
            - 📈 **Linguistic statistics** — word count, exclamation marks,
              ALL-CAPS usage, and more surface-level signals
            """
        )

    st.divider()

    # ── Stats Row ─────────────────────────────────────────────────────────────
    metadata = _load_metadata()
    if metadata:
        st.markdown("## Model Stats")
        s1, s2, s3, s4 = st.columns(4)

        accuracy = metadata.get("accuracy")
        f1 = metadata.get("f1_score")
        samples = metadata.get("training_samples")
        timestamp = metadata.get("timestamp", "")

        # Shorten ISO timestamp to just the date part
        date_str = timestamp[:10] if timestamp else "—"

        with s1:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.1f}%" if accuracy is not None else "—",
            )
        with s2:
            st.metric(
                "F1 Score",
                f"{f1 * 100:.1f}%" if f1 is not None else "—",
            )
        with s3:
            st.metric(
                "Training Samples",
                f"{samples:,}" if samples is not None else "—",
            )
        with s4:
            st.metric("Last Trained", date_str)

        st.divider()

    # ── Call to Action ────────────────────────────────────────────────────────
    st.markdown("## Ready to check an article?")
    st.markdown(
        "Click the button below to go straight to the analysis page, or "
        "choose **Analyze** from the sidebar at any time."
    )

    if st.button("🔍 Analyze an Article", type="primary", use_container_width=False):
        st.session_state["page"] = "Analyze"
        st.rerun()
