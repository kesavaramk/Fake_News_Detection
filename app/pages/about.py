"""About page for the AI Fake News Detector application."""

import json
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_metadata() -> dict:
    metadata_path = _PROJECT_ROOT / "models" / "model_metadata.json"
    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def show() -> None:
    """Render the about page."""

    st.markdown("# ℹ️ About This Tool")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.warning(
        "⚠️ **Disclaimer:** This tool is for **educational purposes only**. "
        "AI is not perfect — always verify news from multiple trusted sources. "
        "Do not rely solely on this tool for fact-checking important claims."
    )

    st.divider()

    # ── How It Works ─────────────────────────────────────────────────────────
    st.markdown("## How It Works")
    st.markdown(
        """
        The detector processes raw article text through a multi-stage pipeline:

        1. **Text Preprocessing** — lowercasing, punctuation normalisation,
           stopword removal, and lemmatisation via NLTK.
        2. **Feature Extraction** — two TF-IDF vectorisers run in parallel:
           a *word-level* one capturing vocabulary patterns, and a
           *character n-gram* one capturing casing and punctuation artefacts
           (e.g. "SHOCKING!!!" vs "according to researchers").
        3. **Ensemble Classification** — three independent classifiers vote:
           - Logistic Regression (linear decision boundary, interpretable)
           - Random Forest (non-linear, robust to noise)
           - Support Vector Machine (effective in high-dimensional TF-IDF space)
        4. **Confidence Scoring** — the mean predicted probability across
           classifiers gives a single confidence value in [0, 100].
        5. **Explainability** — top TF-IDF feature weights are surfaced as
           highlighted words and coloured tags.
        """
    )

    # ── Architecture Diagram ──────────────────────────────────────────────────
    st.markdown("## Architecture")
    st.code(
        """
┌────────────────────────────────────────────────────────┐
│                   Raw Article Text                      │
└───────────────────────────┬────────────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │  TextProcessor   │  (NLTK: tokenise,
                   │  (Preprocessing) │   lemmatise, clean)
                   └────────┬─────────┘
                            │
              ┌─────────────▼──────────────┐
              │       FeatureExtractor      │
              │  ┌──────────┐ ┌──────────┐ │
              │  │Word TF-IDF│ │Char TF-IDF│ │
              │  └──────────┘ └──────────┘ │
              │      (scipy sparse matrix)  │
              └─────────────┬──────────────┘
                            │
        ┌───────────────────▼────────────────────────┐
        │              EnsembleModel                  │
        │  ┌─────────┐  ┌───────────┐  ┌──────────┐  │
        │  │  Logistic│  │  Random   │  │   SVM    │  │
        │  │Regression│  │  Forest   │  │  (SVC)   │  │
        │  └────┬─────┘  └─────┬─────┘  └─────┬────┘  │
        │       └──────────────┴───────────────┘       │
        │                  majority vote                │
        └───────────────────────┬───────────────────────┘
                                │
                   ┌────────────▼────────────┐
                   │        Explainer         │
                   │  top words · HTML markup │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │     Prediction Result    │
                   │  label · confidence ·    │
                   │  highlighted_html ·      │
                   │  linguistic_stats        │
                   └─────────────────────────┘
        """,
        language="text",
    )

    st.divider()

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown("## Model Performance")
    metadata = _load_metadata()
    if metadata:
        p1, p2, p3, p4 = st.columns(4)
        accuracy = metadata.get("accuracy")
        f1 = metadata.get("f1_score")
        samples = metadata.get("training_samples")
        timestamp = metadata.get("timestamp", "")
        date_str = timestamp[:10] if timestamp else "—"

        p1.metric("Accuracy", f"{accuracy * 100:.1f}%" if accuracy is not None else "—")
        p2.metric("F1 Score (weighted)", f"{f1 * 100:.1f}%" if f1 is not None else "—")
        p3.metric("Training Samples", f"{samples:,}" if samples is not None else "—")
        p4.metric("Trained On", date_str)
    else:
        st.info("Model metadata not found. Train the model first to see performance stats.")

    st.divider()

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown("## Tech Stack")
    st.table(
        {
            "Library": [
                "Streamlit",
                "scikit-learn",
                "NLTK",
                "pandas",
                "NumPy",
                "Plotly",
                "requests + BeautifulSoup4",
                "joblib",
                "PyYAML",
            ],
            "Role": [
                "Web application framework",
                "ML models & TF-IDF vectorisers",
                "Text preprocessing & tokenisation",
                "Data manipulation & CSV handling",
                "Numerical operations & feature matrices",
                "Interactive charts (batch pie chart)",
                "URL fetching & HTML scraping",
                "Model serialisation / deserialisation",
                "Configuration file loading",
            ],
            "Version (min)": [
                "≥ 1.28",
                "≥ 1.3",
                "≥ 3.8",
                "≥ 2.0",
                "≥ 1.24",
                "≥ 5.15",
                "≥ 2.31 / ≥ 4.12",
                "≥ 1.3",
                "≥ 6.0",
            ],
        }
    )

    st.divider()

    # ── Limitations ───────────────────────────────────────────────────────────
    st.markdown("## Limitations")
    st.markdown(
        """
        - **Trained on synthetic data** — the model learned from algorithmically
          generated fake/real examples, which may not fully represent real-world
          misinformation in the wild.
        - **Language** — only English text is supported. Other languages will
          produce unreliable results.
        - **Context-blind** — the model analyses linguistic surface patterns, not
          factual accuracy. Stylistically neutral misinformation may be missed.
        - **Satire** — satirical or parody articles may be flagged as fake even
          when the intent is humorous rather than deceptive.
        - **Short texts** — very short snippets (< 20 words) give the model little
          signal and should be treated as uncertain regardless of the output.
        - **Not a replacement for journalists** — professional fact-checking
          organisations, primary sources, and expert review remain essential.
        """
    )
