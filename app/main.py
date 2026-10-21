"""Main Streamlit entry point for the AI Fake News Detector application."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

# ── Project root on sys.path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────
_CSS_PATH = PROJECT_ROOT / "assets" / "style.css"
if _CSS_PATH.exists():
    with _CSS_PATH.open("r", encoding="utf-8") as _f:
        _css = _f.read()
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)


# ── Pipeline loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load (or auto-train and load) the FakeNewsPipeline.

    Returns the pipeline instance. Always returns something; falls back to
    demo mode if training fails.
    """
    from src.pipeline import FakeNewsPipeline  # noqa: PLC0415

    pipeline = FakeNewsPipeline()
    models_dir = PROJECT_ROOT / "models"

    # First-run: no .joblib files present → trigger training
    if not any(models_dir.glob("**/*.joblib")):
        st.info(
            "🤖 First run detected — training AI model (this takes ~30 seconds)…"
        )
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            st.warning(
                "Model training had issues. Running in demo mode.\n\n"
                f"Details: {result.stderr[-500:] if result.stderr else 'none'}"
            )

    loaded = pipeline.load()
    if not loaded:
        st.warning(
            "⚠️ Trained models could not be loaded. Running in rule-based demo mode."
        )

    return pipeline


# ── Session state defaults ─────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "fetched_title" not in st.session_state:
    st.session_state["fetched_title"] = ""
if "fetched_text" not in st.session_state:
    st.session_state["fetched_text"] = ""

# ── Load pipeline & store in session state ─────────────────────────────────
with st.spinner("Loading AI model…"):
    pipeline = load_pipeline()
st.session_state["pipeline"] = pipeline

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 AI Fake News Detector")
    st.markdown("---")

    # Sync selectbox with session state when the CTA button on the Home page
    # sets st.session_state['page'] = 'Analyze'.
    _pages = ["Home", "Analyze", "Batch Analysis", "About"]
    _current_idx = _pages.index(st.session_state["page"]) if st.session_state["page"] in _pages else 0

    selected_page = st.radio(
        "Navigate",
        options=_pages,
        index=_current_idx,
        key="nav_radio",
        label_visibility="collapsed",
    )
    # Keep session state in sync so the CTA button works
    st.session_state["page"] = selected_page

    st.markdown("---")

    # Quick model status indicator
    if pipeline.is_loaded:
        st.success("✅ AI model loaded")
    else:
        st.warning("⚠️ Demo mode (no trained model)")

    st.markdown("")
    st.caption("AI Fake News Detector v1.0")

# ── Page routing ───────────────────────────────────────────────────────────
from app.pages import home, analyze, batch, about  # noqa: E402

if selected_page == "Home":
    home.show()
elif selected_page == "Analyze":
    analyze.show()
elif selected_page == "Batch Analysis":
    batch.show()
elif selected_page == "About":
    about.show()

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; opacity:0.5; font-size:0.8rem;'>"
    "AI Fake News Detector v1.0 | For educational use only"
    "</p>",
    unsafe_allow_html=True,
)
