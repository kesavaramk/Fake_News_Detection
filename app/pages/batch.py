"""Batch analysis page for the AI Fake News Detector application."""

import io

import pandas as pd
import streamlit as st


def show() -> None:
    """Render the batch analysis page."""

    st.markdown("# 📂 Batch Article Analysis")
    st.markdown(
        "Upload a CSV file containing multiple articles to analyse them all at once."
    )

    pipeline = st.session_state.get("pipeline")

    # ── Expected Format ───────────────────────────────────────────────────────
    with st.expander("📋 Expected CSV Format", expanded=False):
        st.markdown(
            "Your CSV must contain at least a **`text`** column. "
            "An optional **`title`** column is also supported."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "title": [
                        "Scientists find new species",
                        "SHOCKING: You won't believe this!!",
                    ],
                    "text": [
                        "Researchers at Harvard University identified a new bird species in the Amazon rainforest.",
                        "WAKE UP! The deep state is HIDING the TRUTH about vaccines!!!",
                    ],
                }
            ),
            use_container_width=True,
        )

    # ── File Upload ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="CSV must contain a 'text' column. Optionally a 'title' column.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file above to get started.")
        return

    # ── Parse & Validate ──────────────────────────────────────────────────────
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    if df.empty:
        st.error("The uploaded CSV is empty.")
        return

    # Normalise column names to lower-case, strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    if "text" not in df.columns:
        st.error(
            "The CSV must contain a **`text`** column. "
            f"Columns found: `{', '.join(df.columns)}`"
        )
        return

    # Drop rows where text is null/empty
    before = len(df)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        st.warning(f"{dropped} row(s) with empty text were skipped.")

    if df.empty:
        st.error("No valid rows remaining after filtering empty text.")
        return

    # ── Preview ───────────────────────────────────────────────────────────────
    st.markdown(f"**Preview** (first 5 of {len(df):,} rows):")
    st.dataframe(df.head(5), use_container_width=True)

    # ── Run Batch Analysis ────────────────────────────────────────────────────
    if not st.button("▶ Run Batch Analysis", type="primary", use_container_width=True):
        return

    if pipeline is None:
        st.error(
            "Pipeline is not loaded. Please reload the page and wait for "
            "the model to initialise."
        )
        return

    results_rows = []
    progress_bar = st.progress(0.0, text="Starting analysis…")
    total = len(df)

    for idx, row in enumerate(df.itertuples(index=False)):
        text = str(getattr(row, "text", ""))
        title = str(getattr(row, "title", "")) if "title" in df.columns else ""
        combined = f"{title}. {text}" if title and title != "nan" else text

        try:
            result = pipeline.predict(combined)
        except Exception as exc:  # noqa: BLE001
            result = {
                "label": "ERROR",
                "confidence": 0.0,
                "is_uncertain": False,
                "mode": "error",
                "explanation": str(exc),
            }

        label = result.get("label", "ERROR")
        confidence = result.get("confidence", 0.0)
        is_uncertain = result.get("is_uncertain", False)
        verdict = "UNCERTAIN" if is_uncertain else label

        results_rows.append(
            {
                "title": title if "title" in df.columns else "",
                "text_preview": text[:120] + ("…" if len(text) > 120 else ""),
                "verdict": verdict,
                "label": label,
                "confidence_%": round(confidence, 1),
                "mode": result.get("mode", ""),
            }
        )

        progress_bar.progress(
            (idx + 1) / total,
            text=f"Analysing article {idx + 1} of {total}…",
        )

    progress_bar.empty()

    results_df = pd.DataFrame(results_rows)

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.markdown("## 📊 Summary")

    total_count = len(results_df)
    fake_count = int((results_df["verdict"] == "FAKE").sum())
    real_count = int((results_df["verdict"] == "REAL").sum())
    uncertain_count = int((results_df["verdict"] == "UNCERTAIN").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Articles", f"{total_count:,}")
    m2.metric("🔴 Fake", f"{fake_count:,}")
    m3.metric("🟢 Real", f"{real_count:,}")
    m4.metric("🟠 Uncertain", f"{uncertain_count:,}")

    # ── Pie Chart ─────────────────────────────────────────────────────────────
    st.markdown("### Distribution")
    try:
        import plotly.graph_objects as go

        labels = []
        values = []
        colors = []
        if fake_count:
            labels.append("Fake")
            values.append(fake_count)
            colors.append("#ff4b4b")
        if real_count:
            labels.append("Real")
            values.append(real_count)
            colors.append("#00c853")
        if uncertain_count:
            labels.append("Uncertain")
            values.append(uncertain_count)
            colors.append("#ffa726")

        if values:
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker_colors=colors,
                        hole=0.35,
                        textinfo="label+percent",
                        hoverinfo="label+value+percent",
                    )
                ]
            )
            fig.update_layout(
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly to see the distribution chart.")

    # ── Color-coded Results Table ─────────────────────────────────────────────
    st.markdown("### Results")

    def _color_verdict(val: str) -> str:
        if val == "FAKE":
            return "background-color: rgba(255,75,75,0.25); color: #ff4b4b; font-weight:bold"
        if val == "REAL":
            return "background-color: rgba(0,200,83,0.2); color: #00c853; font-weight:bold"
        if val == "UNCERTAIN":
            return "background-color: rgba(255,167,38,0.2); color: #ffa726; font-weight:bold"
        return ""

    display_cols = [c for c in ["title", "text_preview", "verdict", "confidence_%", "mode"] if c in results_df.columns]
    styled = (
        results_df[display_cols]
        .style.applymap(_color_verdict, subset=["verdict"])
        .format({"confidence_%": "{:.1f}%"})
    )
    st.dataframe(styled, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇ Download Results CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="fake_news_batch_results.csv",
        mime="text/csv",
        type="secondary",
    )
