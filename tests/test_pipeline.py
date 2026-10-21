"""Unit tests for FakeNewsPipeline (src.pipeline)."""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import FakeNewsPipeline  # noqa: E402


_FAKE_ARTICLE = (
    "YOU WON'T BELIEVE: SHOCKING BREAKING NEWS!! "
    "The deep state has EXPOSED the mainstream media lies. "
    "WAKE UP!! This is URGENT — SHARE BEFORE DELETED!!! "
    "The globalist agenda is REAL and they don't want you to know!!"
)

_REAL_ARTICLE = (
    "According to researchers at the National Institutes of Health, "
    "a study published in the New England Journal of Medicine found that "
    "regular physical activity is associated with a 15% reduction in cardiovascular risk. "
    "Officials said the findings, based on data from 45,000 adults, "
    "are consistent with prior research. The report cited independent replications."
)


class TestFakeNewsPipeline(unittest.TestCase):
    """Test suite for FakeNewsPipeline (demo mode, no trained models required)."""

    @classmethod
    def setUpClass(cls) -> None:
        # Instantiate without loading trained models — runs in demo mode
        cls.pipeline = FakeNewsPipeline()
        # Attempt to load; if models aren't saved yet it will fall back to demo
        cls.pipeline.load()

    # ------------------------------------------------------------------
    # Demo-mode label checks
    # ------------------------------------------------------------------

    def test_demo_mode_fake_article(self) -> None:
        """Highly sensationalist text should be classified as FAKE."""
        result = self.pipeline.predict(_FAKE_ARTICLE)
        self.assertEqual(result["label"], "FAKE")

    def test_demo_mode_real_article(self) -> None:
        """Neutral, well-attributed text should be classified as REAL."""
        result = self.pipeline.predict(_REAL_ARTICLE)
        self.assertEqual(result["label"], "REAL")

    # ------------------------------------------------------------------
    # Return structure
    # ------------------------------------------------------------------

    def test_predict_returns_required_keys(self) -> None:
        result = self.pipeline.predict(_FAKE_ARTICLE)
        required_keys = {
            "label",
            "confidence",
            "is_uncertain",
            "explanation",
            "highlighted_html",
            "top_words",
            "linguistic_stats",
            "mode",
        }
        self.assertTrue(
            required_keys.issubset(set(result.keys())),
            msg=f"Missing keys: {required_keys - set(result.keys())}",
        )
        # confidence should be in [0, 100]
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)
        # label must be one of the two expected values
        self.assertIn(result["label"], {"FAKE", "REAL"})

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def test_batch_predict_length(self) -> None:
        texts = [_FAKE_ARTICLE, _REAL_ARTICLE, _FAKE_ARTICLE]
        results = self.pipeline.predict_batch(texts)
        self.assertEqual(len(results), len(texts))
        for res in results:
            self.assertIn("label", res)
            self.assertIn("confidence", res)


if __name__ == "__main__":
    unittest.main()
