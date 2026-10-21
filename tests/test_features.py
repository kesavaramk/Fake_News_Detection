"""Unit tests for FeatureExtractor (src.features.feature_extractor)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import FeatureExtractor  # noqa: E402


_SAMPLE_TEXTS = [
    "according researchers published journal found significant results",
    "shocking breaking news deep state exposed wake up share deleted",
    "the study showed increases percent data university confirmed results",
    "you wont believe mainstream media lies globalist agenda revealed truth",
    "officials said city council approved new infrastructure plan million",
    "censored suppressed bombshell big pharma new world order wake up",
    "researchers found exercise associated lower rates depression anxiety",
    "shocking exposed urgent breaking banned revealed conspiracy truth wake",
]


class TestFeatureExtractor(unittest.TestCase):
    """Test suite for FeatureExtractor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = FeatureExtractor({})
        cls.X = cls.extractor.fit_transform(_SAMPLE_TEXTS)

    # ------------------------------------------------------------------
    # fit_transform shape
    # ------------------------------------------------------------------

    def test_fit_transform_shape(self) -> None:
        # Should have one row per sample
        self.assertEqual(self.X.shape[0], len(_SAMPLE_TEXTS))
        # Should have more features than samples
        self.assertGreater(self.X.shape[1], len(_SAMPLE_TEXTS))

    # ------------------------------------------------------------------
    # linguistic features extraction
    # ------------------------------------------------------------------

    def test_linguistic_features_extraction(self) -> None:
        # The private method should produce (n_samples, 8)
        ling = self.extractor._extract_linguistic_features(_SAMPLE_TEXTS)
        self.assertIsInstance(ling, np.ndarray)
        self.assertEqual(ling.shape, (len(_SAMPLE_TEXTS), 8))
        # All values should be finite and non-negative
        self.assertTrue(np.all(np.isfinite(ling)))
        self.assertTrue(np.all(ling >= 0))

    # ------------------------------------------------------------------
    # save / load round-trip
    # ------------------------------------------------------------------

    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vectorizers"
            self.extractor.save(save_path)

            loaded = FeatureExtractor({})
            loaded.load(save_path)

            X_reloaded = loaded.transform(_SAMPLE_TEXTS)

        # Shape must match
        self.assertEqual(X_reloaded.shape, self.X.shape)

        # Values must match (sparse comparison via dense)
        diff = (X_reloaded - self.X).toarray()
        np.testing.assert_allclose(diff, 0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
