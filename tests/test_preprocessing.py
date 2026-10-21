"""Unit tests for TextProcessor (src.preprocessing.text_processor)."""

import sys
import unittest
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing import TextProcessor  # noqa: E402


class TestTextProcessor(unittest.TestCase):
    """Test suite for TextProcessor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.processor = TextProcessor()

    # ------------------------------------------------------------------
    # clean_text
    # ------------------------------------------------------------------

    def test_clean_text_removes_urls(self) -> None:
        text = "Check out https://example.com and www.test.org for details."
        cleaned = self.processor.clean_text(text)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("www.", cleaned)
        self.assertNotIn("example.com", cleaned)

    def test_clean_text_removes_html(self) -> None:
        text = "<p>Hello <b>World</b></p>"
        cleaned = self.processor.clean_text(text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("</b>", cleaned)
        self.assertIn("hello", cleaned)
        self.assertIn("world", cleaned)

    # ------------------------------------------------------------------
    # remove_stopwords
    # ------------------------------------------------------------------

    def test_remove_stopwords(self) -> None:
        text = "this is a simple test of the stopword removal function"
        result = self.processor.remove_stopwords(text)
        # Common stop words should be gone
        tokens = result.split()
        stop_words = {"this", "is", "a", "of", "the"}
        for sw in stop_words:
            self.assertNotIn(sw, tokens)
        # Meaningful words should remain
        self.assertIn("simple", result)
        self.assertIn("test", result)

    # ------------------------------------------------------------------
    # lemmatize
    # ------------------------------------------------------------------

    def test_lemmatize(self) -> None:
        text = "running dogs are faster than walking cats"
        result = self.processor.lemmatize(text)
        # WordNetLemmatizer should reduce 'dogs' → 'dog', 'cats' → 'cat'
        self.assertIn("dog", result)
        self.assertIn("cat", result)

    # ------------------------------------------------------------------
    # process (pipeline)
    # ------------------------------------------------------------------

    def test_process_pipeline(self) -> None:
        text = "The researchers found <b>significant</b> results at https://journal.org"
        result = self.processor.process(text)
        # Should be lower-case, no HTML, no URL
        self.assertEqual(result, result.lower())
        self.assertNotIn("<b>", result)
        self.assertNotIn("https://", result)
        # Should be non-empty
        self.assertTrue(len(result) > 0)

    # ------------------------------------------------------------------
    # get_linguistic_stats
    # ------------------------------------------------------------------

    def test_get_linguistic_stats_keys(self) -> None:
        stats = self.processor.get_linguistic_stats("Hello world! How are you?")
        expected_keys = {
            "word_count",
            "sentence_count",
            "avg_word_length",
            "avg_sentence_length",
            "lexical_diversity",
            "caps_ratio",
            "exclamation_count",
            "question_count",
            "punctuation_ratio",
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_linguistic_stats_counts(self) -> None:
        # ALL-CAPS heavy text with exclamation marks
        text = "SHOCKING BREAKING NEWS!! The deep state is EXPOSED!!"
        stats = self.processor.get_linguistic_stats(text)

        # caps_ratio should be high (many uppercase chars)
        self.assertGreater(stats["caps_ratio"], 0.3)

        # exclamation_count should reflect the four '!' characters
        self.assertEqual(stats["exclamation_count"], 4)


if __name__ == "__main__":
    unittest.main()
