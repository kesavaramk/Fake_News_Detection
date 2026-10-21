"""Explainability utilities: key-word attribution and HTML highlighting."""

import re
from typing import Any, Dict, Optional

import numpy as np


class Explainer:
    """Provide human-readable explanations for model predictions.

    Uses the Logistic Regression coefficient vector combined with the TF-IDF
    vocabulary to surface words that most strongly pushed the prediction
    toward *fake* or *real*.

    Parameters
    ----------
    feature_extractor:
        A fitted :class:`~src.features.FeatureExtractor` instance.
    classical_models:
        A trained :class:`~src.models.ClassicalModels` instance.
    config:
        Project configuration dictionary (not currently used, reserved for
        future tuning parameters).
    """

    def __init__(
        self,
        feature_extractor,
        classical_models,
        config: Dict[str, Any],
    ) -> None:
        """Store references to the fitted extractor and models.

        Parameters
        ----------
        feature_extractor:
            Fitted :class:`~src.features.FeatureExtractor`.
        classical_models:
            Trained :class:`~src.models.ClassicalModels`.
        config:
            Project configuration dictionary.
        """
        self._extractor = feature_extractor
        self._models = classical_models
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_top_words(self, text: str, n: int = 15) -> Dict[str, float]:
        """Identify the top *n* words that influence the prediction.

        Uses the LR coefficient vector to score each word that appears in
        both the text and the TF-IDF vocabulary.  A **positive** score
        indicates a *fake-leaning* signal; a **negative** score indicates
        a *real-leaning* signal.

        Parameters
        ----------
        text:
            Raw or pre-processed text to explain.
        n:
            Number of top words to return (default 15).

        Returns
        -------
        dict
            ``{word: score}`` sorted by absolute score descending, limited to
            *n* entries.
        """
        try:
            vocab = self._extractor._word_tfidf.vocabulary_
            coefs = self._models.get_lr_coefficients()
        except Exception:  # noqa: BLE001
            return {}

        words_in_text = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower()))

        scores: Dict[str, float] = {}
        for word in words_in_text:
            if word in vocab:
                idx = vocab[word]
                if idx < len(coefs):
                    scores[word] = float(coefs[idx])

        sorted_words = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return {w: round(s, 4) for w, s in sorted_words[:n]}

    def highlight_text(self, text: str, top_words: Dict[str, float]) -> str:
        """Wrap influential words in HTML ``<span>`` elements.

        Words with a positive score (fake-leaning) receive the class
        ``word-fake``; words with a negative score (real-leaning) receive
        ``word-real``.

        Parameters
        ----------
        text:
            Original text to annotate.
        top_words:
            Dictionary returned by :meth:`get_top_words`.

        Returns
        -------
        str
            HTML string with relevant words wrapped in ``<span>`` tags.
        """
        if not top_words:
            return text

        def _replacer(match: re.Match) -> str:
            word = match.group(0)
            score = top_words.get(word.lower())
            if score is None:
                return word
            css_class = "word-fake" if score > 0 else "word-real"
            return f'<span class="{css_class}">{word}</span>'

        pattern = r"\b(" + "|".join(re.escape(w) for w in top_words) + r")\b"
        highlighted = re.sub(pattern, _replacer, text, flags=re.IGNORECASE)
        return highlighted

    def explain_prediction(
        self, text: str, label: str, confidence: float
    ) -> Dict[str, Any]:
        """Generate a full explanation for a single prediction.

        Parameters
        ----------
        text:
            The input news text.
        label:
            Predicted label (``'FAKE'`` or ``'REAL'``).
        confidence:
            Model confidence as a percentage in ``[0, 100]``.

        Returns
        -------
        dict
            - ``top_words`` – dict returned by :meth:`get_top_words`.
            - ``highlighted_html`` – HTML string from :meth:`highlight_text`.
            - ``summary`` – human-readable explanation sentence.
        """
        top_words = self.get_top_words(text)
        highlighted_html = self.highlight_text(text, top_words)

        fake_words = [w for w, s in top_words.items() if s > 0]
        real_words = [w for w, s in top_words.items() if s < 0]

        if label == "FAKE":
            if fake_words:
                summary = (
                    f"The model classified this article as FAKE with "
                    f"{confidence:.1f}% confidence. "
                    f"Key fake-leaning signals include: "
                    f"{', '.join(fake_words[:5])}."
                )
            else:
                summary = (
                    f"The model classified this article as FAKE with "
                    f"{confidence:.1f}% confidence based on overall linguistic patterns."
                )
        else:
            if real_words:
                summary = (
                    f"The model classified this article as REAL with "
                    f"{confidence:.1f}% confidence. "
                    f"Key credibility signals include: "
                    f"{', '.join(real_words[:5])}."
                )
            else:
                summary = (
                    f"The model classified this article as REAL with "
                    f"{confidence:.1f}% confidence based on overall linguistic patterns."
                )

        return {
            "top_words": top_words,
            "highlighted_html": highlighted_html,
            "summary": summary,
        }

    def get_linguistic_analysis(self, text: str) -> Dict[str, Any]:
        """Return human-readable linguistic statistics for *text*.

        Parameters
        ----------
        text:
            Raw input text.

        Returns
        -------
        dict
            Keys are human-readable labels mapping to formatted values:

            - ``Word Count``
            - ``Sentence Count``
            - ``Caps Ratio``
            - ``Exclamation Marks``
            - ``Question Marks``
            - ``Lexical Diversity``
            - ``Punctuation Ratio``
        """
        import string as _string
        from nltk.tokenize import sent_tokenize, word_tokenize  # noqa: PLC0415

        if not text or not text.strip():
            return {
                "Word Count": 0,
                "Sentence Count": 0,
                "Caps Ratio": "0.00%",
                "Exclamation Marks": 0,
                "Question Marks": 0,
                "Lexical Diversity": "0.00%",
                "Punctuation Ratio": "0.00%",
            }

        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
        except Exception:  # noqa: BLE001
            sentences = text.split(".")
            words = text.split()

        alpha_chars = [c for c in text if c.isalpha()]
        caps_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )

        tokens_lower = [w.lower() for w in words if w.isalpha()]
        lexical_diversity = (
            len(set(tokens_lower)) / len(tokens_lower) if tokens_lower else 0.0
        )

        punct_chars = sum(1 for c in text if c in _string.punctuation)
        punctuation_ratio = punct_chars / len(text) if text else 0.0

        return {
            "Word Count": len(words),
            "Sentence Count": len(sentences),
            "Caps Ratio": f"{caps_ratio * 100:.2f}%",
            "Exclamation Marks": text.count("!"),
            "Question Marks": text.count("?"),
            "Lexical Diversity": f"{lexical_diversity * 100:.2f}%",
            "Punctuation Ratio": f"{punctuation_ratio * 100:.2f}%",
        }
