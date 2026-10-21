"""Feature extraction for the Fake News Detection pipeline."""

import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """Extract a combined sparse feature matrix from raw text.

    Three feature groups are concatenated horizontally:

    1. **Word-level TF-IDF** – unigrams and bigrams over word tokens.
    2. **Character-level TF-IDF** – character n-grams (2–5 chars by default).
    3. **Linguistic features** – eight hand-crafted numeric signals.

    Parameters
    ----------
    config:
        Project configuration dictionary.  Relevant keys (all optional,
        sensible defaults are provided):

        - ``features.word_tfidf.max_features`` (int, default 50 000)
        - ``features.word_tfidf.ngram_range`` (list [min, max], default [1, 2])
        - ``features.word_tfidf.min_df`` (int, default 2)
        - ``features.char_tfidf.max_features`` (int, default 30 000)
        - ``features.char_tfidf.ngram_range`` (list [min, max], default [2, 5])
        - ``features.char_tfidf.min_df`` (int, default 3)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise both TF-IDF vectorisers from *config*.

        Parameters
        ----------
        config:
            Project configuration dictionary.
        """
        feat_cfg = config.get("features", {})

        # ---- word TF-IDF ----
        word_cfg = feat_cfg.get("word_tfidf", {})
        word_ngram = tuple(word_cfg.get("ngram_range", [1, 2]))
        self._word_tfidf = TfidfVectorizer(
            max_features=word_cfg.get("max_features", 50_000),
            ngram_range=word_ngram,
            min_df=word_cfg.get("min_df", 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",
        )

        # ---- char TF-IDF ----
        char_cfg = feat_cfg.get("char_tfidf", {})
        char_ngram = tuple(char_cfg.get("ngram_range", [2, 5]))
        self._char_tfidf = TfidfVectorizer(
            max_features=char_cfg.get("max_features", 30_000),
            ngram_range=char_ngram,
            min_df=char_cfg.get("min_df", 3),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="char_wb",
        )

        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract eight numeric linguistic features for each text.

        Features (in column order):
        0. ``caps_ratio``       – uppercase alpha chars / all alpha chars
        1. ``exclamation_count`` – number of ``!`` characters
        2. ``question_count``   – number of ``?`` characters
        3. ``url_count``        – number of URL patterns found
        4. ``word_count``       – total word tokens
        5. ``avg_word_length``  – mean token character length
        6. ``lexical_diversity`` – unique tokens / total tokens
        7. ``punctuation_ratio`` – punctuation chars / total chars

        Parameters
        ----------
        texts:
            List of raw text strings.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 8)``.
        """
        _url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        _word_re = re.compile(r"\b\w+\b")
        rows: List[List[float]] = []

        for text in texts:
            if not text or not text.strip():
                rows.append([0.0] * 8)
                continue

            # caps_ratio
            alpha_chars = [c for c in text if c.isalpha()]
            caps_ratio = (
                sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if alpha_chars
                else 0.0
            )

            exclamation_count = float(text.count("!"))
            question_count = float(text.count("?"))
            url_count = float(len(_url_re.findall(text)))

            tokens = _word_re.findall(text.lower())
            word_count = float(len(tokens))
            avg_word_length = (
                sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
            )
            lexical_diversity = (
                len(set(tokens)) / len(tokens) if tokens else 0.0
            )

            punct_chars = sum(1 for c in text if c in string.punctuation)
            punctuation_ratio = punct_chars / len(text) if text else 0.0

            rows.append(
                [
                    caps_ratio,
                    exclamation_count,
                    question_count,
                    url_count,
                    word_count,
                    avg_word_length,
                    lexical_diversity,
                    punctuation_ratio,
                ]
            )

        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        """Fit both TF-IDF vectorisers on *texts*.

        Parameters
        ----------
        texts:
            List of pre-processed text strings used for training.

        Returns
        -------
        FeatureExtractor
            Self, for method chaining.
        """
        self._word_tfidf.fit(texts)
        self._char_tfidf.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]):
        """Transform *texts* into a sparse feature matrix.

        Concatenates word TF-IDF, char TF-IDF, and linguistic features.

        Parameters
        ----------
        texts:
            List of pre-processed text strings.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of shape ``(n_samples, n_features)``.
        """
        word_feats = self._word_tfidf.transform(texts)
        char_feats = self._char_tfidf.transform(texts)
        ling_feats = csr_matrix(self._extract_linguistic_features(texts))
        return hstack([word_feats, char_feats, ling_feats], format="csr")

    def fit_transform(self, texts: List[str]):
        """Fit then transform *texts*.

        Convenience method equivalent to calling :meth:`fit` followed by
        :meth:`transform`.

        Parameters
        ----------
        texts:
            List of pre-processed text strings.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: Path) -> None:
        """Persist both vectorisers to *path* directory.

        Creates two files:
        - ``word_tfidf.joblib``
        - ``char_tfidf.joblib``

        Parameters
        ----------
        path:
            Directory in which to save the vectoriser files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._word_tfidf, path / "word_tfidf.joblib")
        joblib.dump(self._char_tfidf, path / "char_tfidf.joblib")

    def load(self, path: Path) -> None:
        """Load both vectorisers from *path* directory.

        Expects the same two files written by :meth:`save`.

        Parameters
        ----------
        path:
            Directory containing ``word_tfidf.joblib`` and
            ``char_tfidf.joblib``.
        """
        path = Path(path)
        self._word_tfidf = joblib.load(path / "word_tfidf.joblib")
        self._char_tfidf = joblib.load(path / "char_tfidf.joblib")
        self._is_fitted = True

    @property
    def feature_names_(self) -> list:
        """Return vocabulary keys from the word TF-IDF vectoriser.

        Returns
        -------
        list of str
        """
        if not self._is_fitted:
            return []
        return list(self._word_tfidf.vocabulary_.keys())
