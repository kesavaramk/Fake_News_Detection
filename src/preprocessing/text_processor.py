"""Text pre-processing utilities for the Fake News Detection pipeline."""

import re
import string
from dataclasses import dataclass, field
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from src.utils import download_nltk_resources


# Common English function words used by is_likely_english()
_COMMON_ENGLISH = frozenset(
    [
        "the", "be", "to", "of", "and", "a", "in", "that", "have",
        "it", "for", "not", "on", "with", "he", "as", "you", "do",
        "at", "this", "but", "his", "by", "from", "they", "we", "or",
        "an", "will", "my", "one", "all", "would", "there", "their",
        "is", "are", "was", "were", "has", "had", "been", "said",
    ]
)


class TextProcessor:
    """NLP text pre-processor for news article text.

    Cleans raw text, removes stop-words, lemmatises tokens, and computes
    a variety of surface-level linguistic statistics useful as features.

    Parameters
    ----------
    language:
        NLTK stop-word language code (default ``'english'``).
    """

    def __init__(self, language: str = "english") -> None:
        """Initialise the TextProcessor.

        Downloads required NLTK resources, creates a
        :class:`~nltk.stem.WordNetLemmatizer`, and loads the stop-word list
        for *language*.

        Parameters
        ----------
        language:
            Stop-word language, e.g. ``'english'``.
        """
        download_nltk_resources()
        self.language = language
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words: frozenset = frozenset(stopwords.words(language))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Return a cleaned version of *text*.

        Steps applied in order:
        1. Lower-case the string.
        2. Remove URLs (``http://``, ``https://``, ``www.`` patterns).
        3. Strip HTML tags.
        4. Remove all characters except letters, digits, and spaces.
        5. Collapse consecutive whitespace to a single space and strip.

        Parameters
        ----------
        text:
            Raw input text.

        Returns
        -------
        str
            Cleaned text.
        """
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Keep only letters, digits, and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove NLTK stop-words from *text*.

        Tokenises *text* with :func:`~nltk.tokenize.word_tokenize` and
        discards tokens that appear in the stop-word list.

        Parameters
        ----------
        text:
            Input text (should already be lower-cased).

        Returns
        -------
        str
            Text with stop-words removed, joined by spaces.
        """
        tokens = word_tokenize(text)
        filtered = [t for t in tokens if t not in self._stop_words]
        return " ".join(filtered)

    def lemmatize(self, text: str) -> str:
        """Lemmatise every token in *text*.

        Parameters
        ----------
        text:
            Input text.

        Returns
        -------
        str
            Space-joined lemmatised tokens.
        """
        tokens = word_tokenize(text)
        lemmatised = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(lemmatised)

    def process(self, text: str) -> str:
        """Full pre-processing pipeline: clean → remove stop-words → lemmatise.

        Parameters
        ----------
        text:
            Raw input text.

        Returns
        -------
        str
            Fully processed text ready for feature extraction.
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text

    def get_linguistic_stats(self, text: str) -> dict:
        """Compute surface-level linguistic statistics for *text*.

        Parameters
        ----------
        text:
            Raw input text.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``word_count`` – total number of word tokens.
            - ``sentence_count`` – number of sentences.
            - ``avg_word_length`` – mean character length of words.
            - ``avg_sentence_length`` – mean words per sentence.
            - ``lexical_diversity`` – unique tokens / total tokens.
            - ``caps_ratio`` – uppercase alpha chars / all alpha chars.
            - ``exclamation_count`` – number of ``!`` characters.
            - ``question_count`` – number of ``?`` characters.
            - ``punctuation_ratio`` – punctuation chars / total chars.
        """
        if not text or not text.strip():
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0,
                "lexical_diversity": 0.0,
                "caps_ratio": 0.0,
                "exclamation_count": 0,
                "question_count": 0,
                "punctuation_ratio": 0.0,
            }

        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        alpha_words = [w for w in words if w.isalpha()]

        word_count = len(words)
        sentence_count = len(sentences)

        avg_word_length = (
            sum(len(w) for w in alpha_words) / len(alpha_words)
            if alpha_words
            else 0.0
        )
        avg_sentence_length = word_count / sentence_count if sentence_count else 0.0

        lexical_diversity = (
            len(set(w.lower() for w in words)) / word_count if word_count else 0.0
        )

        alpha_chars = [c for c in text if c.isalpha()]
        caps_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )

        exclamation_count = text.count("!")
        question_count = text.count("?")

        punct_chars = sum(1 for c in text if c in string.punctuation)
        punctuation_ratio = punct_chars / len(text) if text else 0.0

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 4),
            "avg_sentence_length": round(avg_sentence_length, 4),
            "lexical_diversity": round(lexical_diversity, 4),
            "caps_ratio": round(caps_ratio, 4),
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "punctuation_ratio": round(punctuation_ratio, 4),
        }

    def is_likely_english(self, text: str) -> bool:
        """Return ``True`` if *text* is likely written in English.

        Uses a simple heuristic: tokenises the text and checks whether a
        minimum share of tokens are common English function words.

        Parameters
        ----------
        text:
            Input text.

        Returns
        -------
        bool
        """
        if not text or not text.strip():
            return False
        tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
        if not tokens:
            return False
        overlap = sum(1 for t in tokens if t in _COMMON_ENGLISH)
        ratio = overlap / len(tokens)
        # At least 5 % of tokens must be common English words
        return ratio >= 0.05
