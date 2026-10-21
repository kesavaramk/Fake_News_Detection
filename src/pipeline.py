"""High-level prediction pipeline for the Fake News Detection system."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import get_project_root, load_config, setup_logger


# Clickbait / sensationalist phrase patterns used by the demo fallback
_CLICKBAIT_PHRASES = [
    r"you won'?t believe",
    r"\bshocking\b",
    r"they don'?t want you",
    r"\bwake up\b",
    r"\bshare this\b",
    r"mainstream media",
    r"deep state",
    r"\bfake news\b",
]
_CLICKBAIT_RE = re.compile(
    "|".join(_CLICKBAIT_PHRASES), flags=re.IGNORECASE
)


class FakeNewsPipeline:
    """End-to-end prediction pipeline.

    Wraps pre-processing, feature extraction, ensemble prediction, and
    explainability into a single ``predict`` interface.  When no trained
    models are present on disk, the pipeline falls back to a lightweight
    rule-based demo mode so the application always returns useful output.

    Parameters
    ----------
    config:
        Optional configuration dict.  If ``None`` the config is loaded from
        ``config/config.yaml`` via :func:`~src.utils.load_config`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the pipeline.

        Parameters
        ----------
        config:
            Optional pre-loaded configuration dict.
        """
        self._config: Dict[str, Any] = config if config is not None else load_config()
        self._logger: logging.Logger = setup_logger(__name__)
        self._is_loaded: bool = False

        # These are set by load()
        self._preprocessor = None
        self._extractor = None
        self._classical = None
        self._ensemble = None
        self._explainer = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` when trained models are loaded and ready."""
        return self._is_loaded

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load all trained artefacts from the ``models/`` directory.

        Attempts to load the feature extractor and all three classical
        classifiers.  Sets :attr:`is_loaded` to ``True`` on success.

        Returns
        -------
        bool
            ``True`` if all artefacts loaded successfully, ``False``
            otherwise (e.g., models have not been trained yet).
        """
        try:
            from src.preprocessing import TextProcessor
            from src.features import FeatureExtractor
            from src.models import ClassicalModels, EnsembleModel
            from src.explainability import Explainer

            models_dir = get_project_root() / "models"

            self._preprocessor = TextProcessor()

            self._extractor = FeatureExtractor(self._config)
            self._extractor.load(models_dir / "vectorizers")

            self._classical = ClassicalModels(self._config)
            self._classical.load(models_dir / "classifiers")

            self._ensemble = EnsembleModel(self._classical, self._config)
            self._explainer = Explainer(
                self._extractor, self._classical, self._config
            )

            self._is_loaded = True
            self._logger.info("All models loaded successfully from %s.", models_dir)
            return True

        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Could not load trained models (%s). Running in demo mode.", exc
            )
            self._is_loaded = False
            return False

    # ------------------------------------------------------------------
    # Demo / fallback
    # ------------------------------------------------------------------

    def _demo_predict(self, text: str) -> Dict[str, Any]:
        """Rule-based fallback prediction used when no models are loaded.

        Scores text on:
        - ALL-CAPS word ratio
        - Exclamation mark count
        - Clickbait phrase matches
        - Question mark count

        Parameters
        ----------
        text:
            Raw news text.

        Returns
        -------
        dict
            Same structure as the full :meth:`predict` output with
            ``mode='demo'``.
        """
        words = text.split()
        total_words = max(len(words), 1)

        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        caps_score = min(caps_words / total_words * 100, 30)

        exclamation_count = text.count("!")
        exclaim_score = min(exclamation_count * 5, 25)

        clickbait_matches = len(_CLICKBAIT_RE.findall(text))
        clickbait_score = min(clickbait_matches * 10, 30)

        question_count = text.count("?")
        question_score = min(question_count * 3, 15)

        score = caps_score + exclaim_score + clickbait_score + question_score
        score = max(0.0, min(score, 100.0))

        is_fake = score >= 35
        label = "FAKE" if is_fake else "REAL"
        confidence = round(score if is_fake else 100.0 - score, 2)
        confidence = max(confidence, 50.0)

        is_uncertain = confidence < 65.0

        # Build a minimal explanation without ML models
        signals: List[str] = []
        if caps_words > 0:
            signals.append(f"{caps_words} ALL-CAPS word(s)")
        if exclamation_count > 0:
            signals.append(f"{exclamation_count} exclamation mark(s)")
        if clickbait_matches > 0:
            signals.append(f"{clickbait_matches} sensationalist phrase(s)")
        if question_count > 0:
            signals.append(f"{question_count} question mark(s)")

        if signals:
            summary = (
                f"[Demo mode] Article flagged as {label} ({confidence:.1f}% confidence) "
                f"based on: {', '.join(signals)}."
            )
        else:
            summary = (
                f"[Demo mode] Article appears {label} ({confidence:.1f}% confidence). "
                "No strong sensationalist signals were detected."
            )

        linguistic_stats = {
            "Word Count": len(words),
            "Exclamation Marks": exclamation_count,
            "Question Marks": question_count,
            "ALL-CAPS Words": caps_words,
            "Clickbait Phrases": clickbait_matches,
        }

        return {
            "label": label,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "explanation": summary,
            "highlighted_html": text,
            "top_words": {},
            "linguistic_stats": linguistic_stats,
            "mode": "demo",
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict whether *text* is fake or real news.

        Uses the full ML pipeline when models are loaded; falls back to
        :meth:`_demo_predict` otherwise.

        Parameters
        ----------
        text:
            Raw news article text.

        Returns
        -------
        dict
            - ``label`` – ``'FAKE'`` or ``'REAL'``.
            - ``confidence`` – float in ``[0, 100]``.
            - ``is_uncertain`` – bool.
            - ``explanation`` – human-readable summary string.
            - ``highlighted_html`` – HTML-annotated text.
            - ``top_words`` – dict of influential words.
            - ``linguistic_stats`` – dict of surface statistics.
            - ``mode`` – ``'ml'`` or ``'demo'``.
        """
        if not self._is_loaded:
            return self._demo_predict(text)

        try:
            processed = self._preprocessor.process(text)
            features = self._extractor.transform([processed])

            result = self._ensemble.predict_single(features)

            explanation_dict = self._explainer.explain_prediction(
                text, result["label"], result["confidence"]
            )
            linguistic_stats = self._explainer.get_linguistic_analysis(text)

            return {
                "label": result["label"],
                "confidence": result["confidence"],
                "is_uncertain": result["is_uncertain"],
                "explanation": explanation_dict["summary"],
                "highlighted_html": explanation_dict["highlighted_html"],
                "top_words": explanation_dict["top_words"],
                "linguistic_stats": linguistic_stats,
                "mode": "ml",
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Prediction error: %s. Falling back to demo.", exc)
            result = self._demo_predict(text)
            result["mode"] = "demo"
            return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict a list of texts.

        Parameters
        ----------
        texts:
            List of raw news article strings.

        Returns
        -------
        list[dict]
            A list of result dicts in the same order as *texts*.
        """
        return [self.predict(text) for text in texts]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_and_save(
        self,
        X_train,
        y_train,
        texts_train: List[str],
    ) -> Dict[str, Any]:
        """Train the full pipeline and persist artefacts to disk.

        Parameters
        ----------
        X_train:
            Pre-extracted feature matrix, or ``None`` to build features from
            *texts_train*.
        y_train:
            Binary label array (0 = real, 1 = fake).
        texts_train:
            Raw training texts used to fit the vectorisers.

        Returns
        -------
        dict
            Training metrics:

            - ``train_samples`` – number of training examples.
            - ``n_features`` – total feature dimensions.
            - ``models_saved_to`` – string path.
        """
        from src.preprocessing import TextProcessor
        from src.features import FeatureExtractor
        from src.models import ClassicalModels, EnsembleModel
        from src.explainability import Explainer
        import numpy as np

        self._logger.info("Starting training on %d samples.", len(texts_train))

        # Pre-process
        self._preprocessor = TextProcessor()
        processed_texts = [self._preprocessor.process(t) for t in texts_train]

        # Feature extraction
        self._extractor = FeatureExtractor(self._config)
        X = self._extractor.fit_transform(processed_texts)

        # Use pre-extracted features if provided
        if X_train is not None:
            X = X_train

        # Train models
        self._classical = ClassicalModels(self._config)
        self._classical.train(X, y_train)

        self._ensemble = EnsembleModel(self._classical, self._config)
        self._explainer = Explainer(self._extractor, self._classical, self._config)

        # Persist artefacts
        models_dir = get_project_root() / "models"
        self._extractor.save(models_dir / "vectorizers")
        self._classical.save(models_dir / "classifiers")

        self._is_loaded = True
        self._logger.info("Training complete. Artefacts saved to %s.", models_dir)

        return {
            "train_samples": len(texts_train),
            "n_features": X.shape[1],
            "models_saved_to": str(models_dir),
        }
