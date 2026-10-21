"""Ensemble model that combines classical classifiers via weighted averaging."""

from typing import Any, Dict, Tuple

import numpy as np

from .classical import ClassicalModels


class EnsembleModel:
    """Weighted soft-voting ensemble over LR, SVC, and RF.

    Combines the probability outputs of three :class:`ClassicalModels`
    classifiers using configurable weights.

    Parameters
    ----------
    classical_models:
        A trained (or freshly constructed) :class:`ClassicalModels` instance.
    config:
        Project configuration dictionary.  Relevant keys (all optional):

        - ``ensemble.weights.lr`` (float, default 0.5)
        - ``ensemble.weights.svc`` (float, default 0.3)
        - ``ensemble.weights.rf`` (float, default 0.2)
        - ``ensemble.uncertainty_threshold`` (float, default 0.60)
        - ``ensemble.fake_threshold`` (float, default 0.50)
    """

    def __init__(
        self,
        classical_models: ClassicalModels,
        config: Dict[str, Any],
    ) -> None:
        """Store models and read weights / thresholds from *config*.

        Parameters
        ----------
        classical_models:
            Trained :class:`ClassicalModels` instance.
        config:
            Project configuration dictionary.
        """
        self._models = classical_models

        ens_cfg = config.get("ensemble", {})
        weights_cfg = ens_cfg.get("weights", {})

        self._w_lr = float(weights_cfg.get("lr", 0.5))
        self._w_svc = float(weights_cfg.get("svc", 0.3))
        self._w_rf = float(weights_cfg.get("rf", 0.2))

        # Normalise so weights always sum to 1
        total = self._w_lr + self._w_svc + self._w_rf
        if total > 0:
            self._w_lr /= total
            self._w_svc /= total
            self._w_rf /= total

        self._uncertainty_threshold = float(
            ens_cfg.get("uncertainty_threshold", 0.60)
        )
        self._fake_threshold = float(ens_cfg.get("fake_threshold", 0.50))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_proba_ensemble(self, X) -> np.ndarray:
        """Compute a weighted average probability over all three classifiers.

        Parameters
        ----------
        X:
            Feature matrix (dense or sparse).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` – columns are [P(real), P(fake)].
        """
        probas = self._models.predict_proba(X)
        ensemble_proba = (
            self._w_lr * probas["lr"]
            + self._w_svc * probas["svc"]
            + self._w_rf * probas["rf"]
        )
        return ensemble_proba

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and confidence scores.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - ``labels`` – integer array of shape ``(n_samples,)``
              (1 = fake, 0 = real).
            - ``confidences`` – float array of shape ``(n_samples,)``
              in range ``[0, 100]``.
        """
        proba = self.predict_proba_ensemble(X)
        labels = (proba[:, 1] >= self._fake_threshold).astype(int)
        confidences = np.max(proba, axis=1) * 100.0
        return labels, confidences

    def predict_single(self, X) -> Dict[str, Any]:
        """Predict a single sample and return a rich result dict.

        Parameters
        ----------
        X:
            Feature matrix for a single sample (1 row).

        Returns
        -------
        dict
            - ``label`` – ``'FAKE'`` or ``'REAL'``.
            - ``confidence`` – float in range ``[0, 100]``.
            - ``is_uncertain`` – ``True`` when the highest per-class
              probability falls below :attr:`_uncertainty_threshold`.
        """
        proba = self.predict_proba_ensemble(X)
        fake_proba = float(proba[0, 1])
        real_proba = float(proba[0, 0])
        is_fake = fake_proba >= self._fake_threshold
        confidence = max(fake_proba, real_proba) * 100.0
        is_uncertain = max(fake_proba, real_proba) < self._uncertainty_threshold

        return {
            "label": "FAKE" if is_fake else "REAL",
            "confidence": round(confidence, 2),
            "is_uncertain": is_uncertain,
        }
