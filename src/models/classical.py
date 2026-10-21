"""Classical machine-learning models for fake news classification."""

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .base_model import BaseModel


class ClassicalModels(BaseModel):
    """Wrapper around three classical classifiers: LR, SVC, and RF.

    All three models are trained together and can be persisted as a group.
    :meth:`predict` delegates to Logistic Regression as the primary model.
    :meth:`predict_proba` returns a dictionary with per-model probability
    arrays so that :class:`~src.models.ensemble.EnsembleModel` can combine
    them.

    Parameters
    ----------
    config:
        Project configuration dictionary.  Relevant keys (all optional):

        - ``models.lr.C`` (float, default 1.0)
        - ``models.lr.max_iter`` (int, default 1000)
        - ``models.lr.solver`` (str, default ``'lbfgs'``)
        - ``models.svc.C`` (float, default 1.0)
        - ``models.svc.max_iter`` (int, default 2000)
        - ``models.rf.n_estimators`` (int, default 200)
        - ``models.rf.max_depth`` (int or null, default ``None``)
        - ``models.rf.n_jobs`` (int, default -1)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Build all three model objects from *config*.

        Parameters
        ----------
        config:
            Project configuration dictionary.
        """
        model_cfg = config.get("models", {})

        lr_cfg = model_cfg.get("lr", {})
        self._lr = LogisticRegression(
            C=float(lr_cfg.get("C", 1.0)),
            max_iter=int(lr_cfg.get("max_iter", 1000)),
            solver=lr_cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        svc_cfg = model_cfg.get("svc", {})
        _base_svc = LinearSVC(
            C=float(svc_cfg.get("C", 1.0)),
            max_iter=int(svc_cfg.get("max_iter", 2000)),
            class_weight="balanced",
            random_state=42,
        )
        # Wrap SVC in CalibratedClassifierCV to expose predict_proba
        self._svc = CalibratedClassifierCV(estimator=_base_svc, cv=3)

        rf_cfg = model_cfg.get("rf", {})
        max_depth = rf_cfg.get("max_depth", None)
        self._rf = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=int(max_depth) if max_depth is not None else None,
            class_weight="balanced",
            random_state=42,
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
        )

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train(self, X, y: np.ndarray) -> None:
        """Fit all three classifiers on *(X, y)*.

        Parameters
        ----------
        X:
            Feature matrix (dense or sparse).
        y:
            Binary label array of shape ``(n_samples,)``.
        """
        self._lr.fit(X, y)
        self._svc.fit(X, y)
        self._rf.fit(X, y)

    def predict(self, X) -> np.ndarray:
        """Predict class labels using Logistic Regression.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)``.
        """
        return self._lr.predict(X)

    def predict_proba(self, X) -> Dict[str, np.ndarray]:
        """Return per-class probabilities from all three models.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        dict
            Keys ``'lr'``, ``'svc'``, ``'rf'``, each mapping to an
            ``np.ndarray`` of shape ``(n_samples, 2)``.
        """
        return {
            "lr": self._lr.predict_proba(X),
            "svc": self._svc.predict_proba(X),
            "rf": self._rf.predict_proba(X),
        }

    def save(self, path: Path) -> None:
        """Save all three models under *path* directory.

        Creates:
        - ``lr_model.joblib``
        - ``svc_model.joblib``
        - ``rf_model.joblib``

        Parameters
        ----------
        path:
            Destination directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._lr, path / "lr_model.joblib")
        joblib.dump(self._svc, path / "svc_model.joblib")
        joblib.dump(self._rf, path / "rf_model.joblib")

    def load(self, path: Path) -> None:
        """Load all three models from *path* directory.

        Parameters
        ----------
        path:
            Directory containing the three ``.joblib`` files.
        """
        path = Path(path)
        self._lr = joblib.load(path / "lr_model.joblib")
        self._svc = joblib.load(path / "svc_model.joblib")
        self._rf = joblib.load(path / "rf_model.joblib")

    # ------------------------------------------------------------------
    # Properties & extras
    # ------------------------------------------------------------------

    @property
    def lr_model(self) -> LogisticRegression:
        """The fitted :class:`~sklearn.linear_model.LogisticRegression`."""
        return self._lr

    @property
    def svc_model(self) -> CalibratedClassifierCV:
        """The fitted :class:`~sklearn.calibration.CalibratedClassifierCV`."""
        return self._svc

    @property
    def rf_model(self) -> RandomForestClassifier:
        """The fitted :class:`~sklearn.ensemble.RandomForestClassifier`."""
        return self._rf

    def get_lr_coefficients(self) -> np.ndarray:
        """Return the coefficient vector of the Logistic Regression model.

        Returns
        -------
        np.ndarray
            Shape ``(n_features,)``.
        """
        return self._lr.coef_[0]
