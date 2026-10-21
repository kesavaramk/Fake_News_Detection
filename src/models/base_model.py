"""Abstract base class for all Fake News Detection models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """Abstract interface that every model in the pipeline must implement.

    Concrete subclasses must override all abstract methods.
    """

    @abstractmethod
    def train(self, X, y: np.ndarray) -> None:
        """Fit the model on training data.

        Parameters
        ----------
        X:
            Feature matrix (dense or sparse).
        y:
            Label array of shape ``(n_samples,)``.
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return class predictions for *X*.

        Parameters
        ----------
        X:
            Feature matrix (dense or sparse).

        Returns
        -------
        np.ndarray
            Integer label array of shape ``(n_samples,)``.
        """

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities for *X*.

        Parameters
        ----------
        X:
            Feature matrix (dense or sparse).

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, 2)`` with per-class probabilities.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the model to *path*.

        Parameters
        ----------
        path:
            Destination file or directory path.
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a persisted model from *path*.

        Parameters
        ----------
        path:
            Source file or directory path.
        """
