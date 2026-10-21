"""Utility helpers for the Fake News Detection project."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def get_project_root() -> Path:
    """Return the project root directory (parent of the src package).

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent


def load_config() -> Dict[str, Any]:
    """Load and return the merged project configuration from config/config.yaml.

    Returns
    -------
    dict
        Configuration dictionary loaded from YAML.  Returns an empty dict if
        the file does not exist so callers always receive a dict.
    """
    config_path = get_project_root() / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def setup_logger(name: str) -> logging.Logger:
    """Create and return a configured :class:`logging.Logger`.

    The logger writes INFO-and-above messages to *stdout* with a timestamp
    prefix.  If a handler is already attached (e.g. the root logger was
    configured earlier) a duplicate is not added.

    Parameters
    ----------
    name:
        The logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def download_nltk_resources() -> None:
    """Download required NLTK corpora and models silently if not present.

    Downloads: punkt, punkt_tab, stopwords, wordnet, averaged_perceptron_tagger.
    Errors are silently swallowed so the application can still start even when
    the network is unavailable.
    """
    import nltk  # local import to keep utils importable without nltk

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for resource_path, resource_id in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_id, quiet=True)
            except Exception:  # noqa: BLE001
                pass


def ensure_dirs(paths: List[Path]) -> None:
    """Create directories in *paths* if they do not already exist.

    Parameters
    ----------
    paths:
        List of :class:`~pathlib.Path` objects to create.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
