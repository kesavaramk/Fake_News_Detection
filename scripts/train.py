"""Training script for the Fake News Detection model.

Generates synthetic training data, trains classical ML models, and saves
all artefacts to disk.

Usage
-----
    python scripts/train.py [--output-dir PATH]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so `src` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

_CAPS_WORDS = [
    "SHOCKING", "BREAKING", "EXPOSED", "URGENT", "BOMBSHELL",
    "BANNED", "SUPPRESSED", "CENSORED", "REVEALED", "TRUTH",
    "ALERT", "WARNING", "EXCLUSIVE", "CONFIRMED", "LEAKED",
]

_CLICKBAIT_OPENERS = [
    "YOU WON'T BELIEVE",
    "THEY DON'T WANT YOU TO KNOW",
    "WAKE UP",
    "SHARE BEFORE DELETED",
    "MUST READ",
    "WHAT THEY'RE HIDING",
    "THE REAL TRUTH ABOUT",
    "EVERYONE IS IGNORING",
]

_CONSPIRACY_PHRASES = [
    "deep state",
    "mainstream media lies",
    "globalist agenda",
    "New World Order",
    "Big Pharma",
    "shadow government",
    "elites don't want you to know",
    "controlled by the cabal",
    "secret society",
    "population control",
]

_FAKE_TOPICS = [
    "vaccines", "5G towers", "water supply", "the election", "currency",
    "the government", "Hollywood celebrities", "tech billionaires",
    "global warming data", "the moon landing",
]

_FAKE_MIDDLES = [
    "is a COMPLETE HOAX designed to control you!!",
    "has been SECRETLY altered by the deep state!!!",
    "is PROOF that mainstream media lies to us every day!!",
    "exposes the GLOBALIST AGENDA once and for all!!",
    "CONFIRMS what Big Pharma doesn't want you to know!!!",
    "reveals the TRUTH they've been SUPPRESSING for years!!",
    "shows the New World Order is REAL and HAPPENING NOW!!!",
    "is BANNED in 47 countries — SHARE BEFORE DELETED!!",
    "EXPOSES the shadow government controlling everything!!",
    "is what elites don't want you to read!!!",
]

_NEUTRAL_ATTRIBUTION = [
    "According to",
    "Officials said",
    "Researchers found",
    "The study showed",
    "Analysts noted",
    "Data published by",
    "Experts cited in",
    "The report indicated",
    "Scientists confirmed",
    "The agency stated",
]

_REAL_TOPICS = [
    ("economy", [
        "the unemployment rate fell to {pct}% in {month}, the Labor Department reported.",
        "consumer prices rose {pct}% year-over-year, according to the Bureau of Labor Statistics.",
        "the Federal Reserve held interest rates steady at its {month} meeting.",
        "GDP grew at an annualised rate of {pct}% in the third quarter, officials said.",
        "retail sales increased by {amt} billion dollars compared to the previous quarter.",
    ]),
    ("science", [
        "a new study published in {journal} found that {pct}% of participants showed improvement.",
        "researchers at {university} identified a compound that reduces inflammation by {pct}%.",
        "the peer-reviewed findings, published in {journal}, analysed data from {n} subjects.",
        "a team of scientists confirmed the results after {n} independent replications.",
        "the research, funded by the National Institutes of Health, tracked {n} patients over {yrs} years.",
    ]),
    ("local government", [
        "the city council voted {vote} to approve a new infrastructure plan costing {amt} million dollars.",
        "the mayor's office announced a budget of {amt} million dollars for road improvements.",
        "officials approved a zoning change affecting {n} residential properties.",
        "the school board passed a resolution to hire {n} additional teachers next semester.",
        "a new transit line connecting {n} neighbourhoods will open in {month}.",
    ]),
    ("sports", [
        "the team won {n}-{n2} in a closely contested match on {day}.",
        "the athlete set a new personal record of {pct} seconds in the {n}00-metre event.",
        "the coach cited team discipline as a key factor after the {n}-game winning streak.",
        "attendance at the stadium reached {n},000 fans during the season opener.",
        "the league announced a {amt} million dollar broadcast deal for the upcoming season.",
    ]),
    ("technology", [
        "the company reported quarterly revenue of {amt} billion dollars, up {pct}% year-on-year.",
        "engineers at {university} developed a chip that processes data {pct}% faster.",
        "the software update, released on {day}, patches {n} security vulnerabilities.",
        "a survey of {n},000 users found that {pct}% prefer the new interface design.",
        "the product launch drew {n},000 pre-orders within the first {n2} hours.",
    ]),
]

_JOURNALS = [
    "Nature", "The Lancet", "JAMA", "Science", "Cell",
    "PNAS", "New England Journal of Medicine",
]

_UNIVERSITIES = [
    "MIT", "Stanford University", "Harvard University",
    "the University of Michigan", "Oxford University",
]

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _rand_pct() -> str:
    return str(round(_RNG.uniform(0.5, 12.5), 1))


def _rand_n() -> str:
    return str(int(_RNG.integers(2, 200)))


def _rand_n2() -> str:
    return str(int(_RNG.integers(1, 10)))


def _rand_amt() -> str:
    return str(round(float(_RNG.uniform(5, 500)), 1))


def _rand_yrs() -> str:
    return str(int(_RNG.integers(2, 20)))


def _fill_template(template: str) -> str:
    """Fill placeholder fields in a real-news template."""
    return (
        template
        .replace("{pct}", _rand_pct())
        .replace("{n}", _rand_n())
        .replace("{n2}", _rand_n2())
        .replace("{amt}", _rand_amt())
        .replace("{yrs}", _rand_yrs())
        .replace("{journal}", _RNG.choice(_JOURNALS))
        .replace("{university}", _RNG.choice(_UNIVERSITIES))
        .replace("{month}", _RNG.choice(_MONTHS))
        .replace("{day}", _RNG.choice(_DAYS))
        .replace("{vote}", f"{_RNG.integers(4,9)}-{_RNG.integers(1,4)}")
    )


def _make_fake() -> str:
    """Generate a single fake-news article string."""
    opener = _RNG.choice(_CLICKBAIT_OPENERS)
    topic = _RNG.choice(_FAKE_TOPICS)
    middle = _RNG.choice(_FAKE_MIDDLES)
    caps1 = _RNG.choice(_CAPS_WORDS)
    caps2 = _RNG.choice(_CAPS_WORDS)
    conspiracy = _RNG.choice(_CONSPIRACY_PHRASES)

    exclamations = "!" * int(_RNG.integers(2, 5))
    questions = "?" * int(_RNG.integers(1, 4))

    parts = [
        f"{opener}: {caps1}! {topic} {middle}",
        f"This is {caps2} evidence of the {conspiracy}.",
        f"Why won't they tell you this{questions}",
        f"The {conspiracy} is real and it's HAPPENING{exclamations}",
        f"WAKE UP and see what they're doing to {topic}!!",
    ]
    return " ".join(parts)


def _make_real() -> str:
    """Generate a single real-news article string."""
    topic_name, templates = _REAL_TOPICS[int(_RNG.integers(0, len(_REAL_TOPICS)))]
    template = _RNG.choice(templates)
    sentence = _fill_template(template)

    attribution = _RNG.choice(_NEUTRAL_ATTRIBUTION)
    add_template = _RNG.choice(templates)
    add_sentence = _fill_template(add_template)

    return f"{attribution} {sentence} {add_sentence}"


def generate_synthetic_data(n_samples: int = 2000):
    """Return (texts, labels) arrays with equal fake/real split.

    Parameters
    ----------
    n_samples:
        Total number of samples (must be even).

    Returns
    -------
    tuple[list[str], list[int]]
        texts and binary labels (1 = fake, 0 = real).
    """
    half = n_samples // 2
    texts = [_make_fake() for _ in range(half)] + [_make_real() for _ in range(half)]
    labels = [1] * half + [0] * half

    # Shuffle together
    indices = list(range(n_samples))
    _RNG.shuffle(indices)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(output_dir: Path) -> None:
    """Train models and save artefacts to *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory where models will be saved.
    """
    import nltk
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from src.utils import download_nltk_resources
    from src.preprocessing import TextProcessor
    from src.features import FeatureExtractor
    from src.models import ClassicalModels

    # ---- NLTK resources ----
    print("Downloading NLTK resources…")
    download_nltk_resources()
    for resource in ("punkt", "stopwords", "wordnet", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    # ---- Synthetic data ----
    print("Generating 2000 synthetic training examples…")
    texts, labels = generate_synthetic_data(2000)
    y = np.array(labels)
    print(f"  Fake: {sum(y == 1)}  Real: {sum(y == 0)}")

    # ---- Pre-process ----
    print("Pre-processing texts…")
    processor = TextProcessor()
    processed = [processor.process(t) for t in texts]

    # ---- Feature extraction ----
    print("Extracting features…")
    config = {}  # use defaults
    extractor = FeatureExtractor(config)
    X = extractor.fit_transform(processed)
    print(f"  Feature matrix shape: {X.shape}")

    # ---- Train ----
    print("Training classical models…")
    classical_models = ClassicalModels(config)
    classical_models.train(X, y)

    # ---- Evaluate on training set ----
    y_pred = classical_models.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    report = classification_report(y, y_pred, target_names=["REAL", "FAKE"])

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"F1 Score : {f1:.4f}  (weighted)")
    print("\nClassification Report:")
    print(report)

    # ---- Save models ----
    models_dir = output_dir / "classifiers"
    classical_models.save(models_dir)
    print(f"Models saved to: {models_dir}")

    vectorizers_dir = output_dir / "vectorizers"
    extractor.save(vectorizers_dir)
    print(f"Feature extractor saved to: {vectorizers_dir}")

    # ---- Metadata ----
    metadata = {
        "accuracy": round(float(acc), 6),
        "f1_score": round(float(f1), 6),
        "training_samples": len(texts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Fake News Detection models on synthetic data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "models",
        help="Directory to save trained models (default: <project_root>/models).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train(output_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
