"""Load and validate the small email dataset and derive message-level features."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "emails.csv"
FEATURE_COLUMNS = (
    "word_count",
    "exclamation_count",
    "money_words",
    "all_caps",
    "url_count",
    "spam_keyword_count",
)

_WORD_PATTERN = re.compile(r"\b[\w']+\b")
_MONEY_WORD_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    for word in (
        "money",
        "cash",
        "dollar",
        "dollars",
        "buck",
        "bucks",
        "usd",
        "profit",
        "income",
        "bonus",
    )
)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SPAM_KEYWORD_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for keyword in (
        "act now",
        "claim",
        "congratulations",
        "free",
        "guaranteed",
        "limited time",
        "prize",
        "urgent",
        "verify",
        "winner",
    )
)


def extract_features(email: str) -> dict[str, int]:
    """Return hand-designed, message-level features for one email."""
    words = _WORD_PATTERN.findall(email)
    return {
        "word_count": len(words),
        "url_count": len(_URL_PATTERN.findall(email)),
        "exclamation_count": email.count("!"),
        "money_words": sum(
            len(pattern.findall(email)) for pattern in _MONEY_WORD_PATTERNS
        ),
        "all_caps": sum(
            1 for word in words if len(word) > 1 and word.isalpha() and word.isupper()
        ),
        "spam_keyword_count": sum(
            len(pattern.findall(email)) for pattern in _SPAM_KEYWORD_PATTERNS
        ),
    }


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV, validate its labels and content, and add engineered features."""
    data = pd.read_csv(path)
    required_columns = {"email", "label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )
    if data.empty:
        raise ValueError("Dataset must contain at least one email.")
    if len(data) != 16:
        raise ValueError(f"Dataset must contain exactly 16 emails; found {len(data)}.")
    if data["email"].isna().any() or data["email"].astype(str).str.strip().eq("").any():
        raise ValueError("Every row must contain a non-empty email.")
    data["email"] = data["email"].astype(str)
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    invalid_labels = sorted(set(data["label"]) - {"ham", "spam"})
    if invalid_labels:
        raise ValueError(f"Labels must be 'ham' or 'spam'; found: {invalid_labels}")
    class_counts = data["label"].value_counts()
    if set(class_counts.index) != {"ham", "spam"} or class_counts.to_dict() != {
        "ham": 8,
        "spam": 8,
    }:
        raise ValueError(
            "Dataset must contain exactly 8 ham and 8 spam emails; "
            f"found {class_counts.to_dict()}."
        )

    features = data["email"].map(extract_features).apply(pd.Series)
    return pd.concat([data, features], axis=1)


def main() -> None:
    data = load_data()
    print(f"Validated {len(data)} emails from {DATA_PATH}.")
    print("Class counts:")
    print(data["label"].value_counts().sort_index().to_string())
    print(f"Engineered features: {', '.join(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()