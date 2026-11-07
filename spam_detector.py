"""Train, evaluate, and demonstrate a small email spam classifier."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from prepare_data import FEATURE_COLUMNS, extract_features, load_data

RANDOM_STATE = 42
TEST_SIZE = 0.25
LABEL_ORDER = ("ham", "spam")
NEAR_DUPLICATE_THRESHOLD = 0.85

CHALLENGE_EMAILS = (
    (
        "Legitimate-looking promotion",
        "Hello, our bookstore has a free weekend event. Read the schedule in your account when convenient.",
        "ham",
    ),
    (
        "Phishing-style account notice",
        "Hi, an invoice is ready for review. Verify your account at https://billing-example.test before payment.",
        "spam",
    ),
    (
        "Urgent-sounding legitimate reminder",
        "Urgent reminder from the school office: the field trip form is due today. Please reply to this email.",
        "ham",
    ),
)


def build_model() -> Pipeline:
    """Build a pipeline that fits text vocabulary only on the training partition."""
    features = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), stop_words="english"), "email"),
            ("handcrafted", "passthrough", list(FEATURE_COLUMNS)),
        ]
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def train_model(
    data: pd.DataFrame | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    """Split, fit, and predict the held-out test partition."""
    dataset = load_data() if data is None else data.copy()
    columns = ["email", *FEATURE_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[columns],
        dataset["label"],
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=dataset["label"],
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = pd.Series(model.predict(x_test), index=y_test.index, name="prediction")
    return model, x_test, y_test, predictions


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate binary classification metrics with spam as the positive class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label="spam", zero_division=0),
    }


def find_near_duplicates(
    data: pd.DataFrame,
    test_indices: pd.Index,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[tuple[int, int, float, bool]]:
    """Find highly similar email pairs and identify pairs spanning the split."""
    normalized_emails = [
        re.sub(r"\W+", " ", email.lower()).strip() for email in data["email"]
    ]
    test_index_set = set(test_indices)
    matches = []
    for left in range(len(data)):
        for right in range(left + 1, len(data)):
            similarity = SequenceMatcher(
                None, normalized_emails[left], normalized_emails[right]
            ).ratio()
            if similarity >= threshold:
                crosses_split = (
                    (data.index[left] in test_index_set)
                    != (data.index[right] in test_index_set)
                )
                matches.append((left, right, similarity, crosses_split))
    return matches


def predict_email(model: Pipeline, email: str) -> tuple[str, float]:
    """Return the predicted label and model-estimated spam probability."""
    row: dict[str, Any] = {"email": email, **extract_features(email)}
    example = pd.DataFrame([row])
    probabilities = model.predict_proba(example)[0]
    spam_index = list(model.named_steps["classifier"].classes_).index("spam")
    return str(model.predict(example)[0]), float(probabilities[spam_index])


def main() -> None:
    data = load_data()
    model, x_test, y_test, predictions = train_model(data)
    metrics = calculate_metrics(y_test, predictions)
    train_indices = data.index.difference(x_test.index)
    x_train = data.loc[train_indices, ["email", *FEATURE_COLUMNS]]
    y_train = data.loc[train_indices, "label"]

    print(f"Dataset: {len(data)} emails ({len(x_test)} held out for testing)")
    print(
        f"Split: {1 - TEST_SIZE:.0%} training / {TEST_SIZE:.0%} testing; "
        f"random_state={RANDOM_STATE}"
    )
    training_accuracy = accuracy_score(y_train, model.predict(x_train))
    print(f"Training accuracy (diagnostic only): {training_accuracy:.3f}")
    print("\nTest-set metrics (spam is the positive class):")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

    matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)
    print("\nConfusion matrix (rows = actual, columns = predicted; order: ham, spam):")
    print(matrix)

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["features"].get_feature_names_out()
    coefficients = classifier.coef_[0]
    top_features = sorted(
        zip(feature_names, coefficients),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    print("\nTop 10 Logistic Regression feature weights (positive favors spam):")
    for feature_name, weight in top_features[:10]:
        print(f"- {feature_name}: {weight:+.4f}")
    engineered_weights = [
        (name, weight)
        for name, weight in zip(feature_names, coefficients)
        if name.startswith("handcrafted__")
    ]
    print("\nHandcrafted feature weights (positive favors spam):")
    for feature_name, weight in sorted(
        engineered_weights, key=lambda item: abs(item[1]), reverse=True
    ):
        print(f"- {feature_name.removeprefix('handcrafted__')}: {weight:+.4f}")

    near_duplicates = find_near_duplicates(data, x_test.index)
    cross_split_duplicates = [
        match for match in near_duplicates if match[3]
    ]
    print(
        f"\nNear-duplicate scan (SequenceMatcher threshold "
        f"{NEAR_DUPLICATE_THRESHOLD:.2f}):"
    )
    if near_duplicates:
        for left, right, similarity, crosses_split in near_duplicates:
            location = "cross-split" if crosses_split else "same split"
            print(
                f"- rows {left + 1} and {right + 1}: "
                f"similarity={similarity:.3f} ({location})"
            )
    else:
        print("- No near-duplicate pairs found.")
    if cross_split_duplicates:
        print(
            f"- Potential train/test near-duplicate leakage: "
            f"{len(cross_split_duplicates)} cross-split pair(s)."
        )
    else:
        print("- No cross-split near-duplicate pairs found.")

    challenge_actual = []
    challenge_predictions = []
    print("\nChallenge examples (expected labels are documented judgments):")
    for description, email, expected_label in CHALLENGE_EMAILS:
        label, spam_probability = predict_email(model, email)
        challenge_actual.append(expected_label)
        challenge_predictions.append(label)
        result = "correct" if label == expected_label else "incorrect"
        print(
            f"- {description}: expected={expected_label}, predicted={label}, "
            f"spam_probability={spam_probability:.3f} ({result})"
        )

    challenge_matrix = confusion_matrix(
        challenge_actual,
        challenge_predictions,
        labels=LABEL_ORDER,
    )
    print(
        "\nChallenge confusion matrix "
        "(rows = expected, columns = predicted; order: ham, spam):"
    )
    print(challenge_matrix)

    false_positives = sum(
        expected == "ham" and predicted == "spam"
        for expected, predicted in zip(challenge_actual, challenge_predictions)
    )
    false_negatives = sum(
        expected == "spam" and predicted == "ham"
        for expected, predicted in zip(challenge_actual, challenge_predictions)
    )
    print(f"Challenge false positives: {false_positives}")
    print(f"Challenge false negatives: {false_negatives}")


if __name__ == "__main__":
    main()
