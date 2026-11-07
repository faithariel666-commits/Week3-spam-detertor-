# Email Spam Detector

A small, educational email classifier using pandas and scikit-learn. The repository includes 16 **synthetic example emails** (8 ham and 8 spam), six hand-designed message features, a stratified train/test split, a Logistic Regression classifier, and three challenging demonstration messages.

> **Assignment-data note:** No instructor handout or official list of the 16 sample emails was present in the repository when this project was prepared. The included examples are original synthetic substitutes, not a claim about the instructor's required wording. Replace `data/emails.csv` with the official examples if your instructor supplied a specific list; keep the `email,label` columns and `ham`/`spam` labels.

## Requirements

- Python 3.10 or newer
- pandas
- scikit-learn

## Install

From the repository root, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run the interpreter and pip directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Validate the CSV and display class counts and engineered features:

```powershell
python prepare_data.py
```

Train the classifier, print test metrics and the confusion matrix, and classify the three challenge messages:

```powershell
python spam_detector.py
```

The classifier run also prints training accuracy as a diagnostic, learned Logistic Regression feature weights, a near-duplicate scan across the train/test split, challenge predictions, and the challenge confusion matrix with false-positive and false-negative counts.

The script uses a reproducible 75%/25% stratified train/test split (`random_state=42`). The TF-IDF vocabulary is fitted inside the training pipeline, after the split, to avoid learning vocabulary from test data.

The three challenge emails have manually assigned expected labels for this educational evaluation. These labels represent the intended scenarios, not independently verified real-world truth or authoritative ground truth. The script compares model predictions with these labels and reports the challenge confusion matrix, false positives, and false negatives. The challenge set is separate from the training and test data.

## Dataset and features

`data/emails.csv` contains 16 synthetic emails (8 `ham` and 8 `spam`) and has two columns: `email` and `label`. No instructor handout or official list of sample emails was available when these examples were prepared. If your assignment requires instructor-provided examples, replace the synthetic rows with those examples before submission; keep the `email,label` columns and `ham`/`spam` labels, then rerun both scripts and refresh the analysis report with the new results.

`prepare_data.py` validates the data and derives four assignment features and two additional custom features:

- `word_count`: number of word tokens in the email.
- `exclamation_count`: number of exclamation marks.
- `money_words`: occurrences of `money`, `cash`, `dollar(s)`, `buck(s)`, `USD`, `profit`, `income`, and `bonus`.
- `all_caps`: number of all-uppercase alphabetic tokens with at least two letters.
- `url_count` (custom): counts URLs, which can be a signal of unsolicited or credential-stealing links.
- `spam_keyword_count` (custom): counts promotional or urgency terms, which appear more often in the synthetic spam examples than in ordinary messages.

The classifier combines these numeric features with word and two-word TF-IDF features. These handcrafted signals are simple teaching examples, not reliable rules for deciding whether a real email is safe. The all-caps feature is varied in the current dataset but its learned coefficient remains small.

## Project files

- `data/emails.csv` — 16 labeled synthetic examples
- `prepare_data.py` — CSV validation and feature extraction
- `spam_detector.py` — split, model, evaluation, and challenge predictions
- `analysis_report.md` — measured results and interpretation
- `requirements.txt` — Python dependencies

See [analysis_report.md](analysis_report.md) for the actual run results and limitations.
