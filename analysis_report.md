# Email Spam Detector: Final Analysis Report

## Dataset and feature engineering

The CSV contains exactly 16 synthetic emails: 8 labeled `ham` (legitimate) and 8 labeled `spam`. No official instructor sample-email list was present in the supplied project files; these synthetic examples are substitutes, not a claim that they are the instructor's original messages. Replace them with any required official samples and rerun the analysis before submission.

`prepare_data.py` validates the `email` and `label` columns and derives the four assignment features:

| Feature | Definition |
|---|---|
| `word_count` | Number of word tokens, including alphanumeric tokens and contractions. |
| `exclamation_count` | Number of `!` characters. |
| `money_words` | Count of occurrences of `money`, `cash`, `dollar`/`dollars`, `buck`/`bucks`, `USD`, `profit`, `income`, and `bonus`. |
| `all_caps` | Count of alphabetic tokens of at least two letters that are entirely uppercase. |

It also computes two custom features:

| Custom feature | Justification |
|---|---|
| `url_count` | Spam and phishing messages may include unsolicited or credential-stealing links, so counting URLs provides a link-presence signal that may be more common in spam than ordinary correspondence. |
| `spam_keyword_count` | Promotional and urgency terms (such as `claim`, `free`, `prize`, `urgent`, and `verify`) often occur in spam lures more than in routine legitimate messages, so counting them provides a simple lure-language signal. |

The six numeric features are combined with word and two-word TF-IDF text features. Feature counts are calculated separately for each message using fixed patterns; they do not use dataset-wide label information. `all_caps` varies in this dataset, though the learned weight is small.

## Training, split, and evaluation

The program uses `train_test_split` with a stratified 75% training / 25% testing split and `random_state=42`: 12 training emails and 4 held-out test emails, with two examples from each class in the test set. The model remains Logistic Regression. The TF-IDF vectorizer is part of the scikit-learn pipeline, so it is fitted only on the training partition.

The split is necessary to evaluate performance on examples the classifier did not fit. Measuring on the training examples alone is unreliable because a model can memorize its training data and still perform poorly on new messages. Training-set accuracy is shown only as a diagnostic, not as the estimate of generalization.

### Actual test-set results

The results below are from running `python spam_detector.py` on the included CSV:

| Metric | Result |
|---|---:|
| Accuracy | 1.000 |
| Spam precision | 1.000 |
| Spam recall | 1.000 |
| Spam F1-score | 1.000 |

With spam treated as the positive class, precision of 1.000 means that both emails predicted as spam in this four-email test were actually spam. Recall of 1.000 means both actual spam emails in this test were detected. Confusion matrix (rows = actual labels; columns = predictions; order `ham`, `spam`):

```text
[[2 0]
 [0 2]]
```

All four test messages were correctly classified in this split. This is a result on only four held-out synthetic examples, not evidence that the model is 100% accurate on future or real-world emails.

## Feature weights

The model's learned Logistic Regression coefficients from the same run are printed by `spam_detector.py`. The top overall coefficients by absolute value were:

| Feature | Coefficient |
|---|---:|
| `handcrafted__spam_keyword_count` | +1.2115 |
| `handcrafted__word_count` | -0.3687 |
| `handcrafted__money_words` | +0.2061 |
| `handcrafted__exclamation_count` | +0.2035 |
| `handcrafted__url_count` | +0.1984 |
| `handcrafted__all_caps` | +0.1123 |
| `text__prize` | +0.0967 |
| `text__prize claim` | +0.0681 |
| `text__claim` | +0.0588 |
| `text__claim follow` | +0.0454 |

Positive coefficients move the model toward spam; negative coefficients move it toward ham. `spam_keyword_count` had the largest absolute coefficient in this run and pushed toward spam. Coefficient sizes are affected by feature scales (for example, raw counts versus normalized TF-IDF), so they should be read as this fitted model's weights, not as universal feature importance or causal effects.

## Three challenging emails

The three challenge examples are separate from the 16 training/test emails. Their expected labels are manually assigned judgments of the intended scenarios, not independently verified real-world truth. The model's spam probabilities are its estimates, not calibrated safety guarantees; the label threshold is 0.5.

| Challenge | Expected label | Predicted | Spam probability | Outcome |
|---|---|---|---:|---|
| “Hello, our bookstore has a free weekend event. Read the schedule in your account when convenient.” | ham | ham | 0.345 | Correct; this legitimate-looking message did not fool the classifier. |
| “Hi, an invoice is ready for review. Verify your account at https://billing-example.test before payment.” | spam | ham | 0.317 | **False negative: the suspicious message fooled the classifier and bypassed spam classification.** |
| “Urgent reminder from the school office: the field trip form is due today. Please reply to this email.” | ham | ham | 0.201 | Correct; this legitimate urgent message did not trigger a spam prediction. |

Challenge confusion matrix (rows = manually assigned expected label; columns = model prediction; order `ham`, `spam`):

```text
[[2 0]
 [1 0]]
```

This means the two expected ham emails were both predicted ham (2 true negatives, 0 false positives); the expected spam email was predicted ham (1 false negative, 0 true positives). Thus, **false positives = 0** and **false negatives = 1**. The suspicious account-verification example was the one successful evasion; the two legitimate examples were classified as expected. This three-email challenge set is intentionally small and hand-authored, not a reliable accuracy sample.

False negatives are generally more harmful for a spam filter because phishing or scam messages reach the user's inbox and can lead to credential theft or financial loss. False positives also matter: legitimate messages may be hidden or missed. The cost depends on use, so a real system would need an explicit policy for balancing those risks.

### Why the account-verification message was missed

The message contains one URL and one matched urgency/promotion keyword (`verify`), but its spam probability was only 0.317, below the 0.5 decision threshold. A learned model coefficient is not a rule: the model has seen only a handful of synthetic messages and cannot reliably infer that this particular link and account-verification context are suspicious. The sparse training examples provide little context for distinguishing legitimate account/order notices from phishing. This false negative is direct evidence that a perfect score on the four-message test partition does not ensure the classifier handles a different challenging message.

## Duplicate, leakage, and overfitting checks

The program compares normalized email text using Python's `SequenceMatcher` and reports pairs with similarity at least 0.85, including whether a pair crosses the train/test boundary. On the included dataset and fixed split, the run found **no near-duplicate pairs** and **no cross-split near-duplicate pairs** at that threshold. This check found no obvious duplicate-text leakage, but it does not rule out all forms of leakage or similarity; the threshold is a screening choice, and semantically similar messages can score lower.

No preprocessing leakage was identified in the implemented flow: the split is made before fitting the pipeline, and TF-IDF vocabulary/IDF statistics are learned from training text only. Per-message engineered features use fixed patterns and no labels or corpus-level statistics. The challenge examples are predicted after fitting and are excluded from test metrics.

Measured training accuracy on this split was 1.000, matching test accuracy of 1.000. This absence of a train/test gap does not rule out overfitting: only 12 synthetic training messages and 4 test messages are available. One error would change test accuracy by 25 percentage points, and the challenge false negative shows a weakness not reflected in the test score. The model can fit the deliberately simple wording without learning robust real-world spam patterns.

## Before trusting this filter on real email

Use a substantially larger, diverse, representative dataset with independently verified labels, including current legitimate mail, varied spam/phishing tactics, languages, senders, and message formats. Evaluate with repeated stratified cross-validation and a separate, newer out-of-sample test set; inspect class-specific errors, calibrate the decision threshold against the costs of false positives and false negatives, and monitor performance as spam tactics change. Review privacy and security requirements before processing real mail. The present score is a classroom demonstration only and is not sufficient to deploy or trust as a real email filter.
