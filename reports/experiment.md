# Credit Card Fraud Detection — Project Report

## 1. Dataset Overview
- Source: `creditcard.csv`
- Samples: 284,807 (before removing duplicate rows)
- Features: 30 
- Target: `Class` (0 = Legitimate, 1 = Fraud)
- Missing values: None (a heavily imbalanced dataset)

This imbalance shaped nearly every decision made throughout the project from metric selection to model choice.

## 2. Data Preparation
1. Loaded the dataset and inspected structure, descriptive statistics, and class distribution.
2. Removed duplicate rows.
3. Split into train/test sets (80/20) using stratified 
4. Applied `StandardScaler` via a Pipeline, fit only on training data and applied to test data, avoiding data leakage.

## 3.train test split 

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9991 | 0.8267 | 0.6327 | 0.7168 |
| KNN | 0.9995 | 0.9186 | 0.8061 | 0.8587 |
| Decision Tree (unconstrained) | 0.9991 | 0.7526 | 0.7449 | 0.7487 |

Confusion matrices (format: [[TN, FP], [FN, TP]]):

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 56,851 | 13 | 36 | 62 |
| KNN | 56,857 | 7 | 19 | 79 |
| Decision Tree | 56,840 | 24 | 25 | 73 |

Key observations:
- All three models achieved ~99.9% Accuracy, this number alone is not informative given the class imbalance .
- KNN achieved the best result on every metric that matters for this problem: highest Precision, highest Recall, highest F1-score, fewest false positives (7), and fewest false negatives (19).
- The unconstrained Decision Tree underperformed KNN despite being able to model nonlinear relationships, consistent with the hypothesis that it overfits the training data.

## 4. Cross-Validation (5-Fold Stratified)

| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8702 | 0.6200 | 0.7232 |
| KNN | 0.9374 | 0.7744 | 0.8479 |
| Decision Tree | 0.7449 | 0.7724 | 0.7581 |

Cross-validation confirms the single train/test split was not a lucky outcome , KNN's advantage holds consistently across 5 different folds, and its margin over the other two models is even clearer here than in the single split.

## 5. Additional Experiments

### 5.1 Feature Scaling Impact on KNN

| Scaling | Precision | Recall | F1 |
|---|---:|---:|---:|
| Without Scaling | 1.0000 | 0.0306 | 0.0594 |
| With Scaling | 0.9186 | 0.8061 | 0.8587 |

Without scaling, KNN's Recall collapses to ~3% — it essentially fails to detect fraud at all, because features like `Time` and `Amount` (which have much larger numeric ranges than the PCA-derived `V1`–`V28` features) dominate the distance calculation, drowning out the more informative features. The Precision of 1.0 is misleading here: it comes from catching almost no fraud cases at all (very few positive predictions, and the rare ones happened to be correct). Scaling fixed this dramatically, improving Recall from 3% to over 80%.
### 5.2 Decision Tree — `max_depth` Experiment
| max_depth | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 2 | 0.7653 | 0.7653 | 0.7653 |
| 5 | 0.8941 | 0.7755 | 0.8306 |
| 10 | 0.8902 | 0.7449 | 0.8111 |
| None (unconstrained) | 0.7526 | 0.7449 | 0.7487 |

 The best F1-score was achieved at `max_depth=5`, not with the unconstrained tree. Performance actually degrades as depth increases beyond 5, and the unconstrained tree performs worst of all four options. This is direct evidence of *overfitting*: without a depth limit, the tree grows complex enough to memorize noise in the training data rather than learning patterns that generalize to unseen transactions.

 ## 6.Classification Threshold Experiment
 *Logistic Regression:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.7312 | 0.6939 | 0.7120 |
| 0.5 | 0.8267 | 0.6327 | 0.7168 |
| 0.7 | 0.8310 | 0.6020 | 0.6982 |

KNN:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.8646 | 0.8469 | 0.8557 |
| 0.5 | 0.9186 | 0.8061 | 0.8587 |
| 0.7 | 0.9595 | 0.7245 | 0.8256 |

For both models, lowering the threshold increases Recall at the cost of Precision, and raising it does the opposite. For KNN specifically, threshold = 0.5 (the default) already provides the best F1-score balance, though threshold = 0.3 remains a close and reasonable alternative if maximizing Recall is the priority.


## 6. Final Model Selection
 Selected model: KNN (n_neighbors=5, with feature scaling)
 Selected threshold: 0.5 (default)

| Criterion | Analysis |
|---|---|
| Precision / Recall / F1 | KNN outperformed both other models on every metric: Precision 0.9186, Recall 0.8061, F1 0.8587 — the best balance among all candidates tested. |
| Confusion Matrix | KNN produced the fewest false positives (7) and the fewest false negatives (19) of all three models, meaning it is simultaneously the least disruptive to legitimate customers and the most effective at catching fraud. |
| Cross-validation | KNN's advantage held up across 5-fold cross-validation (mean F1 = 0.8479), confirming the result is stable and not dependent on one particular data split. |
| Class imbalance | Because Accuracy is uninformative on this dataset (99.83% of transactions are legitimate), model selection was based on Precision/Recall/F1 for the fraud class specifically, not Accuracy. |
| Overfitting behavior* | The Decision Tree experiment directly demonstrated overfitting: performance peaked at `max_depth=5` and declined as the tree was allowed to grow deeper, with the unconstrained tree performing worst. KNN, as a non-parametric, instance-based method, does not exhibit this same failure mode and delivered the most consistent results across experiments. || *False Positive / False Negative trade-off | KNN offers the best trade-off available: only 7 false alarms while still catching 79 of 98 fraud cases (FN = 19) — outperforming both alternatives on both sides of this trade-off simultaneously. |
| Problem requirements* | Since missing a fraudulent transaction (False Negative) is generally costlier than a false alarm (False Positive), Recall is prioritized — but not at the total expense of Precision. KNN delivers the highest Recall (0.8061) of all models while also having the highest Precision, making it the clear choice rather than requiring a compromise. |

## 7. Deployment
The final KNN model and scaler were serialized (`models/final_model.pkl`, `models/scaler.pkl`) and wrapped in:- A command-line prediction script (`predict.py`) that reads a transaction from `input.json` and writes a prediction to `output.json`.- A REST API (`api.py`, built with FastAPI) exposing a `POST /predict` endpoint, deployed as a live web service.

## 8. Summary

| Aspect | Decision |
|---|---|
| Final model | KNN (k=5, scaled features) |
| Final threshold | 0.5 |
| Key trade-off | ~19% of fraud cases missed in exchange for a very low false-alarm rate (Precision 0.92) |
| Main limitation | Even the best model does not achieve near-perfect Recall on this imbalanced dataset; further improvement would likely require resampling techniques or cost-sensitive learning approaches beyond simple threshold tuning |