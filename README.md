# Mini Project 01 - Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions. The project is organized into `data/` for the raw dataset, `src/` for the core scripts (data preprocessing, model training, and prediction), `models/` for saved trained models, and `reports/` for experiment logs and results.

To get started, install dependencies with `pip install -r requirements.txt`, then run the scripts in order: `python src/data_prep.py` to clean and prepare the data, `python src/train.py` to train the model, and `python src/predict.py` to generate predictions.

## Results
## Hypothesis Before Modeling:
1. Which model do you expect to perform best for fraud detection? Why?
    I expect the Decision Tree to perform best for fraud detection, because it can capture complex, nonlinear relationships between features — for example, when a combination of feature values together indicates fraud. In contrast, Logistic Regression can only find a linear decision boundary, which may not be flexible enough to capture such patterns.

2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?
    I think Recall is more important here, because failing to detect a fraudulent transaction can be costly for the bank. However, Precision and F1-score should still be considered, since focusing only on Recall could lead to too many false alarms.

3. What do you expect to happen if the model predicts all transactions as legitimate?
    If the model predicts all transactions as legitimate, its Accuracy would be very high (around 99.83%), since the vast majority of transactions are actually legitimate. However, its Recall for the fraud class would be exactly zero, since it would never correctly identify any fraudulent transaction. This shows that Accuracy alone can be misleading for imbalanced datasets, and metrics like Recall, Precision, and F1-score are needed to properly evaluate model performance.

4. Do you expect feature scaling to significantly affect KNN performance?
    Yes, I expect feature scaling to significantly affect KNN performance, because KNN relies directly on distance calculations between data points. If features are not scaled, a feature like Time — which has a much larger numeric range than the other features — would dominate the distance calculation, making the other features (such as V1–V28) almost irrelevant, even if they are actually important for detecting fraud. After scaling, all features contribute more equally to the distance, allowing KNN to make better-informed predictions.

5. Do you expect the Decision Tree to overfit? Why?
    Yes, I expect the Decision Tree to overfit, because in our current code, no constraints were applied to the tree (e.g., max_depth was left unlimited). This means the tree can grow deep enough to memorize noise and specific details in the training data — especially since we only have 492 fraud samples — rather than learning general patterns. As a result, the model may perform very well on training data but poorly on unseen test data. To reduce this risk, hyperparameters such as max_depth or min_samples_leaf could be tuned by testing several values to find the best balance between underfitting and overfitting.

### After Training Analysis
- Was your initial hypothesis correct?
    No, my initial hypothesis was not correct — KNN outperformed the Decision Tree (F1-score: 0.80 vs. 0.71). This is likely because the unconstrained Decision Tree overfit the training data.
- Which model performed best?
  KNN performed best overall, with the highest Precision (0.95), Recall (0.68), and F1-score (0.79) among the three models — followed by Decision Tree and then Logistic Regression. 

- Which metric was most informative?
    F1-score (and Recall) were far more informative than Accuracy, since all three models had nearly identical Accuracy (~99.9%) despite very different Recall and Precision values. This confirms that Accuracy is misleading on this imbalanced dataset.

- How did class imbalance affect the results?
    Class imbalance made it harder for all models to learn patterns for the fraud class, since they saw far fewer fraud examples compared to legitimate ones. This is reflected in the consistently lower Recall scores across all models compared to their near-perfect Accuracy, showing that imbalance directly limited the models ability to correctly identify fraudulent transactions.

- What was the trade-off between False Positives and False Negatives?
    KNN showed the best trade-off, with very few false alarms (3 False Positives) while still catching most fraud cases (30 False Negatives). Decision Tree caught a similar number of frauds but triggered more false alarms (26 False Positives). Logistic Regression had the weakest trade-off, missing the most fraud cases (40 False Negatives) despite having more false alarms than KNN.

## Model Evaluation

- Why can a model achieve very high Accuracy while still being a poor fraud detection system?
    A model can have high accuracy but fail at fraud detection because fraud cases are extremely rare (class imbalance). If the model simply predicts every transaction as "non-fraud," it achieves high accuracy but catches zero actual frauds. Therefore, metrics like recall, precision, and F1-score are far more meaningful than accuracy in such imbalanced scenarios.

## Effect of Scaling:
 - Why is KNN sensitive to scaling?
    KNN is sensitive to feature scaling because it relies directly on distance calculations between data points. When features have very different numeric ranges  such as Time compared to V1–V28 (which are roughly centered around 0)  the feature with the larger range dominates the distance calculation. This causes KNN to effectively ignore the contribution of smaller-scale features, even if they are more informative for distinguishing between classes.
- Why is Decision Tree less sensitive?
    Decision Tree is less sensitive to scaling because it evaluates one feature at a time against a threshold, rather than combining multiple features into a single distance calculation like KNN does. Since the comparison only involves one feature and one threshold, the relative order of values remains unchanged regardless of scaling.

## Hyperparameter Analysis:

- Did overfitting occur?
    YES, overfitting occurred with max_depth=None , When max_depth=None, the tree grows until all leaves are pure (perfectly classifies training data)
    But on test data, performance drops significantly (F1 drops from 0.831 → 0.749)
    The model learned noise and specific patterns from training data that don't generalize
- Which value provides the best balance?
    max_depth=5 provides the best balance. no ovrfiting
    

Status: 🚧 In progress