import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_prep import load_data, split_data


# ===================================
# scaling with pipeline:
# ===================================
def build_pipeline(model, scale=True):
    steps = []
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    return Pipeline(steps)


def cross_validation_models(pipeline, model_name, X, y, cv, scoring):
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    result = {
        'model_name': model_name,
        'mean_precision': scores['test_precision'].mean(),
        'mean_recall': scores['test_recall'].mean(),
        'mean_f1': scores['test_f1'].mean()
    }
    return result


# ===================================
# train_and_evaluate models :
# ===================================

def train_and_evaluate(pipeline, model_name, X_train, y_train, X_test, y_test, verbose=True):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report_dictionary = classification_report(y_test, y_pred, output_dict=True)
    if verbose:
        print(f"\n------{model_name}------")
        print(f"\nconfusion matrix :{confusion_matrix(y_test, y_pred)}")
        print(f"\nclassification report :{classification_report(y_test, y_pred)}")

    return {
        'model_name': model_name,
        'model': pipeline,
        'y_pred': y_pred,
        'accuracy': report_dictionary['accuracy'],
        'precision': report_dictionary['1']['precision'],
        'recall': report_dictionary['1']['recall'],
        'f1-score': report_dictionary['1']['f1-score']
    }


# ======================================
# calling data
# ======================================

if __name__ == "__main__":
    df = load_data('data/creditcard.csv')
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = split_data(df, target_col='Class')

    # ---------- Train/Test Split Results ----------

    results = {}

    results['LogisticRegression'] = train_and_evaluate(
        build_pipeline(LogisticRegression(random_state=42, max_iter=1000), scale=True),
        'Logistic Regression', X_train, y_train, X_test, y_test, verbose=False
    )

    results['KNeighborsClassifier'] = train_and_evaluate(
        build_pipeline(KNeighborsClassifier(n_neighbors=5), scale=True),
        'Knn', X_train, y_train, X_test, y_test, verbose=False
    )

    results['DecisionTreeClassifier'] = train_and_evaluate(
        build_pipeline(DecisionTreeClassifier(random_state=42), scale=False),
        'Decision Tree', X_train, y_train, X_test, y_test, verbose=False
    )

    summary_df = pd.DataFrame({
        name: {
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1_score': res['f1-score']
        }
        for name, res in results.items()
    }).T
    print("\n===== Train/Test Split Results =====")
    print(summary_df)

    for name, res in results.items():
        print(f"\n{name}\n{confusion_matrix(y_test, res['y_pred'])}")

    # ---------- Cross-Validation Results ----------

    models = {
        'Logistic Regression': build_pipeline(LogisticRegression(random_state=42, max_iter=1000), scale=True),
        'KNN': build_pipeline(KNeighborsClassifier(n_neighbors=5), scale=True),
        'Decision Tree': build_pipeline(DecisionTreeClassifier(random_state=42), scale=False)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
    }

    cv_results = []

    for name, pipeline in models.items():
        print(f" Starting cross-validation ")
        res = cross_validation_models(pipeline, name, X, y, cv, scoring)
        cv_results.append(res)
        print(f"Finished ")

    cv_summary_df = pd.DataFrame(cv_results).set_index('model_name')
    print("\n===== Cross Validation Results =====")
    print(cv_summary_df)

    # ---------- Scaling Experiment (KNN) ----------

    results_scaling_experiment = {}

    results_scaling_experiment['knn(without scaling)'] = train_and_evaluate(
        build_pipeline(KNeighborsClassifier(n_neighbors=5), scale=False),
        'knn(without scaling)', X_train, y_train, X_test, y_test, verbose=False
    )

    results_scaling_experiment['knn(with scaling)'] = train_and_evaluate(
        build_pipeline(KNeighborsClassifier(n_neighbors=5), scale=True),
        'knn(with scaling)', X_train, y_train, X_test, y_test, verbose=False
    )

    scaling_comparison_df = pd.DataFrame({
        name: {
            'precision': res['precision'],
            'recall': res['recall'],
            'f1-score': res['f1-score']
        }
        for name, res in results_scaling_experiment.items()
    }).T
    print("\n===== Scaling Comparison Results =====")
    print(scaling_comparison_df)

   







