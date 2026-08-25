import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers

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

    # ---------- Train/Test Split Results ----------

def run_train_test_experiments(X_train, y_train, X_test, y_test):

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
    for name, res in results.items():
            print(f"\n{name}\n{confusion_matrix(y_test, res['y_pred'])}")

    return results, summary_df

    # ---------- Cross-Validation Results ----------
def run_cross_validation_experiments(X, y):

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
        print(f" Starting cross-validation {name} ")
        res = cross_validation_models(pipeline, name, X, y, cv, scoring)
        cv_results.append(res)
        print(f"Finished {name}")

    cv_summary_df = pd.DataFrame(cv_results).set_index('model_name')

    return cv_summary_df


    #---------- Scaling Experiment (KNN) ----------
def run_scaling_experiment(X_train, y_train, X_test, y_test):

    results_scaling_experiment = {}

    results_scaling_experiment['knn(without scaling)'] = train_and_evaluate(
        build_pipeline(KNeighborsClassifier(n_neighbors=5), scale=False),
        'knn(without scaling)', X_train, y_train, X_test, y_test, verbose=False
    )

    results_scaling_experiment['knn(with scaling)'] = train_and_evaluate(
        build_pipeline(KNeighborsClassifier(n_neighbors=5), scale=True),
        'knn(with scaling)', X_train, y_train, X_test, y_test, verbose=False
    )

    scaling_df = pd.DataFrame({
        name: {
            'precision': res['precision'],
            'recall': res['recall'],
            'f1-score': res['f1-score']
        }
        for name, res in results_scaling_experiment.items()
    }).T
    return scaling_df

   # ---------- Decision Tree max_depth Experiment ----------
def run_depth_experiment(X_train, y_train, X_test, y_test):

    depth_results = {}

    for depth in [2, 5, 10, None]:
        depth_results[f'max_depth={depth}'] = train_and_evaluate(
            build_pipeline(DecisionTreeClassifier(max_depth=depth, random_state=42), scale=False),
            f'max_depth={depth}', X_train, y_train, X_test, y_test, verbose=False
        )

    depth_df = pd.DataFrame({
        name: {
            'precision': res['precision'],
            'recall': res['recall'],
            'f1-score': res['f1-score']
        }
        for name, res in depth_results.items()
    }).T
    return depth_df

 # ---------- Classification Threshold  ----------
def run_threshold_experiment(pipeline , X_train, y_train, X_test, y_test, thresholds=[0.3 , 0.5, 0.7]):
    pipeline.fit(X_train , y_train)

    threshold_results={}
    for threshold in thresholds:
        y_prob=pipeline.predict_proba(X_test)[:,1]
        y_pred= (y_prob > threshold).astype(int)

        report_dic= classification_report (y_test , y_pred , output_dict=True)
        threshold_results[f'threshold={threshold}']={
          
        'precision': report_dic['1']['precision'],
        'recall': report_dic['1']['recall'],
        'f1-score': report_dic['1']['f1-score']
        }

    threshold_df=pd.DataFrame(threshold_results).T.round(4)
    return threshold_df
#-----------------save final model -----------------
def save_final_model(X_train, y_train, model, model_path='models/final_model.pkl', scaler_path='models/scaler.pkl'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model.fit(X_train_scaled, y_train)

    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)

    print(f"Scaler saved to {scaler_path}")
    print(f"Model saved to {model_path}")


#----------------- calling function ------------------

if __name__ == "__main__":
    df = load_data('data/creditcard.csv')
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = split_data(df, target_col='Class')
    
    print("\n" + "="*50)
    print("====== Running Train/Test Split Experiments =======")
    results, summary_df = run_train_test_experiments(X_train, y_train, X_test, y_test)
    print(summary_df)

    print("\n" + "="*50)
    print("====== Running Cross-Validation Experiments ======")
    cv_df = run_cross_validation_experiments(X, y)
    print(cv_df)
    
    print("\n" + "="*50)
    print("====== Running Scaling Experiment ======")
    scaling_df = run_scaling_experiment(X_train, y_train, X_test, y_test)
    print(scaling_df)
    
    print("\n" + "="*50)
    print("====== Running Depth Experiment ======")
    depth_df = run_depth_experiment(X_train, y_train, X_test, y_test)
    print(depth_df)
    
    print("\n" + "="*50)
    print("======= Running Threshold Experiment ======")
    print("\n===== Logistic Regression Threshold Analysis =====")
    lr_pipeline = build_pipeline(LogisticRegression(random_state=42, max_iter=1000), scale=True)
    threshold_df_lr = run_threshold_experiment(lr_pipeline, X_train, y_train, X_test, y_test)
    print(threshold_df_lr)
    print("\n===== KNN Threshold Analysis =====")
    knn_pipeline = build_pipeline(KNeighborsClassifier(n_neighbors=5), scale=True)
    threshold_df_knn = run_threshold_experiment(knn_pipeline, X_train, y_train, X_test, y_test)
    print(threshold_df_knn)

    print("\n"+"="*50)
    print("\n======saving final model======")
    save_final_model(X_train ,y_train ,KNeighborsClassifier(n_neighbors=5))
    

   