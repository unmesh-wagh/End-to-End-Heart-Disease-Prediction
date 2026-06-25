#!/usr/bin/env python3
"""
Clinical Risk Screening: End-to-End Model Training & Pipeline Serialization
Target Output: Serialized model artifacts and diagnostic evaluation reports
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

# ── Feature metadata: clinical direction + valid ranges ──────────────────────
FEATURE_METADATA = {
    "age":      {"label": "Age (years)",                   "min": 29,  "max": 77,  "inverse": False},
    "trestbps": {"label": "Resting Blood Pressure (mmHg)", "min": 94,  "max": 200, "inverse": False},
    "chol":     {"label": "Serum Cholesterol (mg/dl)",     "min": 126, "max": 564, "inverse": False},
    "fbs":      {"label": "Fasting Blood Sugar >120 mg/dl", "min": 0,   "max": 1,   "inverse": False},
    "thalach":  {"label": "Max Heart Rate Achieved (bpm)", "min": 71,  "max": 202, "inverse": True,
                 "note": "Higher thalach = healthier response = LOWER risk."},
    "exang":    {"label": "Exercise-Induced Angina",       "min": 0,   "max": 1,   "inverse": False},
    "oldpeak":  {"label": "ST Depression (exercise vs rest)", "min": 0.0, "max": 6.2, "inverse": False},
    "slope":    {"label": "Slope of Peak Exercise ST",     "min": 1,   "max": 3,   "inverse": False},
    "cp":       {"label": "Chest Pain Type",               "categories": [1, 2, 3, 4],
                 "category_labels": {1: "Typical Angina", 2: "Atypical Angina",
                                     3: "Non-Anginal", 4: "Asymptomatic"}},
    "restecg":  {"label": "Resting ECG Result",            "categories": [0, 1, 2],
                 "category_labels": {0: "Normal", 1: "ST-T Abnormality",
                                     2: "LV Hypertrophy"}},
    "thal":     {"label": "Thalassemia",                   "categories": [3, 6, 7],
                 "category_labels": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}},
    "sex":      {"label": "Sex",                           "categories": [0, 1],
                 "category_labels": {0: "Female", 1: "Male"}},
    "ca":       {"label": "Major Vessels Colored by Fluoroscopy", "min": 0, "max": 3, "inverse": False},
}

CONTINUOUS_COLS = ['age', 'trestbps', 'chol', 'fbs',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca']
CATEGORICAL_COLS = ['cp', 'restecg', 'thal', 'sex']


def setup_environment():
    print("[1/8] Initializing workspace directories...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (10, 6)


def ingest_and_clean_data():
    print("[2/8] Ingesting dataset from UCI Machine Learning Repository...")
    heart_disease = fetch_ucirepo(id=45)

    X_raw = heart_disease.data.features
    y_raw = heart_disease.data.targets
    df = pd.concat([X_raw, y_raw], axis=1)

    target_col = 'num' if 'num' in df.columns else y_raw.columns[0]
    df.rename(columns={target_col: 'target'}, inplace=True)
    df['target'] = (df['target'] > 0).astype(int)

    print("[3/8] Executing deterministic feature imputation & strict type casting...")
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

    # CRITICAL FIX: Cast to integers to prevent Streamlit inference drift (float != int in OrdinalEncoder)
    cols_to_int = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                   'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal']
    for col in cols_to_int:
        df[col] = df[col].astype(int)

    return df


def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), CONTINUOUS_COLS),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
             unknown_value=-1), CATEGORICAL_COLS),
        ],
        remainder='drop'
    )
    return preprocessor


def process_features(df):
    print("[4/8] Processing features with fitted ColumnTransformer...")
    feature_cols = CONTINUOUS_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_baseline_models(X_train, y_train, cv_strategy):
    print("[5/8] Running 5-fold cross-validation across baseline model architectures...")
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    candidates = {
        'Logistic Regression': Pipeline([('pre', build_preprocessor()), ('clf', LogisticRegression(max_iter=2000, random_state=42))]),
        'Random Forest': Pipeline([('pre', build_preprocessor()), ('clf', RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))]),
        'XGBoost': Pipeline([('pre', build_preprocessor()), ('clf', XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1))]),
    }

    results = {}
    for name, pipeline in candidates.items():
        metrics = cross_validate(
            pipeline, X_train, y_train, cv=cv_strategy, scoring=scoring_metrics)
        results[name] = {
            'pipeline': pipeline,
            'mean_recall': metrics['test_recall'].mean(),
            'mean_f1': metrics['test_f1'].mean(),
            'mean_auc': metrics['test_roc_auc'].mean(),
        }
        print(
            f" -> {name:20} | Recall: {results[name]['mean_recall']:.3f} | F1: {results[name]['mean_f1']:.3f}")

    best_name = max(results, key=lambda k: results[k]['mean_recall'])
    print(f"\n → Champion elected by recall: {best_name}")
    return results, best_name


def optimize_champion_model(X_train, y_train, cv_strategy, champion_name):
    print(f"[6/8] Tuning champion '{champion_name}' with GridSearchCV...")
    if 'Random Forest' in champion_name:
        base_pipeline = Pipeline([('pre', build_preprocessor(
        )), ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))])
        param_grid = {'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [
            4, 6, 8, None], 'clf__class_weight': ['balanced', None]}
    elif 'XGBoost' in champion_name:
        base_pipeline = Pipeline([('pre', build_preprocessor()), ('clf', XGBClassifier(
            eval_metric='logloss', random_state=42, n_jobs=-1))])
        param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [
            3, 5, 7], 'clf__scale_pos_weight': [1, 2]}
    else:
        base_pipeline = Pipeline([('pre', build_preprocessor(
        )), ('clf', LogisticRegression(max_iter=2000, random_state=42))])
        param_grid = {'clf__C': [0.01, 0.1, 1, 10],
                      'clf__class_weight': ['balanced', None]}

    tuner = GridSearchCV(estimator=base_pipeline, param_grid=param_grid,
                         cv=cv_strategy, scoring='recall', n_jobs=-1)
    tuner.fit(X_train, y_train)
    print(f" -> Best params: {tuner.best_params_}")
    return tuner.best_estimator_


def generate_production_reports(champion_pipeline, X_test, y_test):
    print("[7/8] Evaluating holdout set and generating diagnostic reports...")
    y_pred = champion_pipeline.predict(X_test)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['Healthy', 'At-Risk'])
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
    plt.title("Holdout Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300)
    plt.close()

    # SHAP Analytics
    clf_step = champion_pipeline.named_steps['clf']
    X_test_transformed = champion_pipeline.named_steps['pre'].transform(X_test)
    transformed_feature_names = CONTINUOUS_COLS + CATEGORICAL_COLS
    X_test_df = pd.DataFrame(
        X_test_transformed, columns=transformed_feature_names)

    # Adaptive explainer logic
    if 'LogisticRegression' in type(clf_step).__name__:
        explainer = shap.LinearExplainer(clf_step, X_test_df)
    else:
        explainer = shap.TreeExplainer(clf_step)

    shap_values_raw = explainer.shap_values(X_test_df)

    if isinstance(shap_values_raw, list):
        shap_to_plot = shap_values_raw[1]
    elif shap_values_raw.ndim == 3:
        shap_to_plot = shap_values_raw[:, :, 1]
    else:
        shap_to_plot = shap_values_raw

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_to_plot, X_test_df, show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def serialize_production_artifacts(champion_pipeline, feature_cols):
    print("[8/8] Serializing artifacts for production...")
    joblib.dump(champion_pipeline, 'models/heart_disease_model.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    with open('models/feature_metadata.json', 'w') as f:
        json.dump(FEATURE_METADATA, f, indent=2)
    print("\nPipeline compilation complete. Production systems operational.")


if __name__ == '__main__':
    setup_environment()
    cleaned_df = ingest_and_clean_data()
    X_train, X_test, y_train, y_test, feature_cols = process_features(
        cleaned_df)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results, champion_name = evaluate_baseline_models(
        X_train, y_train, cv_strategy)
    champion_pipeline = optimize_champion_model(
        X_train, y_train, cv_strategy, champion_name)
    generate_production_reports(champion_pipeline, X_test, y_test)
    serialize_production_artifacts(champion_pipeline, feature_cols)
