# End-to-End Heart Disease Prediction

Machine learning web app for heart disease prediction with a complete training and deployment pipeline.

## Cardiovascular Risk Screening Engine

An end-to-end machine learning pipeline designed for clinical decision support. This system uses patient clinical metrics to predict coronary artery disease risk, prioritizing high sensitivity (Recall) to identify critical cases early.

## Pipeline Overview

This project implements a robust machine learning architecture that automates the transition from raw data ingestion to an interactive, explainable diagnostic dashboard.

### Key Features

* **Adaptive Model Selection**
  Evaluates Logistic Regression, Random Forest, and XGBoost, automatically selecting the best-performing model based on Recall.

* **Production-Grade Preprocessing**
  Uses `sklearn.ColumnTransformer` for feature scaling and categorical encoding, ensuring consistent transformations during training and inference.

* **Explainable AI (XAI)**
  Integrates SHAP for local prediction explanations, improving interpretability of model decisions.

* **Interactive Dashboard**
  Built with Streamlit for real-time predictions and visualization.

## Tech Stack

* Python
* Scikit-learn
* XGBoost
* Pandas, NumPy
* Matplotlib, SHAP
* Streamlit

## Project Structure

```text
End-to-End-Heart-Disease-Prediction/
├── models/
├── reports/
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Evaluation Metrics

Primary metric:

* Recall (Sensitivity)

Additional metrics:

* Accuracy
* Precision
* F1 Score
* ROC-AUC
