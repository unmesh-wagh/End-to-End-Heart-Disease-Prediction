# End-to-End-Heart-Disease-Prediction
Machine learning web app for heart disease prediction with complete training and deployment pipeline.

# 🩺 Cardiovascular Risk Screening Engine

An end-to-end Machine Learning pipeline designed for clinical decision support. This system uses patient clinical metrics to predict coronary artery disease risk, prioritizing high sensitivity (Recall) to ensure critical cases are identified.

## 🚀 Pipeline Overview
This project implements a robust machine learning architecture that automates the transition from raw data ingestion to an interactive, explainable diagnostic web dashboard.

### Key Features
* **Adaptive Champion Selection**: The pipeline evaluates Logistic Regression, Random Forest, and XGBoost architectures, automatically selecting the model with the best Recall score.
* **Production-Grade Preprocessing**: Uses `sklearn.ColumnTransformer` to handle scaling and categorical encoding, ensuring that inference data is transformed identically to training data, eliminating column drift.
* **Explainable AI (XAI)**: Integrates **SHAP** to provide local, patient-specific explanations, helping clinicians understand *why* the model assigned a specific risk score.

## 🛠 Project Structure
```text
├── models/             # Serialized .pkl artifacts and JSON feature metadata
├── reports/            # Automated performance reports (Confusion Matrix, SHAP plots)
├── app.py              # Streamlit web interface
├── train.py            # Training, cross-validation, and artifact serialization
└── README.md           # This project documentation
```
⚙️ Quick Start
1. Requirements
Install the necessary dependencies:

Bash
pip install -r requirements.txt
2. Training
Generate the production artifacts (models, metadata, and reports):

Bash
python train.py
3. Launching the App
Start the interactive dashboard:

Bash
streamlit run app.py
