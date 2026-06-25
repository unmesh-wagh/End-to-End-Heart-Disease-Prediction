import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import shap

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cardiovascular Risk Screener", layout="wide")

# --- CACHED ARTIFACT INGESTION ---


@st.cache_resource
def load_production_artifacts():
    try:
        pipeline = joblib.load('models/heart_disease_model.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        with open('models/feature_metadata.json', 'r') as f:
            metadata = json.load(f)
        return pipeline, feature_cols, metadata
    except FileNotFoundError:
        st.error(
            "⚠️ Run 'train.py' first to generate files in the /models directory.")
        st.stop()


pipeline, feature_cols, metadata = load_production_artifacts()

# --- APP LAYOUT ---
st.title("🩺 Cardiovascular Risk Screening Engine")
st.markdown("---")

# --- MAIN INTERFACE: PATIENT DATA INPUT FORM ---
with st.form("patient_metrics_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Demographics & Vitals")
        age = st.slider("Age (Years)", 29, 80, 50)
        sex = st.selectbox("Biological Sex", options=[
                           (1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
        chol = st.slider("Serum Cholesterol", 120, 500, 200)

    with col2:
        st.markdown("##### Symptom & History Profile")
        cp = st.selectbox("Chest Pain Type", options=[
                          1, 2, 3, 4], format_func=lambda x: metadata["cp"]["category_labels"][str(x)])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
        exang = st.selectbox("Exercise-Induced Angina", options=[0, 1])
        ca = st.slider("Major Vessels Colored", 0, 3, 0)

    with col3:
        st.markdown("##### Electrocardiogram Metrics")
        restecg = st.selectbox("Resting ECG Results", options=[
                               0, 1, 2], format_func=lambda x: metadata["restecg"]["category_labels"][str(x)])
        thalach = st.slider("Max Heart Rate (thalach)", 70, 220, 150)
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 0.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise", options=[1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[
                            3, 6, 7], format_func=lambda x: metadata["thal"]["category_labels"][str(x)])

    submit_diagnostics = st.form_submit_button("Generate Risk Analysis")

# --- REAL-TIME INFERENCE & EXPLAINABLE AI PIPELINE ---
if submit_diagnostics:
    st.markdown("---")

    # 1. Construct raw input matching training columns exactly
    raw_input_dict = {
        'age': age, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'cp': cp, 'restecg': restecg, 'thal': thal, 'sex': sex
    }
    input_df = pd.DataFrame([raw_input_dict])[feature_cols]

    # 2. Execute pipeline (automatically scales and encodes)
    risk_probability = pipeline.predict_proba(input_df)[0][1]
    risk_pct = risk_probability * 100

    col_metrics, col_viz = st.columns([2, 3])

    with col_metrics:
        st.markdown("##### Patient Assessment Summary")
        if risk_pct < 30.0:
            st.success("**Low Screening Risk Profile**")
            st.metric("Probability of Disease",
                      f"{risk_pct:.1f}%", "- Low Risk", delta_color="inverse")
        elif 30.0 <= risk_pct < 70.0:
            st.warning("**Moderate Screening Risk Profile**")
            st.metric("Probability of Disease",
                      f"{risk_pct:.1f}%", "+ Moderate Risk", delta_color="normal")
        else:
            st.error("**Elevated Clinical Screening Risk Profile**")
            st.metric("Probability of Disease",
                      f"{risk_pct:.1f}%", "+ Action Required", delta_color="normal")

    with col_viz:
        st.markdown("##### SHAP Patient Local Explanation")

        # 3. Transform data for SHAP using the embedded preprocessor
        preprocessor = pipeline.named_steps['pre']
        clf_step = pipeline.named_steps['clf']

        input_transformed = preprocessor.transform(input_df)
        X_test_df = pd.DataFrame(input_transformed, columns=feature_cols)

        # 4. Apply appropriate Explainer
        if 'LogisticRegression' in type(clf_step).__name__:
            explainer = shap.LinearExplainer(clf_step, X_test_df)
        else:
            explainer = shap.TreeExplainer(clf_step)

        shap_values = explainer.shap_values(X_test_df)

        # Extract correct array shape for Waterfall
        if isinstance(shap_values, list):
            local_shap_values = shap_values[1][0]
            base_value = explainer.expected_value[1]
        elif shap_values.ndim == 3:
            local_shap_values = shap_values[0, :, 1]
            base_value = explainer.expected_value[1]
        else:
            local_shap_values = shap_values[0]
            base_value = explainer.expected_value

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(
            shap.Explanation(
                values=local_shap_values,
                base_values=base_value,
                data=input_df.iloc[0],  # Show raw patient values on the plot
                feature_names=feature_cols
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
