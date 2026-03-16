# app.py — Customer Churn Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Load model & pipeline objects ─────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load('final_model.pkl')
    scaler        = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ── Header ─────────────────────────────────────────────────────
st.title("🏦 Customer Churn Predictor")
st.markdown("**Powered by XGBoost** · Predicts whether a digital bank customer is likely to churn.")
st.divider()

# ── Sidebar — input form ───────────────────────────────────────
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Fill in the customer details below.")

credit_score     = st.sidebar.slider("Credit Score",        300, 850, 650)
age              = st.sidebar.slider("Age",                  18,  92,  38)
tenure           = st.sidebar.slider("Tenure (years)",        0,  10,   5)
balance          = st.sidebar.number_input("Account Balance (£)",
                                           min_value=0.0,
                                           max_value=250000.0,
                                           value=75000.0,
                                           step=1000.0)
num_products     = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card      = st.sidebar.selectbox("Has Credit Card?",
                                        [1, 0],
                                        format_func=lambda x: "Yes" if x else "No")
is_active        = st.sidebar.selectbox("Is Active Member?",
                                        [1, 0],
                                        format_func=lambda x: "Yes" if x else "No")
estimated_salary = st.sidebar.number_input("Estimated Salary (£)",
                                           min_value=0.0,
                                           max_value=200000.0,
                                           value=50000.0,
                                           step=1000.0)
geography        = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender           = st.sidebar.selectbox("Gender",
                                        [1, 0],
                                        format_func=lambda x: "Male" if x else "Female")

# ── Feature engineering (must mirror training pipeline exactly) ─
bal_salary_ratio = balance / (estimated_salary + 1)
active_with_card = is_active * has_cr_card
zero_balance     = 1 if balance == 0 else 0
age_group        = pd.cut([age],
                           bins=[0, 30, 45, 60, 100],
                           labels=[0, 1, 2, 3])[0]
age_group        = int(age_group)

geo_france  = 1 if geography == "France"  else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain   = 1 if geography == "Spain"   else 0

# ── Assemble input dataframe ────────────────────────────────────
input_dict = {
    'CreditScore':        credit_score,
    'Gender':             gender,
    'Age':                age,
    'Tenure':             tenure,
    'Balance':            balance,
    'NumOfProducts':      num_products,
    'HasCrCard':          has_cr_card,
    'IsActiveMember':     is_active,
    'EstimatedSalary':    estimated_salary,
    'BalanceSalaryRatio': bal_salary_ratio,
    'ActiveWithCard':     active_with_card,
    'ZeroBalance':        zero_balance,
    'AgeGroup':           age_group,
    'Geography_France':   geo_france,
    'Geography_Germany':  geo_germany,
    'Geography_Spain':    geo_spain,
}

input_df = pd.DataFrame([input_dict])[feature_names]
input_scaled = scaler.transform(input_df)

# ── Prediction ──────────────────────────────────────────────────
churn_proba = model.predict_proba(input_scaled)[0][1]
churn_pred  = int(churn_proba >= 0.5)

# ── Main panel — results ────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{churn_proba*100:.1f}%")

with col2:
    label = "⚠️ Likely to Churn" if churn_pred else "✅ Likely to Stay"
    st.metric("Prediction", label)

with col3:
    risk = ("🔴 High" if churn_proba > 0.7
            else "🟡 Medium" if churn_proba > 0.4
            else "🟢 Low")
    st.metric("Risk Level", risk)

st.divider()

# ── Gauge chart ──────────────────────────────────────────────────
col_gauge, col_factors = st.columns([1, 1])

with col_gauge:
    st.subheader("Churn Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(churn_proba * 100, 1),
        delta = {'reference': 20, 'suffix': '%'},
        title = {'text': "Churn Probability (%)"},
        gauge = {
            'axis':  {'range': [0, 100]},
            'bar':   {'color': "#FF5722"},
            'steps': [
                {'range': [0,  40], 'color': '#E8F5E9'},
                {'range': [40, 70], 'color': '#FFF9C4'},
                {'range': [70,100], 'color': '#FFEBEE'},
            ],
            'threshold': {
                'line':  {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── Key factors panel ────────────────────────────────────────────
with col_factors:
    st.subheader("Key Input Summary")
    summary_data = {
        'Feature':    ['Age', 'Balance', 'Products', 'Active', 'Credit Score'],
        'Value':      [age, f'£{balance:,.0f}', num_products,
                       'Yes' if is_active else 'No', credit_score],
        'Risk Signal': [
            '⚠️ High risk age' if age > 45 else '✅ Normal',
            '⚠️ Zero balance'  if balance == 0 else '✅ Has balance',
            '⚠️ Only 1 product' if num_products == 1 else '✅ Multiple products',
            '⚠️ Inactive'      if not is_active else '✅ Active',
            '⚠️ Low score'     if credit_score < 500 else '✅ Normal',
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

st.divider()

# ── Feature importance bar ────────────────────────────────────────
st.subheader("Model Feature Importances")
importances = model.feature_importances_
feat_imp_df = (pd.DataFrame({'Feature': feature_names, 'Importance': importances})
               .sort_values('Importance', ascending=False)
               .head(10))

fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature',
                 orientation='h', color='Importance',
                 color_continuous_scale='Blues',
                 title='Top 10 Features Driving Predictions')
fig_imp.update_layout(height=380,
                      yaxis={'categoryorder': 'total ascending'},
                      margin=dict(t=40, b=0))
st.plotly_chart(fig_imp, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("ML Assignment · Customer Churn Prediction · XGBoost Classifier · "
           "Trained on Bank Customer Churn Dataset (Kaggle)")
