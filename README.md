# 🏦 Customer Churn Prediction System
### A Machine Learning System for Digital Banking — Built, Evaluated & Deployed

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://churn2predictor.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Repo-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/otujacob/churn_predictor)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)
---


## 🚀 Live Demo

> **Try the deployed app here → [https://churn2predictor.streamlit.app/]**
>
> Enter any customer profile and get a real-time churn probability prediction instantly.
---

## 📌 Project Overview

Every day, digital banks silently lose customers — no warning, no goodbye.  
This project builds a complete machine learning system that predicts **which customers are at risk of churning**, before they do.

This was built as a full ML systems assignment covering:
- Problem framing & justification
- Data pipeline & feature engineering
- Model implementation, debugging & tuning
- Experimental evaluation & model selection
- Live deployment on Streamlit Community Cloud
---

## 🏆 Final Results

| Model | Accuracy | F1 Score | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | ~0.79 | ~0.56 | ~0.52 | ~0.77 |
| Random Forest | ~0.85 | ~0.61 | ~0.57 | ~0.86 |
| **XGBoost (Tuned) ✅** | **~0.87** | **~0.63** | **~0.60** | **~0.88** |

> **Winner: XGBoost (Tuned)** — highest ROC-AUC, F1, and cross-validation stability across all 5 evaluation methods.
---

## 🗂️ Project Structure

churn-predictor/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── final_model.pkl         # Trained XGBoost model
├── scaler.pkl              # Fitted StandardScaler
├── feature_names.pkl       # Feature names for pipeline consistency
└── README.md               # About the project & result 
```

---

## ⚙️ ML Pipeline

```
Raw Data (10,000 rows)
    ↓
Drop irrelevant columns (RowNumber, CustomerId, Surname)
    ↓
Encode categoricals (Label Encoding + One-Hot Encoding)
    ↓
Feature Engineering (4 new interaction features)
    ↓
Train / Test Split (80/20 · Stratified)
    ↓
SMOTE Oversampling (training set only)
    ↓
StandardScaler (fit on train · transform both)
    ↓
Model Training → Logistic Regression | Random Forest | XGBoost
    ↓
Hyperparameter Tuning (GridSearchCV · 5-Fold CV)
    ↓
Evaluation (ROC-AUC · F1 · Recall · PR Curves · Cross-Validation)
    ↓
Deployment → Streamlit Community Cloud
```

---

## 🔧 Feature Engineering

Four new interaction features were created to expose behavioural signals:

| Feature | Description |
|---|---|
| `BalanceSalaryRatio` | Balance ÷ Salary — measures relative financial engagement |
| `ActiveWithCard` | IsActiveMember × HasCrCard — combined activity signal |
| `ZeroBalance` | Binary flag for £0 balance — disengagement indicator |
| `AgeGroup` | Age bucketed into 4 life-stage bands (18–30, 31–45, 46–60, 60+) |

---

## 📊 Evaluation Methods

Five complementary methods were used — because accuracy alone lies on imbalanced data:

- ✅ **ROC-AUC** — threshold-independent ranking quality
- ✅ **F1 Score** — precision/recall balance on minority class
- ✅ **Recall** — proportion of actual churners correctly caught
- ✅ **Precision-Recall Curve** — imbalance-aware performance
- ✅ **5-Fold Cross-Validation** — stability across data splits

---

## 🖥️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/otujacob/churn_predictor
cd churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
matplotlib
seaborn
plotly
```

---

## 📚 References

Key papers that informed this work:

- P. P. Singh et al., "Investigating customer churn in banking: a machine learning approach," *Data Science and Management*, 2024.
- A. Manzoor et al., "A review on machine learning methods for customer churn prediction," *IEEE Access*, 2024.
- M. Imani et al., "Comprehensive analysis of Random Forest and XGBoost with SMOTE," *Technologies*, 2025.
- K. Peng et al., "Research on customer churn prediction and model interpretability," *PLOS ONE*, 2023.
- T. Chen & C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.

---

## 👤 Author

**OTU SAMUEL JACOB**  
Product Manager | ML Enthusiast  
[LinkedIn Profile](linkedin.com/in/otu-jacob) · [Streamlit App](https://churn2predictor.streamlit.app/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built as part of a Machine Learning Systems assignment — from problem definition to live deployment.*
