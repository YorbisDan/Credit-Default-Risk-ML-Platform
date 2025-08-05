# Credit Default Risk Prediction for First-Time Borrowers

This project aims to assess the credit default risk of first-time loan applicants using machine learning models. The models were trained and evaluated using PySpark, and the results are visualized through an interactive Streamlit dashboard.

---

## 🚀 Features

- Upload prediction results (CSV)
- View model performance metrics: Recall, F1-Score, AUC
- Visual comparison of model performance
- Confusion matrix visualization
- Static model comparison across Logistic Regression, Random Forest, ANN, and XGBoost

---

## 📂 File Structure

📁 CreditDefaultPrediction/
│
├── streamlit_app.py             # ✅ App code
├── final_predictions.csv        # ✅ Example prediction file
├── requirements.txt             # ✅ Python packages
├── README.md                    # ✅ Project description
├── 📁 notebooks/                # Modeling and EDA work
│   └── FinalProject_V3.ipynb
│
└── 📁 data/
    └── RawLoanData_C.csv        # Sample

---

## 📋 How to Run

### 1. Install required packages

We recommend using a virtual environment:

```bash

pip install -r requirements.txt

