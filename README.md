# Credit Default Risk Prediction for Borrowers with Limited Credit History

This project aims to assess credit default risk using machine learning models applied to large-scale lending data. The models were trained and evaluated in PySpark, and the results are visualized through an interactive Streamlit dashboard for comparative analysis and decision support.

Dataset available on Kaggle: https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv

---

## Features

- Upload prediction results (CSV)
- View model performance metrics: Recall, F1-Score, AUC
- Visual comparison of model performance
- Confusion matrix visualization
- Static model comparison across Logistic Regression, Random Forest, ANN, and XGBoost
---

## File Structure
```
📁 CreditDefaultPrediction/
│
├── streamlit_app.py             # App code
├── final_predictions.csv        # Example prediction file
├── requirements.txt             # Python packages
├── README.md                    # Project description
├── 📁 notebooks/                # Modeling and EDA work
│   └── FinalProject_V3.ipynb
│
└── 📁 data/
    └── RawLoanData_C.csv        # Sample
```

## How to Run

### 1. Install required packages

We recommend using a virtual environment:

```bash

pip install -r requirements.txt




