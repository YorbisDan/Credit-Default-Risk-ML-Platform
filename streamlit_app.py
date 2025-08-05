import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from pyspark.sql import SparkSession

# Set page layout
st.set_page_config(layout="wide")

# Initialize Spark session
spark = SparkSession.builder.appName("CreditDefaultApp").getOrCreate()

# Title
st.title("Credit Default Risk Prediction for First-Time Borrowers")

st.markdown("""
This interactive dashboard presents the evaluation of four predictive models used to assess credit default risk.
The models were trained and evaluated using PySpark and additional tools such as XGBoost and ANN.
""")

# Sidebar Upload and Model Selection
st.sidebar.header("Step 1: Upload Your Predictions")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

model_choice = st.sidebar.selectbox("Step 2: Select the Model Used", ["Logistic Regression", "Random Forest", "ANN", "XGBoost"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(f"Uploaded Predictions Data ({model_choice})")
    st.dataframe(df.head())

    # Simulated Target and Feature Selector
    st.sidebar.header("Step 3: ML Feature Selection (Simulated)")
    all_columns = df.columns.tolist()
    target_column = st.sidebar.selectbox("Select Target Column", all_columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns", all_columns, default=[c for c in all_columns if c != target_column])

    # Evaluation Metrics
    st.subheader(f"Performance Metrics for {model_choice}")
    if {'default_flag', 'prediction'}.issubset(df.columns):
        try:
            recall = recall_score(df["default_flag"], df["prediction"])
            f1 = f1_score(df["default_flag"], df["prediction"])
            auc = roc_auc_score(df["default_flag"], df["probability"].apply(lambda x: x if isinstance(x, float) else eval(x)[1]))
        except:
            auc = "N/A"

        metrics_data = {
            "Metric": ["Recall", "F1-Score", "AUC"],
            "Score": [recall, f1, auc]
        }
        st.dataframe(pd.DataFrame(metrics_data).set_index("Metric").style.format("{:.4f}"))

        # Confusion Matrix
        cm = confusion_matrix(df["default_flag"], df["prediction"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.subheader("Confusion Matrix")
        st.pyplot(fig)

        # Download Button
        st.download_button(
            label="Download Processed Predictions",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"{model_choice.replace(' ', '_').lower()}_predictions.csv",
            mime='text/csv'
        )
    else:
        st.warning("The CSV must contain 'default_flag' and 'prediction' columns.")
else:
    st.info("Please upload a CSV file with prediction results.")

# Static Metrics Comparison
st.subheader("Static Comparison of All Models")
data = {
    "Model": ["Logistic", "Random F.", "ANN", "XGBoost"],
    "Recall": [0.6741, 0.0000, 0.0693, 0.0566],
    "F1-Score": [0.4299, 0.0000, 0.1230, 0.1000],
    "AUC": [0.7066, 0.6936, 0.7081, 0.7111]
}
metrics_df = pd.DataFrame(data)

# Three Metric Charts in One Row
col1, col2, col3 = st.columns(3)

with col1:
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    sns.barplot(data=metrics_df, x="Model", y="Recall", palette="pastel", ax=ax1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Recall")
    ax1.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    sns.barplot(data=metrics_df, x="Model", y="F1-Score", palette="pastel", ax=ax2)
    ax2.set_ylim(0, 1)
    ax2.set_title("F1-Score")
    ax2.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
    st.pyplot(fig2)

with col3:
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    sns.barplot(data=metrics_df, x="Model", y="AUC", palette="pastel", ax=ax3)
    ax3.set_ylim(0, 1)
    ax3.set_title("AUC")
    ax3.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
    st.pyplot(fig3)

# Query Builder
st.subheader("Query Builder (Spark SQL)")
query = st.text_area("Enter your Spark SQL query here", height=100)

if st.button("Run Query"):
    if uploaded_file is not None:
        spark_df = spark.createDataFrame(df)
        spark_df.createOrReplaceTempView("borrowers")
        try:
            result_df = spark.sql(query).toPandas()
            st.write("Query Results:")
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Query failed: {e}")
    else:
        st.warning("Please upload a CSV file first.")