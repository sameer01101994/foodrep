import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

# ── ML / viz ───────────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# ── Streamlit config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="COD Cancellation Dashboard",
                   layout="wide",
                   page_icon="🛵")

# ── Sidebar – file upload ──────────────────────────────────────────────────────
st.sidebar.header("📂 Upload data")
up_file = st.sidebar.file_uploader(
    "Drop a CSV or Excel file containing Zomato orders",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
)
@st.cache_data(show_spinner=False)
def read_data(buffer):
    if buffer.name.endswith(("xlsx", "xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)

if up_file is None:
    st.info("⬅️  Upload your dataset to get started.")
    st.stop()

df_raw = read_data(up_file)

# ── Data filter – COD only ─────────────────────────────────────────────────────
df = df_raw[df_raw["Payment_Mode"].str.contains("cash", case=False, na=False)]

if df.empty:
    st.error("No Cash‑on‑Delivery rows found! Check the column `Payment_Mode`.")
    st.stop()

# ── Data overview ──────────────────────────────────────────────────────────────
st.title("🛵 Zomato COD Cancellations (Last Year)")
st.write(f"Total COD orders in file: **{len(df):,}**")
st.dataframe(df.head())

# ── Trend analysis ‑ cancellations over time ───────────────────────────────────
with st.expander("📈 Time‑series & categorical insights", expanded=True):
    col1, col2 = st.columns(2)

    # 1️⃣ Daily / weekly trend
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    ts_choice = col1.radio(
        "Aggregate by",
        options=["Day", "Week", "Month"],
        index=2,
        horizontal=True,
    )
    freq_map = {"Day": "D", "Week": "W", "Month": "M"}
    trend = (df.groupby(pd.Grouper(key="Order_Date", freq=freq_map[ts_choice]))
               .size()
               .reset_index(name="Cancellations"))
    fig_trend = px.line(trend,
                        x="Order_Date",
                        y="Cancellations",
                        markers=True,
                        title=f"Cancellations per {ts_choice}")
    col1.plotly_chart(fig_trend, use_container_width=True)

    # 2️⃣ Top N cities / reasons
    top_n = col2.slider("Top N", 3, 15, 5, step=1)
    tab_cities, tab_reasons = col2.tabs(["City", "Reason"])
    with tab_cities:
        city_counts = df["City"].value_counts().nlargest(top_n).reset_index()
        fig_city = px.bar(city_counts,
                          x="index",
                          y="City",
                          labels={"index": "City", "City": "Cancellations"},
                          title=f"Top {top_n} Cities")
        st.plotly_chart(fig_city, use_container_width=True)
    with tab_reasons:
        reason_counts = df["Cancellation_Reason"].value_counts().nlargest(top_n).reset_index()
        fig_reason = px.bar(reason_counts,
                            x="index",
                            y="Cancellation_Reason",
                            labels={"index": "Reason", "Cancellation_Reason": "Cancellations"},
                            title=f"Top {top_n} Cancellation Reasons")
        st.plotly_chart(fig_reason, use_container_width=True)

# ── ML section ─────────────────────────────────────────────────────────────────
st.header("🧠 Predictive Models – Will a COD order be cancelled?")

# Define target
target_col = "Cancelled"  # create if missing
if target_col not in df.columns:
    df[target_col] = np.where(df["Cancellation_Reason"].notna(), 1, 0)

# Feature / target split
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify dtypes
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Model choice & hyper‑parameters
model_name = st.sidebar.selectbox(
    "Choose model",
    ("Logistic Regression", "K‑NN", "Decision Tree", "Random Forest", "Gradient Boosting"),
)

metric = st.sidebar.radio("Primary metric", ("Accuracy", "F1", "ROC‑AUC"))

def build_model(name):
    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    elif name == "K‑NN":
        n = st.sidebar.slider("n_neighbors", 3, 25, 7, 1)
        clf = KNeighborsClassifier(n_neighbors=n)
    elif name == "Decision Tree":
        depth = st.sidebar.slider("max_depth", 2, 20, 5, 1)
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    elif name == "Random Forest":
        trees = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
        clf = RandomForestClassifier(n_estimators=trees,
                                     random_state=42,
                                     n_jobs=-1)
    elif name == "Gradient Boosting":
        lr = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
        trees = st.sidebar.slider("n_estimators", 50, 500, 100, 50)
        clf = GradientBoostingClassifier(learning_rate=lr,
                                         n_estimators=trees,
                                         random_state=42)
    return clf

clf = build_model(model_name)

# Preprocess pipeline
numeric_tf = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_tf = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", numeric_tf, num_cols),
    ("cat", categorical_tf, cat_cols),
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf)
])

# Train‑test split
test_size = st.sidebar.slider("Test set %", 10, 40, 20, step=5) / 100
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# Fit & predict
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

# Metrics
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
roc = roc_auc_score(y_test, probs) if probs is not None else np.nan

metric_map = {"Accuracy": acc, "F1": f1, "ROC‑AUC": roc}
st.subheader(f"{model_name} performance")
st.write(
    f"""
* **Accuracy** : {acc:.3f}  
* **Precision**: {prec:.3f}  
* **Recall**   : {rec:.3f}  
* **F1‑Score**: {f1:.3f}  
* **ROC‑AUC**  : {roc:.3f if not np.isnan(roc) else 'N/A'}  
"""
)

# Confusion matrix heat‑map
cm = confusion_matrix(y_test, preds)
fig_cm = go.Figure(
    data=go.Heatmap(z=cm,
                    x=["Pred 0", "Pred 1"],
                    y=["True 0", "True 1"],
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False,
                    hoverinfo="skip"))
fig_cm.update_layout(title="Confusion Matrix")
st.plotly_chart(fig_cm, use_container_width=False)

# ── Download trained model (optional) ──────────────────────────────────────────
import joblib, base64, os, tempfile
with st.expander("⬇️ Download trained model (.pkl)", expanded=False):
    if st.button("Create download link"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            joblib.dump(pipe, f.name)
            b64 = base64.b64encode(open(f.name, "rb").read()).decode()
            href = f'<a href="data:file/output_model.pkl;base64,{b64}" download="{model_name.lower().replace(" ", "_")}_model.pkl">Download</a>'
            st.markdown(href, unsafe_allow_html=True)
            os.remove(f.name)
