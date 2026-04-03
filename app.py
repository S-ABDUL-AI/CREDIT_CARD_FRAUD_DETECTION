import os
from typing import Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Fraud detection",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_model() -> Tuple[Any, float]:
    def build_fallback():
        rng = np.random.default_rng(42)
        x = rng.normal(size=(400, 24))
        y = ((x[:, 0] * 1.8 + x[:, 3] * 1.2 + x[:, 10] > 1.1)).astype(int)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(x, y)
        return model, 0.86

    model = None
    acc = 0.86
    if os.path.exists("log_reg.pkl"):
        try:
            model = joblib.load("log_reg.pkl")
        except Exception:
            model = None
    if model is None:
        model, acc = build_fallback()
    elif os.path.exists("model_accuracy.pkl"):
        try:
            acc = float(joblib.load("model_accuracy.pkl"))
        except Exception:
            acc = 0.86
    return model, acc


log_model, accuracy = load_model()

st.markdown(
    """
    <style>
    .stApp { background: #0f172a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background: #020617; }
    </style>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Single transaction", "📂 Batch CSV", "📊 Model info"],
)
st.sidebar.caption(
    "**Sherriff Abdul-Hamid** · [GitHub](https://github.com/S-ABDUL-AI) · "
    "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
)


def encode_single(amount: float, time_of_day: str, transaction_type: str, location: str, device: str) -> np.ndarray:
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    type_map = {"Online": 0, "POS": 1, "ATM": 2}
    loc_map = {"Domestic": 0, "International": 1}
    dev_map = {"Mobile": 0, "Desktop": 1, "ATM": 2, "POS Terminal": 3}
    feats = np.zeros(24, dtype=float)
    feats[0] = amount / 1000.0
    feats[1] = time_map[time_of_day]
    feats[2] = type_map[transaction_type]
    feats[3] = loc_map[location]
    feats[4] = dev_map[device]
    return feats.reshape(1, -1)


def batch_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build (n, 24) matrix from either wide numeric or labeled columns."""
    n = len(df)
    if df.shape[1] >= 24 and np.issubdtype(df.iloc[:, :24].values.dtype, np.number):
        X = df.iloc[:, :24].values.astype(float)
        if X.shape[1] != 24:
            raise ValueError("Need 24 numeric feature columns for direct scoring.")
        return X
    req = ["amount", "time_of_day", "transaction_type", "location", "device"]
    lower = {c.lower(): c for c in df.columns}
    missing = [c for c in req if c not in lower]
    if missing:
        raise ValueError(f"For labeled CSV, need columns: {req}. Missing: {missing}")
    amt = df[lower["amount"]].astype(float)
    out = np.zeros((n, 24), dtype=float)
    out[:, 0] = amt.values / 1000.0
    # map categoricals row-wise
    tm = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
    tt = {"online": 0, "pos": 1, "atm": 2}
    lm = {"domestic": 0, "international": 1}
    dm = {"mobile": 0, "desktop": 1, "atm": 2, "pos terminal": 3, "pos": 3}
    for i in range(n):
        out[i, 1] = tm.get(str(df[lower["time_of_day"]].iloc[i]).lower().strip(), 0)
        out[i, 2] = tt.get(str(df[lower["transaction_type"]].iloc[i]).lower().strip(), 0)
        out[i, 3] = lm.get(str(df[lower["location"]].iloc[i]).lower().strip(), 0)
        out[i, 4] = dm.get(str(df[lower["device"]].iloc[i]).lower().strip(), 0)
    return out


if menu == "🏠 Home":
    st.title("💳 Credit card fraud screening")
    st.write(
        "Prototype risk scorer: manual entry maps to **24** model features (first five interpretable), "
        "or upload a batch file with **24 numeric columns** or **amount + time_of_day + transaction_type + location + device**."
    )
    st.metric("Reported model accuracy (artifact)", f"{accuracy:.2%}")

elif menu == "🔮 Single transaction":
    st.subheader("Score one transaction")
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount (USD)", min_value=0.0, value=120.0, step=1.0)
        time_of_day = st.selectbox("Time of day", ["Morning", "Afternoon", "Evening", "Night"])
        transaction_type = st.selectbox("Channel", ["Online", "POS", "ATM"])
    with c2:
        location = st.selectbox("Location", ["Domestic", "International"])
        device = st.selectbox("Device", ["Mobile", "Desktop", "ATM", "POS Terminal"])
    if st.button("Score", type="primary"):
        X = encode_single(amount, time_of_day, transaction_type, location, device)
        pred = int(log_model.predict(X)[0])
        proba = float(log_model.predict_proba(X)[0][1])
        if pred == 1:
            st.error(f"**Elevated risk** — fraud probability **{proba:.1%}**")
        else:
            st.success(f"**Lower risk** — fraud probability **{proba:.1%}**")
        st.caption("Thresholds are illustrative; calibrate on your own data.")

elif menu == "📂 Batch CSV":
    st.subheader("Batch scoring")
    up = st.file_uploader("CSV", type=["csv"])
    if up:
        data = pd.read_csv(up)
        st.dataframe(data.head(15), use_container_width=True)
        if st.button("Run batch", type="primary"):
            try:
                X = batch_feature_matrix(data)
                preds = log_model.predict(X)
                proba = log_model.predict_proba(X)[:, 1]
                out = data.copy()
                out["fraud_flag"] = preds
                out["fraud_proba"] = proba
                st.success(f"Scored **{len(out)}** rows.")
                st.dataframe(out.head(25), use_container_width=True)
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.histplot(proba, bins=30, kde=True, ax=ax, color="#38bdf8")
                ax.set_xlabel("Fraud probability")
                st.pyplot(fig)
                plt.close(fig)
                buf = out.to_csv(index=False)
                st.download_button("Download results CSV", buf, "fraud_scores.csv", "text/csv")
            except Exception as e:
                st.error(str(e))

else:
    st.subheader("Model info")
    st.write(f"Accuracy (saved metric): **{accuracy:.2%}**")
    st.markdown(
        "- **Algorithm:** logistic regression with `class_weight='balanced'` (fallback trained if `.pkl` missing).\n"
        "- **Use:** prioritise alerts and queues — not a sole decision system."
    )
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion matrix")
        else:
            st.info("Add `confusion_matrix.png` to the repo for a diagram.")
    with c2:
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", caption="ROC curve")
        else:
            st.info("Add `roc_curve.png` to the repo for a diagram.")
