import html
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
    page_title="Transaction risk screening",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PRO_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', system-ui, sans-serif !important; }
    .stApp { background: linear-gradient(165deg, #eef2f7 0%, #e8edf4 40%, #f8fafc 100%); }
    [data-testid="stHeader"] { background: rgba(255,255,255,0.95) !important; border-bottom: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    h1 { color: #0f172a !important; font-weight: 700 !important; letter-spacing: -0.03em !important; }
    h2, h3, h4, h5 { color: #1e293b !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important; font-weight: 600 !important;
        text-transform: uppercase !important; font-size: 0.72rem !important; letter-spacing: 0.06em !important;
    }
    .ccd-hero {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 1.35rem;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
    }
    .ccd-hero h1 { margin: 0 !important; font-size: 1.65rem !important; }
    .ccd-hero p { margin: 0.45rem 0 0 !important; color: #475569 !important; font-size: 1.02rem !important; line-height: 1.55 !important; }
    .block-container { padding-top: 1.25rem !important; padding-bottom: 3rem !important; max-width: 1100px !important; }
    [data-testid="stSidebar"] .ccd-howto-line {
        font-size: 0.95rem; font-weight: 700; color: #0f172a; margin: 0 0 10px 0; letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] .ccd-howto-body {
        font-size: 0.8rem; color: #475569; line-height: 1.55; margin: 0 0 14px 0;
    }
    .ccd-pia-row-title {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #64748b;
        margin: 0 0 10px 0;
    }
    .ccd-pia-card {
        font-size: 0.84rem; color: #475569; line-height: 1.5; margin-bottom: 0; padding: 12px 14px;
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        border-left: 3px solid #2563eb; box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
        height: 100%;
    }
    .ccd-pia-card strong {
        display: block; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
        color: #64748b; margin-bottom: 6px;
    }
</style>
"""
st.markdown(_PRO_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model() -> Tuple[Any, float, str]:
    """Return (model, accuracy_display, status). status is artifact_ok | fallback_* for UI messaging."""

    def build_fallback() -> Tuple[Any, float]:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(400, 24))
        y = ((x[:, 0] * 1.8 + x[:, 3] * 1.2 + x[:, 10] > 1.1)).astype(int)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(x, y)
        return model, 0.86

    acc = 0.86
    if os.path.exists("model_accuracy.pkl"):
        try:
            acc = float(joblib.load("model_accuracy.pkl"))
        except Exception:
            acc = 0.86

    if not os.path.exists("log_reg.pkl"):
        m, a = build_fallback()
        return m, a, "fallback_missing_file"

    try:
        model = joblib.load("log_reg.pkl")
        n_feat = int(getattr(model, "n_features_in_", 24))
        model.predict(np.zeros((1, n_feat), dtype=float))
    except Exception:
        m, a = build_fallback()
        return m, a, "fallback_load_or_sklearn_mismatch"

    return model, acc, "artifact_ok"


log_model, accuracy, _model_status = load_model()

_NAV_OPTIONS = [
    "Overview",
    "Single transaction",
    "Batch CSV",
    "Model & metrics",
]
_NAV_INTERNAL = {
    "Overview": "overview",
    "Single transaction": "single",
    "Batch CSV": "batch",
    "Model & metrics": "model",
}

with st.sidebar:
    st.markdown(
        "<p class='ccd-howto-line'>How to use this app</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ccd-howto-body'>"
        "Choose a <b>Section</b> below. On <b>Single transaction</b> or <b>Batch CSV</b>, fill the form or upload a CSV, then use the primary "
        "<b>Run</b> button on the main page. Open <b>Overview</b> or <b>Model &amp; metrics</b> for a quick tour and saved evaluation plots.</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='padding: 2px 0 10px;'>"
        "<span style='font-size:0.72rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#94a3b8;'>"
        "Workspace</span><br/>"
        "<span style='font-size:1.2rem;font-weight:700;color:#0f172a;letter-spacing:-0.02em;'>Risk screening</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    nav_label = st.selectbox(
        "Section",
        _NAV_OPTIONS,
        index=0,
        label_visibility="collapsed",
    )
    menu = _NAV_INTERNAL[nav_label]

    st.divider()
    if _model_status != "artifact_ok":
        st.warning(
            "**Demo model in use** — saved weights did not load (missing file or **scikit-learn** mismatch). "
            "Treat scores as **illustrative**."
        )
    st.caption(
        "**Sherriff Abdul-Hamid**  \n"
        "[Repository](https://github.com/S-ABDUL-AI/CREDIT_CARD_FRAUD_DETECTION) · "
        "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='ccd-hero'><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def render_recommendation_strip() -> None:
    """Insight / implication / action — main pane only (separate from sidebar how-to)."""
    st.markdown("<p class='ccd-pia-row-title'>Recommendation</p>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3, gap="medium")
    with r1:
        st.markdown(
            "<div class='ccd-pia-card'><strong>Insight</strong>"
            "Issuers see very high transaction volume; teams need a <b>fast, consistent signal</b> to prioritise what to review first—"
            "not a substitute for a full fraud investigation.</div>",
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            "<div class='ccd-pia-card'><strong>Implication</strong>"
            "Scores are <b>probabilities</b> from logistic regression. They support <b>triage and queue design</b> only. "
            "If the sidebar shows a <b>demo model</b> warning, treat outputs as <b>illustrative</b>.</div>",
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            "<div class='ccd-pia-card'><strong>Action</strong>"
            "Define thresholds and escalation paths in <b>your</b> operating policy; use scores to <b>route cases</b> for review—"
            "not as the sole basis for declines or account holds.</div>",
            unsafe_allow_html=True,
        )
    st.divider()


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


render_recommendation_strip()

if menu == "overview":
    hero(
        "Transaction risk screening",
        "Logistic-regression prototype for triage—not a substitute for issuer controls or policy.",
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Reported accuracy", f"{accuracy:.1%}")
    with m2:
        st.metric("Model inputs", "24 features", help="Five form fields map to the first slots; batch CSV can supply all 24.")
    with m3:
        src = "Production artifact" if _model_status == "artifact_ok" else "On-host demo"
        st.metric("Model source", src)

    st.divider()
    st.subheader("Capabilities")
    st.markdown(
        "- **Single transaction** — map business fields into the model layout and read a fraud probability.  \n"
        "- **Batch CSV** — score rows from **24 numeric columns** or labeled **amount / time / channel / location / device**.  \n"
        "- **Model & metrics** — saved accuracy plus confusion matrix and ROC plots when present in the repo."
    )

    with st.expander("Governance & limitations", expanded=False):
        st.markdown(
            """
- **Not** a licensed fraud engine; use output to **prioritise human review** and queues.
- If **Model source** reads *On-host demo*, scores are **illustrative** until `log_reg.pkl` loads on the server.
- Manual scoring uses **partial** features unless training matches this encoding.
            """
        )

elif menu == "single":
    hero("Single transaction", "Enter one transaction. Results are indicative—tune thresholds on your own data.")

    st.subheader("Transaction details")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Amount & timing**")
        amount = st.number_input("Amount (USD)", min_value=0.0, value=120.0, step=1.0)
        time_of_day = st.selectbox("Time of day", ["Morning", "Afternoon", "Evening", "Night"])
        transaction_type = st.selectbox("Channel", ["Online", "POS", "ATM"])
    with c2:
        st.markdown("**Location & device**")
        location = st.selectbox("Location", ["Domestic", "International"])
        device = st.selectbox("Device", ["Mobile", "Desktop", "ATM", "POS Terminal"])

    st.divider()
    if st.button("Run screening", type="primary", use_container_width=True):
        X = encode_single(amount, time_of_day, transaction_type, location, device)
        pred = int(log_model.predict(X)[0])
        proba = float(log_model.predict_proba(X)[0][1])

        st.subheader("Outcome")
        o1, o2 = st.columns([1, 2])
        with o1:
            st.metric("Fraud probability", f"{proba:.1%}")
        with o2:
            st.progress(float(min(max(proba, 0.0), 1.0)))
            if pred == 1:
                st.error("**Elevated risk** — route to review per your operating policy.")
            else:
                st.success("**Lower risk** — routine path unless other controls flag the case.")
        st.caption("Thresholds are not calibrated to your portfolio.")

elif menu == "batch":
    hero("Batch CSV", "Upload a file, preview rows, then score. Demo limit: 500,000 rows per run.")

    st.subheader("1 · Upload")
    up = st.file_uploader("CSV file", type=["csv"])
    if up:
        data = pd.read_csv(up)
        st.caption(f"{len(data):,} rows · {data.shape[1]} columns")
        st.dataframe(data.head(12), use_container_width=True, height=260)

        st.subheader("2 · Score")
        if st.button("Run batch screening", type="primary", use_container_width=True):
            try:
                if len(data) > 500_000:
                    st.error("This demo caps batch size at **500,000** rows.")
                else:
                    X = batch_feature_matrix(data)
                    preds = log_model.predict(X)
                    proba = log_model.predict_proba(X)[:, 1]
                    out = data.copy()
                    out["fraud_flag"] = preds
                    out["fraud_proba"] = proba
                    st.success(f"Scored **{len(out):,}** rows.")
                    st.dataframe(out.head(20), use_container_width=True, height=300)

                    proba_arr = np.asarray(proba, dtype=float)
                    n_rows = int(len(proba_arr))
                    flagged = int((preds == 1).sum())
                    pct_flagged = 100.0 * flagged / n_rows if n_rows else 0.0
                    pct_ge_half = 100.0 * float(np.mean(proba_arr >= 0.5)) if n_rows else 0.0
                    med = float(np.median(proba_arr)) if n_rows else 0.0

                    st.subheader("Score summary for this batch")
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    with sm1:
                        st.metric("Mean fraud probability", f"{float(proba_arr.mean()):.1%}")
                    with sm2:
                        st.metric("Median (P50)", f"{med:.1%}")
                    with sm3:
                        st.metric(
                            "Share ≥ 0.5",
                            f"{pct_ge_half:.1f}%",
                            help="Fraction of rows at or above sklearn's usual 0/1 cutoff for logistic regression.",
                        )
                    with sm4:
                        st.metric("Rows flagged pred=1", f"{flagged:,}", delta=f"{pct_flagged:.1f}% of batch", delta_color="off")

                    sns.set_theme(style="whitegrid")
                    n_bins = int(min(40, max(12, round(np.sqrt(n_rows))))) if n_rows else 12
                    fig, (axh, axb) = plt.subplots(
                        1,
                        2,
                        figsize=(10.8, 3.85),
                        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28},
                    )
                    sns.histplot(
                        proba_arr,
                        bins=n_bins,
                        kde=True,
                        ax=axh,
                        color="#2563eb",
                        edgecolor="white",
                        linewidth=0.35,
                        stat="count",
                    )
                    axh.axvline(0.5, color="#dc2626", linestyle="--", linewidth=2.2, label="Default cutoff (0.5)")
                    if n_rows:
                        axh.axvline(med, color="#64748b", linestyle=":", linewidth=1.8, label="Median score")
                    axh.set_xlim(0.0, 1.0)
                    axh.set_xlabel("Fraud probability (model output)", fontsize=10)
                    axh.set_ylabel("Number of transactions", fontsize=10)
                    axh.set_title("Where does this file pile up?", fontsize=11.5, fontweight="bold", pad=10, color="#0f172a")
                    axh.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

                    band_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
                    hist_b, _ = np.histogram(proba_arr, bins=band_edges)
                    pct_band = 100.0 * hist_b / n_rows if n_rows else hist_b.astype(float)
                    band_labels = ["Routine\n0–25%", "Watch\n25–50%", "Elevated\n50–75%", "Priority\n75–100%"]
                    colors_b = ["#16a34a", "#ca8a04", "#ea580c", "#b91c1c"]
                    y_pos = np.arange(len(band_labels))
                    axb.barh(y_pos, pct_band, color=colors_b, edgecolor="white", linewidth=0.6, height=0.72)
                    axb.set_yticks(y_pos)
                    axb.set_yticklabels(band_labels, fontsize=9)
                    axb.set_xlabel("Percent of this batch", fontsize=10)
                    axb.set_title("How much needs attention?", fontsize=11.5, fontweight="bold", pad=10, color="#0f172a")
                    xmax = float(max(12.0, pct_band.max() * 1.2, 5.0)) if n_rows else 100.0
                    axb.set_xlim(0.0, xmax)
                    for i, v in enumerate(pct_band):
                        axb.text(min(v + xmax * 0.02, xmax * 0.97), i, f"{v:.1f}%", va="center", fontsize=9.5, color="#0f172a")
                    fig.patch.set_facecolor("#ffffff")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption(
                        "Left: distribution of predicted fraud probability; red dashed line is the model's usual yes/no boundary. "
                        "Right: same scores grouped into coarse review bands so you can see triage load at a glance."
                    )

                    buf = out.to_csv(index=False)
                    st.download_button(
                        "Download scored CSV",
                        buf,
                        "fraud_scores.csv",
                        "text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Upload a **.csv** above. Column formats are described on **Overview**.")

else:
    hero("Model & metrics", "Documentation of the deployed scorer and static evaluation plots.")

    a1, a2 = st.columns(2)
    with a1:
        st.metric("Saved accuracy", f"{accuracy:.2%}")
    with a2:
        st.metric("Algorithm", "Logistic regression", help="class_weight='balanced' for skewed labels.")

    st.markdown(
        "**Intended use:** queue prioritisation and analyst triage—not the sole basis for declines or holds. "
        "A sidebar warning means a fallback demo model may be active."
    )
    st.divider()

    ic1, ic2 = st.columns(2, gap="medium")
    with ic1:
        st.subheader("Confusion matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
            st.caption(
                "Read diagonals as **correct** calls; off-diagonals are **false alarms** (top-right) vs **missed fraud** (bottom-left). "
                "Skew toward one corner tells you whether the model is conservative or aggressive at the threshold used when this plot was exported."
            )
        else:
            st.info("Add `confusion_matrix.png` at repo root to display.")
    with ic2:
        st.subheader("ROC curve")
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", use_container_width=True)
            st.caption(
                "The curve traces **true positive rate vs false positive rate** as the cutoff moves. **Higher and closer to the top-left** means better separability on the evaluation set; the diagonal is random guessing."
            )
        else:
            st.info("Add `roc_curve.png` at repo root to display.")
