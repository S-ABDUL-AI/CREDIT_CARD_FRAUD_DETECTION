# Credit card fraud detection (Streamlit)

[Live app (Streamlit Cloud)](https://p6ebwni5egxm26enox8gyp.streamlit.app/) · [Source on GitHub](https://github.com/S-ABDUL-AI/CREDIT_CARD_FRAUD_DETECTION)

A Streamlit app that **screens** credit-card-style transactions with **logistic regression** (`class_weight='balanced'`). It supports a **single manual score**, **batch CSV scoring**, and a **model info** page with confusion matrix and ROC images when present.

## Features

- **Single prediction** — amount, time of day, channel, location, device (mapped into the model’s feature layout).
- **Batch CSV** — either **24 numeric columns** matching the model, or labeled columns `amount`, `time_of_day`, `transaction_type`, `location`, `device`.
- **Model info** — saved accuracy plus optional `confusion_matrix.png` and `roc_curve.png`.

## File layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI |
| `log_reg.pkl` | Trained logistic regression |
| `model_accuracy.pkl` | Scalar accuracy for display |
| `model_features.pkl` | Present in some setups (optional) |
| `train_model.py` | Training script |
| `requirements.txt` | Dependencies |
| `runtime.txt` | Python version for Cloud (`python-3.12`) |

## Streamlit Community Cloud

Deploy from [share.streamlit.io](https://share.streamlit.io/) with **Main file path** `app.py`.

- **“You do not have access to this app”** (redirect to sign-in): the deployment may be **private**. In Streamlit Cloud: app → **Share** or **Settings → Sharing** → set to **Public** so the `https://….streamlit.app` link works for everyone without logging in.
- **App crashes on load / wrong scores:** `scikit-learn` on the server must be **compatible** with how `log_reg.pkl` was saved. This repo pins a **narrow sklearn range** in `requirements.txt`. If loading or a quick `predict` check fails, the app uses a **built-in demo model** and shows a **sidebar warning** so users know scores are illustrative.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Sherriff Abdul-Hamid** — [GitHub](https://github.com/S-ABDUL-AI) · [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) · sherriffhamid001@gmail.com
