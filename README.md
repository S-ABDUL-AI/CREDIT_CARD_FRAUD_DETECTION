# CREDIT-CARD-FRAUD-DETECTION-
An app that predicts whether a credit card transaction is fraudulent or authentic
A Streamlit-based web application** to detect fraudulent credit card transactions using Logistic Regression. This app allows single transaction prediction, batch prediction using CSV files, and provides model performance visualization.
## 🏠 Features
- 🔮 **Single Prediction**: Enter transaction details manually to check if a transaction is fraudulent.  
- 📂 **Batch Prediction**: Upload a CSV file containing multiple transactions for batch fraud detection.  
- 📊 **Model Info**: View performance metrics, including **Confusion Matrix**, **ROC Curve**, and **Model Accuracy**.  
- 💡 Handles imbalanced datasets with class weighting (`class_weight='balanced'`).  

FILE STRUCTURE 
├── app.py                # Streamlit app
├── log_reg.pkl           # Trained Logistic Regression model
├── model_accuracy.pkl    # Model accuracy
├── confusion_matrix.png  # Confusion matrix image
├── roc_curve.png         # ROC curve image
├── requirements.txt      # Python dependencies
└── README.md             # This file


## Streamlit Community Cloud

Deploy from [share.streamlit.io](https://share.streamlit.io/) with **Main file path** `app.py`.

**“You do not have access to this app”** (redirect to sign-in): the deployment is **private**. In Streamlit Cloud open the app → **Share** or **Settings → Sharing** → set the app to **Public**. Only then will `https://….streamlit.app` work for everyone without logging in.

**App crashes on load** (model error): the saved `log_reg.pkl` must match the Cloud `scikit-learn` version. This repo pins a compatible range; if loading still fails, the app falls back to a demo logistic model so the UI keeps working.

Author:Sherriff Abdul-Hamid
Email: sherriffhamid001@gmail.com
