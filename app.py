"""
Streamlit dashboard for Telco churn scoring.

Run from the project root:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
GRAPHS_DIR = ROOT / "graphs"
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

FEATURE_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sample_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def score_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.copy()
    if "customerID" in X.columns:
        ids = X["customerID"]
        X = X.drop(columns=["customerID"])
    else:
        ids = None
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"])

    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    X["SeniorCitizen"] = X["SeniorCitizen"].astype(int)
    X = X[FEATURE_ORDER]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= st.session_state.get("threshold", 0.5)).astype(int)

    out = pd.DataFrame({
        "churn_probability": proba.round(4),
        "prediction": ["Churn" if p else "Stay" for p in pred],
        "risk": pd.cut(
            proba,
            bins=[-0.01, 0.35, 0.60, 1.01],
            labels=["Low", "Medium", "High"],
        ),
    })
    if ids is not None:
        out.insert(0, "customerID", ids.values)
    return out


def page_overview():
    st.title("Customer churn dashboard")
    st.caption("Tuned Random Forest pipeline · test ROC-AUC 0.839 · churn recall 0.79")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test ROC-AUC", "0.839")
    c2.metric("Churn recall", "0.79")
    c3.metric("Churn precision", "0.50")
    c4.metric("Churn rate (data)", "26.5%")

    st.markdown(
        "The model is biased toward **catching churners** (`class_weight='balanced'`). "
        "Use the scoring pages to rank customers; change the decision threshold if outreach budget is limited."
    )

    tabs = st.tabs(["Class mix", "Tenure & charges", "Confusion matrix", "PR curve", "Feature importance"])
    files = [
        "churn_distribution.png",
        "tenure_monthlycharges_by_churn.png",
        "confusion_matrix.png",
        "precision_recall_curve.png",
        "feature_importance.png",
    ]
    captions = [
        "Churn is the minority class (~27%). Accuracy alone is misleading.",
        "Churners have shorter tenure (median 10 vs 38 months) and higher monthly bills.",
        "Default 0.5 threshold: 296 true churners caught, 78 missed, 293 false alarms.",
        "Move along this curve to trade precision vs recall for your outreach capacity.",
        "Contract, tenure, and InternetService dominate permutation importance.",
    ]
    for tab, fname, caption in zip(tabs, files, captions):
        with tab:
            path = GRAPHS_DIR / fname
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"Run the notebook or `src/evaluate.py` to generate `{fname}`.")
            st.write(caption)


def page_score_one(model):
    st.title("Score a customer")
    st.caption("Enter account details. The saved pipeline handles scaling and one-hot encoding.")

    if model is None:
        st.error("No trained model found at `models/best_model.pkl`. Train first, then reload.")
        return

    threshold = st.slider(
        "Churn decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Predict Churn when probability is at or above this value. Lower = more recall, more false alarms.",
    )
    st.session_state["threshold"] = threshold

    demo, account, services = st.columns(3)

    with demo:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with account:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        default_total = round(monthly * max(tenure, 1), 2)
        total = st.number_input("Total charges ($)", min_value=0.0, value=float(default_total), step=10.0)

    with services:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        if phone == "No":
            multiple = "No phone service"
            st.selectbox("Multiple lines", ["No phone service"], disabled=True)
        else:
            multiple = st.selectbox("Multiple lines", ["No", "Yes"])

        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        if internet == "No":
            addons = {k: "No internet service" for k in INTERNET_ADDONS}
            st.caption("Internet add-ons locked to “No internet service”.")
        else:
            addons = {
                "OnlineSecurity": st.selectbox("Online security", ["No", "Yes"]),
                "OnlineBackup": st.selectbox("Online backup", ["No", "Yes"]),
                "DeviceProtection": st.selectbox("Device protection", ["No", "Yes"]),
                "TechSupport": st.selectbox("Tech support", ["No", "Yes"]),
                "StreamingTV": st.selectbox("Streaming TV", ["No", "Yes"]),
                "StreamingMovies": st.selectbox("Streaming movies", ["No", "Yes"]),
            }

    row = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        **addons,
    }

    if st.button("Predict churn risk", type="primary"):
        scored = score_frame(model, pd.DataFrame([row]))
        prob = float(scored.loc[0, "churn_probability"])
        label = scored.loc[0, "prediction"]
        risk = scored.loc[0, "risk"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Decision", label)
        m3.metric("Risk band", str(risk))

        if label == "Churn":
            st.warning(
                "Flagged as likely to churn at the current threshold. "
                "Contract type, tenure, and internet plan are the strongest drivers in this model."
            )
        else:
            st.success("Below the churn threshold. Still monitor if tenure is low or the contract is month-to-month.")


def page_batch(model):
    st.title("Batch score")
    st.caption("Upload a CSV with the same columns as the Telco dataset (customerID optional; Churn optional).")

    if model is None:
        st.error("No trained model found at `models/best_model.pkl`. Train first, then reload.")
        return

    threshold = st.slider(
        "Churn decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=st.session_state.get("threshold", 0.50),
        step=0.05,
        key="batch_threshold",
    )
    st.session_state["threshold"] = threshold

    sample = load_sample_data()
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is None and sample is not None:
        st.info("No file uploaded — previewing the first 200 rows of the project dataset.")
        frame = sample.head(200).copy()
    elif uploaded is not None:
        frame = pd.read_csv(uploaded)
    else:
        st.warning("Place the Telco CSV in `data/` or upload a file to score.")
        return

    missing = [c for c in FEATURE_ORDER if c not in frame.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    scored = score_frame(model, frame)
    n_flagged = int((scored["prediction"] == "Churn").sum())
    st.metric("Flagged as churn", f"{n_flagged} / {len(scored)}")
    st.dataframe(
        scored.sort_values("churn_probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button("Download scores CSV", csv, "churn_scores.csv", "text/csv")


def main():
    st.set_page_config(page_title="Churn dashboard", page_icon="📉", layout="wide")
    model = load_model()

    page = st.sidebar.radio(
        "Pages",
        ["Overview", "Score a customer", "Batch score"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Model artifact: `models/best_model.pkl`  \n"
        "Charts: `graphs/`"
    )

    if page == "Overview":
        page_overview()
    elif page == "Score a customer":
        page_score_one(model)
    else:
        page_batch(model)


if __name__ == "__main__":
    main()
