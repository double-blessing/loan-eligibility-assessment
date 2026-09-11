import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Loan Eligibility Assessment",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# =========================================================
# MODEL
# =========================================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(3, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout = nn.Dropout(0.20)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(self.dropout(x[:, -1, :])).squeeze(1)


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "lstm_uci_model.pt"
    scaler_path = MODEL_DIR / "lstm_uci_scaler.joblib"

    if not model_path.exists():
        return None, None, f"Model not found: {model_path}"

    if not scaler_path.exists():
        return None, None, f"Scaler not found: {scaler_path}"

    model = LSTMClassifier()
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler, None


@st.cache_data
def load_metrics():
    candidates = [
        RESULTS_DIR / "metrics.json",
        RESULTS_DIR / "metrics" / "lstm_metrics.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    return None


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def naira(value):
    return f"₦{value:,.0f}"


def repayment_label(code):
    code = int(code)
    if code <= 0:
        return "Paid on time / no delay"
    if code == 1:
        return "Late by 1 month"
    if code == 2:
        return "Late by 2 months"
    if code == 3:
        return "Late by 3 months"
    if code == 4:
        return "Late by 4 months"
    if code == 5:
        return "Late by 5 months"
    if code == 6:
        return "Late by 6 months"
    if code == 7:
        return "Late by 7 months"
    if code == 8:
        return "Late by 8 months"
    return "Late by 9 months or more"


def make_prediction(model, scaler, records):
    values = records[["Repayment Status", "Bill Amount", "Payment Amount"]].to_numpy(
        dtype=np.float32
    )

    scaled = scaler.transform(values.reshape(1, -1)).reshape(1, 6, 3)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor)).item())

    predicted_default = int(probability >= 0.5)
    return probability, predicted_default


def build_explanation(records, probability):
    repayment = records["Repayment Status"].to_numpy(dtype=int)
    bills = records["Bill Amount"].to_numpy(dtype=float)
    payments = records["Payment Amount"].to_numpy(dtype=float)

    on_time = int(np.sum(repayment <= 0))
    delayed = int(np.sum(repayment > 0))

    latest_bill = bills[-1]
    latest_payment = payments[-1]

    if latest_bill > 0:
        latest_payment_ratio = latest_payment / latest_bill
    else:
        latest_payment_ratio = 1.0

    explanations = []

    if on_time >= 5:
        explanations.append(
            f"Payments were recorded as on time in {on_time} of the 6 observations."
        )
    elif delayed >= 4:
        explanations.append(
            f"Payment delays were recorded in {delayed} of the 6 observations."
        )
    else:
        explanations.append(
            f"There were {delayed} delayed-payment observations and {on_time} on-time observations."
        )

    if latest_bill > 0 and latest_payment_ratio >= 1:
        explanations.append(
            "In the most recent observation, the recorded payment covered the recorded bill amount."
        )
    elif latest_bill > 0 and latest_payment_ratio >= 0.5:
        explanations.append(
            "In the most recent observation, the recorded payment covered a substantial part of the bill amount."
        )
    elif latest_bill > 0:
        explanations.append(
            "In the most recent observation, the recorded payment was much lower than the bill amount."
        )
    else:
        explanations.append("The most recent recorded bill amount was zero.")

    if len(bills) >= 2:
        if bills[-1] < bills[0] * 0.8:
            explanations.append("The recorded bill amount decreased substantially from the earliest observation.")
        elif bills[-1] > bills[0] * 1.2:
            explanations.append("The recorded bill amount increased substantially from the earliest observation.")
        else:
            explanations.append("The recorded bill amount remained within a relatively similar range.")

    if probability < 0.30:
        risk_text = "The model estimates a lower probability of default."
    elif probability < 0.50:
        risk_text = "The model estimates a moderate probability of default."
    elif probability < 0.70:
        risk_text = "The model estimates a relatively high probability of default."
    else:
        risk_text = "The model estimates a high probability of default."

    return explanations, risk_text


def reset_assessment():
    for key in list(st.session_state.keys()):
        if key.startswith("repayment_") or key.startswith("bill_") or key.startswith("payment_"):
            del st.session_state[key]


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Loan Eligibility")
    st.caption("LSTM-based financial behaviour assessment")

    page = st.radio(
        "Go to",
        ["Home", "Loan Assessment", "Model Performance", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "Academic prototype for initial loan-eligibility decision support. "
        "It does not replace a bank's final lending decision."
    )


# =========================================================
# HOME
# =========================================================
if page == "Home":
    st.title("Loan Eligibility Assessment")
    st.subheader("Understand an applicant's repayment risk before making a lending decision.")

    st.write(
        "This system uses a Long Short-Term Memory (LSTM) Recurrent Neural Network "
        "to examine an applicant's financial behaviour across six sequential observations. "
        "It produces a model-based estimate of repayment risk and a simple eligibility recommendation."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "LSTM RNN")

    with col2:
        st.metric("Observations", "6")

    with col3:
        st.metric("Behavioural inputs", "3")

    st.divider()

    st.markdown("### How it works")

    steps = [
        ("1", "Enter financial behaviour", "Provide the applicant's six available financial records, from oldest to newest."),
        ("2", "The system checks the pattern", "The LSTM analyses repayment status, bill amount and payment amount together."),
        ("3", "Risk is estimated", "The model produces an estimated probability of payment default."),
        ("4", "A recommendation is shown", "The result is presented in simple language to support an initial eligibility decision."),
    ]

    for number, title, description in steps:
        st.markdown(f"**{number}. {title}**")
        st.write(description)

    st.info(
        "Important: the underlying public dataset used for this academic project "
        "contains a next-month default outcome rather than a direct bank approval/rejection field. "
        "Therefore, this prototype uses predicted repayment/default risk as a basis for an "
        "initial loan-eligibility recommendation."
    )


# =========================================================
# LOAN ASSESSMENT
# =========================================================
elif page == "Loan Assessment":
    st.title("Loan Eligibility Assessment")
    st.write(
        "Enter the applicant's financial behaviour below. "
        "Start with the oldest available record and finish with the most recent."
    )

    model, scaler, error = load_model()

    if error:
        st.error("The trained model could not be loaded.")
        st.code(error)
        st.stop()

    st.success("The trained LSTM model is ready.")

    st.divider()

    st.markdown("### What information do I need?")

    st.write(
        "For each observation, you need only three pieces of information:"
    )

    info1, info2, info3 = st.columns(3)

    with info1:
        st.markdown("**Payment behaviour**")
        st.caption("Was the payment on time or late?")

    with info2:
        st.markdown("**Amount owed**")
        st.caption("How much was recorded as the bill?")

    with info3:
        st.markdown("**Amount paid**")
        st.caption("How much was actually paid?")

    st.divider()

    st.markdown("### Applicant's financial history")

    st.caption(
        "You do not need to understand the numbers used by the machine-learning model. "
        "Choose the description that best matches the applicant's repayment record."
    )

    repayment_options = {
        "Paid on time / no delay": 0,
        "Late by 1 month": 1,
        "Late by 2 months": 2,
        "Late by 3 months": 3,
        "Late by 4 months": 4,
        "Late by 5 months": 5,
        "Late by 6 months": 6,
        "Late by 7 months": 7,
        "Late by 8 months": 8,
        "Late by 9 months or more": 9,
    }

    rows = []

    for i in range(6):
        observation_name = "Oldest record" if i == 0 else (
            "Most recent record" if i == 5 else f"Record {i + 1}"
        )

        st.markdown(f"**{i + 1}. {observation_name}**")

        c1, c2, c3 = st.columns([1.35, 1, 1])

        with c1:
            repayment_text = st.selectbox(
                "How was the payment?",
                list(repayment_options.keys()),
                key=f"repayment_{i}",
                help="Choose the option that best describes the applicant's repayment status for this record.",
            )

        with c2:
            bill = st.number_input(
                "Amount owed",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                key=f"bill_{i}",
                help="Enter the bill/outstanding amount recorded for this observation.",
            )

        with c3:
            payment = st.number_input(
                "Amount paid",
                min_value=0.0,
                value=5000.0,
                step=1000.0,
                key=f"payment_{i}",
                help="Enter the amount the applicant paid for this observation.",
            )

        rows.append(
            {
                "Observation": i + 1,
                "Repayment Status": repayment_options[repayment_text],
                "Bill Amount": bill,
                "Payment Amount": payment,
            }
        )

    records = pd.DataFrame(rows)

    st.divider()

    with st.expander("What do these terms mean?"):
        st.markdown(
            """
            **Amount owed:** the amount recorded as the applicant's bill or outstanding balance.

            **Amount paid:** the amount the applicant paid during that observation.

            **Payment behaviour:** whether the applicant paid on time or how many months late the payment was.

            **Six observations:** the model needs six sequential records because it was trained using six observations for each person.
            """
        )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        assess = st.button(
            "Assess Loan Eligibility",
            type="primary",
            use_container_width=True,
        )

    with col_b:
        clear = st.button(
            "Clear / Start Again",
            use_container_width=True,
        )

    if clear:
        reset_assessment()
        st.rerun()

    if assess:
        probability, predicted_default = make_prediction(model, scaler, records)

        st.divider()
        st.markdown("## Assessment Result")

        risk_percent = probability * 100

        if predicted_default == 0:
            st.success("### LIKELY ELIGIBLE")
            st.write(
                "Based on the financial behaviour entered, the model estimates a lower "
                "probability of payment default."
            )
        else:
            st.error("### HIGHER RISK — NOT RECOMMENDED")
            st.write(
                "Based on the financial behaviour entered, the model estimates a higher "
                "probability of payment default."
            )

        r1, r2 = st.columns(2)

        with r1:
            st.metric("Estimated default risk", f"{risk_percent:.1f}%")

        with r2:
            st.metric(
                "Assessment",
                "Lower risk" if predicted_default == 0 else "Higher risk",
            )

        st.progress(min(max(probability, 0.0), 1.0))

        st.divider()

        st.markdown("### Why did the system give this result?")

        explanations, risk_text = build_explanation(records, probability)

        st.write(risk_text)

        for item in explanations:
            st.write(f"• {item}")

        st.caption(
            "These explanations summarise the entered financial behaviour for the user. "
            "They are supporting explanations, not a claim that the LSTM independently "
            "identified each item as a causal reason."
        )

        st.divider()

        st.markdown("### Important")

        st.warning(
            "This is an academic decision-support prototype. "
            "The result should be treated as an initial screening recommendation, "
            "not as a final loan approval or rejection."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.write(
        "This page is intended for project evaluation, supervision and academic review."
    )

    metrics = load_metrics()

    if metrics is None:
        st.error("No metrics file was found in the results folder.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")

    with c2:
        st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")

    with c3:
        st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")

    with c4:
        st.metric("F1-score", f"{metrics['f1'] * 100:.2f}%")

    with c5:
        st.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.2f}%")

    st.divider()

    st.markdown("### Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])

    cm_display = pd.DataFrame(
        cm,
        index=["Actual: No Default", "Actual: Default"],
        columns=["Predicted: No Default", "Predicted: Default"],
    )

    st.table(cm_display)

    st.caption(
        "The confusion matrix shows how many test cases were correctly or incorrectly "
        "classified at the model's 0.50 decision threshold."
    )

    st.divider()

    st.markdown("### Model Architecture")

    architecture = pd.DataFrame(
        {
            "Layer": [
                "Input",
                "LSTM Layer 1",
                "LSTM Layer 2",
                "Dropout",
                "Output",
            ],
            "Configuration": [
                "6 observations × 3 features",
                "64 hidden units",
                "32 hidden units",
                "0.20",
                "1 neuron",
            ],
        }
    )

    st.table(architecture)

    st.markdown("### Training Configuration")

    training = pd.DataFrame(
        {
            "Setting": [
                "Optimizer",
                "Learning rate",
                "Loss function",
                "Batch size",
                "Training epochs",
                "Data split",
                "Random seed",
            ],
            "Value": [
                "Adam",
                "0.001",
                "Binary Cross-Entropy with Logits",
                "128",
                "30",
                "80% training / 20% testing",
                "42",
            ],
        }
    )

    st.table(training)


# =========================================================
# ABOUT
# =========================================================
else:
    st.title("About the Project")

    st.markdown("### Prediction of Loan Eligibility Prediction Using RNN")

    st.write(
        "This project implements a Long Short-Term Memory (LSTM) Recurrent Neural Network "
        "for loan-eligibility decision support based on sequential financial behaviour."
    )

    st.markdown("### What the system analyses")

    st.write(
        "The implemented model uses three behavioural inputs across six sequential observations:"
    )

    st.write("• Repayment status")
    st.write("• Bill amount")
    st.write("• Payment amount")

    st.markdown("### What the result means")

    st.write(
        "The LSTM estimates the probability that the applicant will default. "
        "The application then converts that probability into a simple lower-risk or "
        "higher-risk eligibility recommendation."
    )

    st.info(
        "Dataset note: the public dataset used for this implementation contains a "
        "next-month default target rather than a direct loan-approval target. "
        "The application therefore presents the model output as an initial loan-eligibility "
        "decision-support assessment based on repayment risk."
    )

    st.markdown("### Academic disclaimer")

    st.write(
        "The system is an undergraduate research prototype. It is not intended to make "
        "binding financial decisions, and a real lending institution would require additional "
        "applicant information, validation, regulatory controls and human review."
    )