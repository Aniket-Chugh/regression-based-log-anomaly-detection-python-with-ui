import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import base64

from data_loader import generate_sample_data, load_csv
from model import train_model, predict, evaluate_model
from detector import detect_anomalies

st.set_page_config(page_title="Cyber Log Anomaly Detector", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🛡️ Log Data Anomaly Detector")
st.markdown("### Cybersecurity Simulation using Regression Analysis")

st.sidebar.header("⚙️ My Project Controls")

data_option = st.sidebar.radio("Select Data Source", ["Sample Data", "Upload CSV"])

scenario_type = "Mixed Threats"
if data_option == "Sample Data":
    scenario_type = st.sidebar.selectbox("Test Scenario", ["Normal Traffic", "Mixed Threats", "DDoS Attack", "System Failure/Drop"])

model_type = st.sidebar.selectbox("Math Model", ["linear", "polynomial"])
degree = 3
if model_type == "polynomial":
    degree = st.sidebar.slider("Polynomial Degree", 2, 5, 3)

threshold = st.sidebar.slider("Anomaly Threshold", 5.0, 50.0, 15.0)
run_button = st.sidebar.button("🚀 Run My Detection Logic")

with st.sidebar.expander("🎓 How My Logic Works"):
    st.write("""
    **Here is the step-by-step logic I used in this project:**

    1. **Data Collection:** I take system logs showing requests over time.
    2. **Regression:** I use a math model to learn what 'normal' behavior looks like.
    3. **Creating a Baseline:** The model predicts the expected activity for any given second.
    4. **Deviation Check:** I measure the gap between real activity and expected activity.
    5. **Threat Flagging:** If the gap crosses the threshold, I flag it as an anomaly!

    *Instead of writing rules for every virus, I just find what's normal and catch everything else.*
    """)


if data_option == "Sample Data":
    df = generate_sample_data(scenario=scenario_type)
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = load_csv(uploaded)

if run_button:
    if 'df' not in locals():
        st.error("Please load data before running detection!")
    else:
        st.markdown("---")
        st.header("Step-by-Step Analysis")

        # Step 1
        with st.spinner("Step 1: Reading network logs..."):
            time.sleep(1)
        st.success("✅ **Step 1 Completed:** I've loaded the network logs. This data shows the timestamp and the amount of activity happening on the server.")

        # Step 2
        X = df["timestamp"].values.reshape(-1, 1)
        y = df["activity"].values
        with st.spinner(f"Step 2: Training {model_type} regression model..."):
            time.sleep(1)
        model, poly = train_model(X, y, model_type=model_type, degree=degree)
        st.success("✅ **Step 2 Completed:** I trained the machine learning model on the data. The model has now 'learned' the normal up-and-down pattern of the system.")

        # Step 3
        with st.spinner("Step 3: Calculating expected baseline..."):
            y_pred = predict(model, X, poly)
            time.sleep(1)
        st.success("✅ **Step 3 Completed:** Using the trained model, I predicted what the activity *should* have been if there were no attacks.")


        with st.spinner("Step 4: Looking for cybersecurity threats..."):
            anomalies, abs_residuals, tags = detect_anomalies(y, y_pred, threshold)
            time.sleep(1)
        st.success(f"✅ **Step 4 Completed:** I compared the real data to the expected baseline. Any data point that deviates by more than {threshold} from normal is flagged as an anomaly!")

        df["predicted"] = y_pred
        df["anomaly"] = anomalies
        df["residual"] = abs_residuals
        df["issue_type"] = tags

        anomaly_count = anomalies.sum()
        metrics = evaluate_model(y, y_pred)

        st.markdown("---")
        st.header("📊 My Interactive Results Dashboard")
        st.write("Here, you can easily see what I explained above. The blue line is real traffic, and the green line is what my model thought was normal. The red dots are the attacks my code caught.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Traffic Pattern vs Expected Baseline**")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(df["timestamp"], df["activity"], label="Actual Data", color='blue', alpha=0.6)
            ax1.plot(df["timestamp"], df["predicted"], label="Expected Baseline", color='green', linewidth=2)
            ax1.scatter(
                df[df["anomaly"]]["timestamp"],
                df[df["anomaly"]]["activity"],
                color="red",
                label="Anomalies Found",
                s=50,
                zorder=5
            )
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Server Requests")
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            st.markdown("**Deviation Analyzer**")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(df["timestamp"], df["residual"], color='orange', alpha=0.7)
            ax2.axhline(y=threshold, color='red', linestyle='--', label=f"Detection Threshold ({threshold})")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("How far from 'normal'")
            ax2.legend()
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("📈 Final Security Report")
        st.write("Here is the final math and summary from the detection run.")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Total Logs Processed", f"{len(df)}")
        metric_col2.metric("Threats Found", f"{anomaly_count}")
        metric_col3.metric("Threat Rate", f"{(anomaly_count/len(df))*100:.2f}%")
        metric_col4.metric("My Model Accuracy (R²)", f"{metrics['r2']:.2f}")

        if anomaly_count > 0:
            st.write("### 🚨 Threat Details")
            st.write("Below is a breakdown of every single attack my logic flagged, classified by whether it was a spike or a drop.")
            detected_df = df[df["anomaly"]][["timestamp", "activity", "predicted", "residual", "issue_type"]]
            st.dataframe(detected_df)


        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📁 Download Security CSV Report",
            data=csv,
            file_name='cyber_anomaly_report_results.csv',
            mime='text/csv',
        )
