
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

METRICS_FILE = OUTPUT_DIR / "metrics.csv"
HISTORY_FILE = OUTPUT_DIR / "cleaned_hourly_data.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"
FORECAST_FILE = OUTPUT_DIR / "next_24_hours_forecast.csv"

# Page configuration
st.set_page_config(
    page_title="AI Energy Forecasting",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ AI-Powered Energy Forecasting")
st.markdown(
    "### Smart energy insights using Machine Learning"
)
st.write(
    "Explore historical household power demand, "
    "model accuracy, and the next 24-hour forecast."
)

st.sidebar.title("Dashboard")
st.sidebar.info("Model: Random Forest Regressor")
st.sidebar.write("Forecast horizon: 24 hours")
st.sidebar.write("Target: Average active power (kW)")

# Check required files
required_files = [
    METRICS_FILE,
    HISTORY_FILE,
    PREDICTIONS_FILE,
    FORECAST_FILE,
]

missing_files = [
    file.name for file in required_files if not file.exists()
]

if missing_files:
    st.error("Some output files are missing: " + ", ".join(missing_files))
    st.info("Run `python main.py` first, then refresh this page.")
    st.stop()

# Load data
metrics = pd.read_csv(METRICS_FILE)

history = pd.read_csv(
    HISTORY_FILE,
    index_col=0,
    parse_dates=True,
)

predictions = pd.read_csv(PREDICTIONS_FILE)

forecast = pd.read_csv(
    FORECAST_FILE,
    index_col=0,
    parse_dates=True,
)

# Convert numeric columns safely
history["power_kw"] = pd.to_numeric(
    history["power_kw"], errors="coerce"
)
history = history.dropna(subset=["power_kw"])

predictions["actual_power_kw"] = pd.to_numeric(
    predictions["actual_power_kw"], errors="coerce"
)
predictions["predicted_power_kw"] = pd.to_numeric(
    predictions["predicted_power_kw"], errors="coerce"
)
predictions = predictions.dropna(
    subset=["actual_power_kw", "predicted_power_kw"]
)

forecast["forecast_power_kw"] = pd.to_numeric(
    forecast["forecast_power_kw"], errors="coerce"
)
forecast = forecast.dropna(subset=["forecast_power_kw"])

# Top summary cards
st.subheader("Model Performance")

metric_values = dict(
    zip(metrics["Metric"], metrics["Value"])
)

mae = float(metric_values["MAE"])
rmse = float(metric_values["RMSE"])
r2 = r2_score(
    predictions["actual_power_kw"],
    predictions["predicted_power_kw"],
)

col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error", f"{mae:.4f} kW")
col2.metric("Root Mean Squared Error", f"{rmse:.4f} kW")
col3.metric("R² Score", f"{r2:.4f}")

st.caption(
    "Metrics are calculated on the held-out test period. "
    "Lower MAE and RMSE indicate smaller prediction errors."
)

# Historical power demand
st.divider()
st.subheader("📈 Historical Power Demand")

days = st.slider(
    "Days of historical data to display",
    min_value=1,
    max_value=30,
    value=7,
)

recent_history = history.tail(days * 24)

st.line_chart(
    recent_history[["power_kw"]],
    y_label="Average power (kW)",
)

# Actual versus predicted
st.divider()
st.subheader("🤖 Actual vs Predicted Power")

comparison = predictions.tail(500).reset_index(drop=True)
comparison.index.name = "Test sample"

st.line_chart(
    comparison[["actual_power_kw", "predicted_power_kw"]],
    y_label="Power (kW)",
)

st.caption(
    "Showing up to the last 500 test samples for readability."
)

# Next 24 hours
st.divider()
st.subheader("🔮 Next 24-Hour Forecast")

st.line_chart(
    forecast[["forecast_power_kw"]],
    y_label="Forecast power (kW)",
)

st.dataframe(
    forecast,
    use_container_width=True,
)

# Download forecast
st.download_button(
    label="⬇️ Download Forecast CSV",
    data=forecast.to_csv(index=True).encode("utf-8"),
    file_name="next_24_hours_forecast.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Built with Python, Pandas, Scikit-learn, and Streamlit. "
    "Forecasts are estimates, not guarantees of future consumption."
)
