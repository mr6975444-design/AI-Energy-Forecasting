from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "images"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

TXT_PATH = DATA_DIR / "household_power_consumption.txt"


# ============================================================
# 2. DOWNLOAD DATASET
# ============================================================

def download_dataset():
    if TXT_PATH.exists():
        print("Dataset found locally. Skipping download.")
        return

    print("Dataset file not found.")
    print("Please place household_power_consumption.txt inside the data folder.")
    raise FileNotFoundError(
        "data/household_power_consumption.txt not found."
    )
    print("Downloading UCI electricity dataset...")

    try:
        dataset = fetch_ucirepo(id=235)

        data = dataset.data.features.copy()

        print("Dataset columns:")
        print(data.columns.tolist())

        # Combine Date and Time into one timestamp
        data["timestamp"] = pd.to_datetime(
            data["Date"].astype(str) + " " + data["Time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )

        # Convert power values to numbers
        data["Global_active_power"] = pd.to_numeric(
            data["Global_active_power"],
            errors="coerce"
        )

        # Save only the columns we need
        data[["timestamp", "Global_active_power"]].to_csv(
            TXT_PATH,
            index=False
        )

        print("Dataset downloaded successfully.")

    except Exception as error:
        print("\nDataset download failed.")
        print("Error:", error)
        raise


# ============================================================
# 3. LOAD AND CLEAN DATA
# ============================================================

def load_and_clean_data():

    print("\nLoading dataset...")

    df = pd.read_csv(
        TXT_PATH,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    print(f"Raw rows: {len(df):,}")

    # Combine Date and Time into one timestamp
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    # Convert power values to numbers
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    # Keep required columns
    df = df[
        [
            "timestamp",
            "Global_active_power"
        ]
    ]

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "timestamp",
            "Global_active_power"
        ]
    )

    # Sort chronologically
    df = df.sort_values("timestamp")

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset="timestamp"
    )

    # Set timestamp as index
    df = df.set_index("timestamp")

    print("Resampling to hourly average power...")

    # Convert minute-level data to hourly average power
    hourly_df = (
        df["Global_active_power"]
        .resample("1h")
        .mean()
        .to_frame(name="power_kw")
    )

    # Remove missing hours
    hourly_df = hourly_df.dropna()

    print(f"Hourly rows: {len(hourly_df):,}")

    return hourly_df

    print("\nLoading dataset...")

    df = pd.read_csv(
        TXT_PATH,
        low_memory=False
    )

    print(f"Raw rows: {len(df):,}")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Convert power to numeric
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    # Keep required columns
    df = df[
        [
            "timestamp",
            "Global_active_power"
        ]
    ]

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "timestamp",
            "Global_active_power"
        ]
    )

    # Sort chronologically
    df = df.sort_values("timestamp")

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset="timestamp"
    )

    # Set timestamp as index
    df = df.set_index("timestamp")

    print("Resampling to hourly average power...")

    # Convert 1-minute measurements into hourly averages
    hourly_df = (
        df["Global_active_power"]
        .resample("1h")
        .mean()
        .to_frame(name="power_kw")
    )

    # Remove missing hours
    hourly_df = hourly_df.dropna()

    print(f"Hourly rows: {len(hourly_df):,}")

    return hourly_df


# ============================================================
# 4. CREATE FEATURES
# ============================================================

def create_features(hourly_df):

    print("\nCreating machine learning features...")

    df = hourly_df.copy()

    # Time-based features
    df["hour"] = df.index.hour

    df["day_of_week"] = df.index.dayofweek

    df["month"] = df.index.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Lag features
    df["lag_1"] = (
        df["power_kw"]
        .shift(1)
    )

    df["lag_24"] = (
        df["power_kw"]
        .shift(24)
    )

    df["lag_168"] = (
        df["power_kw"]
        .shift(168)
    )

    # Previous 24-hour rolling average
    df["rolling_mean_24"] = (
        df["power_kw"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    # Remove rows with missing feature values
    df = df.dropna()

    print(f"Feature rows: {len(df):,}")

    return df


# ============================================================
# 5. TRAIN MODEL
# ============================================================

def train_model(feature_df):

    print("\nTraining Random Forest model...")

    features = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24"
    ]

    target = "power_kw"

    X = feature_df[features]
    y = feature_df[target]

    # Chronological 80/20 split
    split_index = int(
        len(feature_df) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return (
        model,
        features,
        X_test,
        y_test,
        predictions,
        split_index
    )


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

def evaluate_model(
    feature_df,
    y_test,
    predictions,
    split_index
):

    print("\nEvaluating model...")

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Previous-day baseline
    baseline = feature_df["lag_24"].iloc[
        split_index:
    ]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline
    )

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)

    print(f"MAE  : {mae:.4f} kW")
    print(f"RMSE : {rmse:.4f} kW")
    print(f"R²   : {r2:.4f}")

    print("\n" + "=" * 50)
    print("BASELINE PERFORMANCE")
    print("=" * 50)

    print(f"MAE  : {baseline_mae:.4f} kW")
    print(f"RMSE : {baseline_rmse:.4f} kW")
    print(f"R²   : {baseline_r2:.4f}")

    metrics = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "R2",
            "Baseline MAE",
            "Baseline RMSE",
            "Baseline R2"
        ],
        "Value": [
            mae,
            rmse,
            r2,
            baseline_mae,
            baseline_rmse,
            baseline_r2
        ]
    })

    metrics.to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False
    )

    return metrics


# ============================================================
# 7. SAVE PREDICTIONS
# ============================================================

def save_predictions(
    y_test,
    predictions
):

    predictions_df = pd.DataFrame({
        "actual_power_kw": y_test.values,
        "predicted_power_kw": predictions
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    print(
        "\nTest predictions saved to "
        "outputs/test_predictions.csv"
    )


# ============================================================
# 8. SAVE MODEL
# ============================================================

def save_model(model):

    model_path = (
        MODEL_DIR /
        "energy_forecast_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to {model_path}"
    )


# ============================================================
# 9. CREATE VISUALIZATIONS
# ============================================================

def create_visualizations(
    hourly_df,
    y_test,
    predictions
):

    print("\nCreating visualizations...")

    # --------------------------------------------------------
    # Graph 1: Energy consumption trend
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    hourly_df["power_kw"].plot()

    plt.title(
        "Hourly Electricity Power Consumption"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Average Power (kW)"
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR /
        "energy_consumption_trend.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Graph 2: Actual vs predicted
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        y_test.index,
        y_test.values,
        label="Actual"
    )

    plt.plot(
        y_test.index,
        predictions,
        label="Predicted"
    )

    plt.title(
        "Actual vs Predicted Electricity Demand"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Power (kW)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR /
        "actual_vs_predicted.png",
        dpi=150
    )

    plt.close()

    print(
        "Visualizations saved to images/"
    )


# ============================================================
# 10. FORECAST NEXT 24 HOURS
# ============================================================

def forecast_next_24_hours(
    model,
    hourly_df
):

    print(
        "\nGenerating next 24-hour forecast..."
    )

    history = hourly_df[
        "power_kw"
    ].copy()

    forecasts = []

    for _ in range(24):

        next_timestamp = (
            history.index[-1]
            + pd.Timedelta(hours=1)
        )

        hour = next_timestamp.hour

        day_of_week = (
            next_timestamp.dayofweek
        )

        month = (
            next_timestamp.month
        )

        is_weekend = int(
            day_of_week >= 5
        )

        lag_1 = (
            history.iloc[-1]
        )

        lag_24 = (
            history.iloc[-24]
        )

        lag_168 = (
            history.iloc[-168]
        )

        rolling_mean_24 = (
            history.iloc[-24:]
            .mean()
        )

        feature_row = pd.DataFrame({
            "hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "is_weekend": [is_weekend],
            "lag_1": [lag_1],
            "lag_24": [lag_24],
            "lag_168": [lag_168],
            "rolling_mean_24": [
                rolling_mean_24
            ]
        })

        prediction = model.predict(
            feature_row
        )[0]

        history.loc[
            next_timestamp
        ] = prediction

        forecasts.append({
            "timestamp": next_timestamp,
            "forecast_power_kw": prediction
        })

    forecast_df = pd.DataFrame(
        forecasts
    )

    forecast_df.to_csv(
        OUTPUT_DIR /
        "next_24_hours_forecast.csv",
        index=False
    )

    # --------------------------------------------------------
    # Forecast graph
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        forecast_df["timestamp"],
        forecast_df["forecast_power_kw"],
        marker="o"
    )

    plt.title(
        "Next 24 Hours Electricity Demand Forecast"
    )

    plt.xlabel("Time")

    plt.ylabel(
        "Forecast Power (kW)"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR /
        "next_24_hours_forecast.png",
        dpi=150
    )

    plt.close()

    print(
        "Next 24-hour forecast saved to "
        "outputs/next_24_hours_forecast.csv"
    )

    return forecast_df


# ============================================================
# 11. SAVE CLEANED DATA
# ============================================================

def save_cleaned_data(hourly_df):

    output_path = (
        OUTPUT_DIR /
        "cleaned_hourly_data.csv"
    )

    hourly_df.to_csv(
        output_path
    )

    print(
        f"Cleaned data saved to {output_path}"
    )


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print(
        "AI-POWERED ENERGY CONSUMPTION "
        "FORECASTING SYSTEM"
    )
    print("=" * 60)

    # Download dataset
    download_dataset()

    # Load and clean data
    hourly_df = load_and_clean_data()

    # Save cleaned data
    save_cleaned_data(
        hourly_df
    )

    # Create machine learning features
    feature_df = create_features(
        hourly_df
    )

    # Train model
    (
        model,
        features,
        X_test,
        y_test,
        predictions,
        split_index
    ) = train_model(
        feature_df
    )

    # Evaluate model
    evaluate_model(
        feature_df,
        y_test,
        predictions,
        split_index
    )

    # Save predictions
    save_predictions(
        y_test,
        predictions
    )

    # Save model
    save_model(
        model
    )

    # Create graphs
    create_visualizations(
        hourly_df,
        y_test,
        predictions
    )

    # Forecast next 24 hours
    forecast_next_24_hours(
        model,
        hourly_df
    )

    print("\n" + "=" * 60)
    print(
        "PROJECT COMPLETED SUCCESSFULLY!"
    )
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()