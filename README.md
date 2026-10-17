# AI-Powered Energy Consumption Forecasting System

## 🚀 Live Demo

**Try the AI-Powered Energy Forecasting Dashboard here:**

[Open Live Dashboard](https://ai-energy-forecasting-yn385rbpndmsjyfee7xsgp.streamlit.app/)

Explore historical electricity demand, model evaluation metrics, actual-versus-predicted values, and the next 24-hour power forecast.

An AI-based machine learning project that forecasts hourly household electricity power demand using historical electricity consumption data.

The project uses the **UCI Individual Household Electric Power Consumption Dataset**, performs time-series feature engineering, trains a Random Forest regression model, evaluates its performance against a simple previous-day baseline, and generates a forecast for the next 24 hours.

## Project Overview

Electricity demand changes throughout the day and across different days of the week. Accurate demand forecasting can help support smarter energy planning and energy management.

This project demonstrates a complete machine learning workflow:

1. Load electricity consumption data
2. Clean and preprocess the data
3. Convert minute-level measurements into hourly averages
4. Create time-based and historical consumption features
5. Train a Random Forest regression model
6. Evaluate the model using MAE, RMSE, and R²
7. Compare the model with a previous-day baseline
8. Generate the next 24-hour electricity demand forecast
9. Save the trained model, predictions, metrics, and visualizations

## Dataset

The project uses:

**UCI Individual Household Electric Power Consumption Dataset**

Dataset source:

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains measurements of household electric power consumption collected at one-minute intervals.

For this project, the `Global_active_power` measurement is aggregated into hourly average power demand.

**Unit:** kW (kilowatts)

Dataset license: **CC BY 4.0**

The dataset is provided by the UCI Machine Learning Repository and is used with appropriate attribution.

## Machine Learning Approach

### Data preprocessing

The original dataset contains minute-level measurements.

The project:

- Handles missing values
- Converts timestamps into a usable datetime format
- Sorts observations chronologically
- Removes duplicate timestamps
- Aggregates minute-level measurements into hourly averages

### Feature Engineering

The model uses the following features:

| Feature | Description |
|---|---|
| `hour` | Hour of the day |
| `day_of_week` | Day of the week |
| `month` | Month |
| `is_weekend` | Whether the day is a weekend |
| `lag_1` | Previous hour's power |
| `lag_24` | Power from the previous day |
| `lag_168` | Power from the previous week |
| `rolling_mean_24` | Previous 24-hour average power |

### Train/Test Split

The data is divided chronologically:

- **80%** training data
- **20%** testing data

A chronological split is used instead of a random split because electricity forecasting is a time-dependent problem.

## Model

The main machine learning model is:

**Random Forest Regressor**

Configuration:

- 100 trees
- Maximum depth: 16
- Minimum samples per leaf: 2
- Random state: 42

The model predicts hourly average household power demand in kW.

## Baseline

The Random Forest model is compared with a simple **previous-day baseline**.

The baseline assumes that the electricity demand for an hour will be similar to the demand at the same hour on the previous day.

This provides a simple reference point for evaluating whether the machine learning model adds predictive value.

## Results

The model was evaluated on the chronological test set.

| Metric | Random Forest | Previous-Day Baseline |
|---|---:|---:|
| MAE | **0.3388 kW** | 0.5366 kW |
| RMSE | **0.4896 kW** | 0.7912 kW |
| R² | **0.5661** | -0.1333 |

The Random Forest model achieved lower MAE and RMSE than the baseline and a higher R² score.

These results indicate that the machine learning model captured useful patterns in the historical electricity consumption data.

## Visualizations

The project generates three visualizations:

### 1. Electricity Consumption Trend

Shows hourly household electricity power consumption over time.

File:

`images/energy_consumption_trend.png`

### 2. Actual vs Predicted

Compares actual electricity demand with the model's predictions on the test dataset.

File:

`images/actual_vs_predicted.png`

### 3. Next 24-Hour Forecast

Shows the predicted household electricity demand for the next 24 hours.

File:

`images/next_24_hours_forecast.png`

## Project Structure

```text
AI-Energy-Forecasting/
│
├── data/
│   └── household_power_consumption.txt
│
├── models/
│   └── energy_forecast_model.joblib
│
├── outputs/
│   ├── cleaned_hourly_data.csv
│   ├── metrics.csv
│   ├── test_predictions.csv
│   └── next_24_hours_forecast.csv
│
├── images/
│   ├── energy_consumption_trend.png
│   ├── actual_vs_predicted.png
│   └── next_24_hours_forecast.png
│
├── docs/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- UCI Machine Learning Repository

## Installation

Clone the repository and open the project folder.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

Make sure the dataset is available in:

```text
data/household_power_consumption.txt
```

Then run:

```bash
python main.py
```

The program will:

- Load the dataset
- Clean and aggregate the data
- Create machine learning features
- Train the Random Forest model
- Evaluate the model
- Save predictions
- Save the trained model
- Generate visualizations
- Generate a 24-hour forecast

## Generated Outputs

After running the project, the following files are created:

### Model

```text
models/energy_forecast_model.joblib
```

### Metrics

```text
outputs/metrics.csv
```

### Test Predictions

```text
outputs/test_predictions.csv
```

### Future Forecast

```text
outputs/next_24_hours_forecast.csv
```

### Cleaned Data

```text
outputs/cleaned_hourly_data.csv
```

## Limitations

This project is a demonstration of electricity demand forecasting using historical household data.

Some limitations include:

- The model uses data from a single household dataset.
- Weather information is not included.
- Real-time electricity prices are not included.
- The model does not account for future changes in household behavior.
- The forecast is based primarily on historical consumption patterns.

Therefore, the model should be considered a machine learning demonstration rather than a production electricity-grid forecasting system.

## Future Improvements

Possible improvements include:

- Add weather information such as temperature and humidity
- Add holiday and calendar information
- Compare Random Forest with XGBoost or other forecasting models
- Add an interactive dashboard
- Build a Flask or FastAPI prediction API
- Add real-time electricity consumption data
- Deploy the model as a cloud application
- Add model monitoring and automatic retraining

## Conclusion

This project demonstrates how machine learning can be applied to historical electricity consumption data to forecast hourly household power demand.

The Random Forest model achieved better performance than the previous-day baseline on the test dataset and produced a 24-hour electricity demand forecast.

The project provides a foundation for developing more advanced smart-city and energy-management forecasting systems.

## Dataset Attribution

Individual Household Electric Power Consumption Dataset.

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Dataset licensed under **CC BY 4.0**.