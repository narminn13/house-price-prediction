
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error
)


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    BASE_DIR
    / "reports"
    / "final_model_predictions.csv"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# 2. Load out-of-fold predictions
df = pd.read_csv(PREDICTIONS_PATH)

required_columns = [
    "Model",
    "Fold",
    "Actual Price",
    "Predicted Price"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in predictions file: {missing_columns}"
    )

print("Prediction records:", len(df))
print("Models:", df["Model"].unique())


# 3. Calculate metrics for each model
overall_results = []

for model_name, group in df.groupby("Model"):

    actual = group["Actual Price"]
    predicted = group["Predicted Price"]

    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    mape = (
        (actual - predicted).abs() / actual
    ).mean() * 100

    median_ae = median_absolute_error(
        actual,
        predicted
    )

    overall_results.append({
        "Model": model_name,
        "Samples": len(group),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE (%)": mape,
        "Median Absolute Error": median_ae
    })


overall_df = pd.DataFrame(overall_results)

overall_df = overall_df.sort_values(
    by="MAE"
)


# 4. Calculate metrics for each fold
fold_results = []

for (model_name, fold), group in df.groupby(
    ["Model", "Fold"]
):

    actual = group["Actual Price"]
    predicted = group["Predicted Price"]

    fold_results.append({
        "Model": model_name,
        "Fold": fold,
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": root_mean_squared_error(
            actual, predicted
        ),
        "R2": r2_score(
            actual, predicted
        )
    })


fold_df = pd.DataFrame(fold_results)


# 5. Calculate error statistics
df["Residual"] = (
    df["Actual Price"] - df["Predicted Price"]
)

df["Absolute Error"] = df["Residual"].abs()

error_results = []

for model_name, group in df.groupby("Model"):

    error_results.append({
        "Model": model_name,
        "Mean Residual": group["Residual"].mean(),
        "Median Residual": group["Residual"].median(),
        "Mean Absolute Error": group["Absolute Error"].mean(),
        "Maximum Absolute Error": group["Absolute Error"].max(),
        "Overpredictions": (
            group["Predicted Price"] > group["Actual Price"]
        ).sum(),
        "Underpredictions": (
            group["Predicted Price"] < group["Actual Price"]
        ).sum()
    })


error_df = pd.DataFrame(error_results)


# 6. Display results
print("\n========== OVERALL EVALUATION ==========")

print(
    overall_df.round(3).to_string(index=False)
)

print("\n========== FOLD-LEVEL EVALUATION ==========")

print(
    fold_df.round(3).to_string(index=False)
)

print("\n========== ERROR ANALYSIS SUMMARY ==========")

print(
    error_df.round(3).to_string(index=False)
)


# 7. Save reports
overall_df.to_csv(
    REPORT_DIR / "final_evaluation_summary.csv",
    index=False
)

fold_df.to_csv(
    REPORT_DIR / "final_evaluation_by_fold.csv",
    index=False
)

error_df.to_csv(
    REPORT_DIR / "final_error_summary.csv",
    index=False
)

df.to_csv(
    REPORT_DIR / "final_evaluation_predictions.csv",
    index=False
)


print("\nFinal evaluation completed!")
print("All evaluation reports saved successfully.")
