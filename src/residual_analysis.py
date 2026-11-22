
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "reports"
    / "final_evaluation_predictions.csv"
)

REPORT_DIR = BASE_DIR / "reports"
VISUAL_DIR = BASE_DIR / "visualizations"

REPORT_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)


# 2. Load predictions
df = pd.read_csv(DATA_PATH)

df["Residual"] = (
    df["Actual Price"] - df["Predicted Price"]
)

df["Absolute Error"] = df["Residual"].abs()


# 3. Create price categories
df["Price Group"] = pd.qcut(
    df["Actual Price"],
    q=5,
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)


# 4. Analyze each model
price_group_results = []

for model_name, group in df.groupby("Model"):

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # 4A. Actual vs Predicted
    plt.figure(figsize=(8, 6))

    plt.scatter(
        group["Actual Price"],
        group["Predicted Price"],
        alpha=0.55
    )

    min_value = min(
        group["Actual Price"].min(),
        group["Predicted Price"].min()
    )

    max_value = max(
        group["Actual Price"].max(),
        group["Predicted Price"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / f"{safe_name}_actual_vs_predicted.png",
        dpi=300
    )

    plt.close()


    # 4B. Residual distribution
    plt.figure(figsize=(8, 6))

    plt.hist(
        group["Residual"],
        bins=40,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"{model_name}: Residual Distribution")
    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / f"{safe_name}_residual_distribution.png",
        dpi=300
    )

    plt.close()


    # 4C. Residuals vs Predicted
    plt.figure(figsize=(8, 6))

    plt.scatter(
        group["Predicted Price"],
        group["Residual"],
        alpha=0.55
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title(f"{model_name}: Residuals vs Predicted")
    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / f"{safe_name}_residuals_vs_predicted.png",
        dpi=300
    )

    plt.close()


    # 4D. Error by actual price group
    grouped = (
        group
        .groupby("Price Group", observed=False)
        .agg(
            Samples=("Absolute Error", "size"),
            MAE=("Absolute Error", "mean"),
            Median_Error=("Absolute Error", "median"),
            Mean_Residual=("Residual", "mean")
        )
        .reset_index()
    )

    grouped.insert(
        0,
        "Model",
        model_name
    )

    price_group_results.append(grouped)

    print("\nError by price group:")
    print(grouped.round(2).to_string(index=False))


# 5. Combine price group results
price_group_df = pd.concat(
    price_group_results,
    ignore_index=True
)


# 6. Find largest prediction errors
top_errors = (
    df
    .sort_values(
        by="Absolute Error",
        ascending=False
    )
    .groupby("Model")
    .head(20)
    .copy()
)

top_errors = top_errors[
    [
        "Model",
        "Fold",
        "Actual Price",
        "Predicted Price",
        "Residual",
        "Absolute Error",
        "Price Group"
    ]
]


# 7. Display overall residual summary
residual_summary = (
    df
    .groupby("Model")
    .agg(
        Mean_Residual=("Residual", "mean"),
        Median_Residual=("Residual", "median"),
        Residual_Std=("Residual", "std"),
        Maximum_Error=("Absolute Error", "max")
    )
    .reset_index()
)

print("\n========== RESIDUAL SUMMARY ==========")
print(residual_summary.round(2).to_string(index=False))

print("\n========== LARGEST PREDICTION ERRORS ==========")
print(top_errors.head(20).round(2).to_string(index=False))


# 8. Save reports
price_group_df.to_csv(
    REPORT_DIR / "error_by_price_group.csv",
    index=False
)

top_errors.to_csv(
    REPORT_DIR / "largest_prediction_errors.csv",
    index=False
)

residual_summary.to_csv(
    REPORT_DIR / "residual_summary.csv",
    index=False
)


print("\nResidual analysis completed!")
print("Reports and visualizations saved successfully.")
