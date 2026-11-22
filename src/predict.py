
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CHECK REQUIRED FILES
# --------------------------------------------------

if not TEST_PATH.exists():
    raise FileNotFoundError(
        "test.csv was not found. "
        "Please place it inside data/raw/."
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Final model was not found. "
        "Run python src/train.py first."
    )


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

print("Loading test dataset...")

test_df = pd.read_csv(TEST_PATH)

print(f"Test dataset shape: {test_df.shape}")

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 4. SAVE ORIGINAL IDS
# --------------------------------------------------

if "Id" not in test_df.columns:
    raise ValueError("The test dataset must contain an Id column.")

test_ids = test_df["Id"].copy()


# --------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------

print("\nCreating engineered features...")

test_df["TotalSF"] = (
    test_df["TotalBsmtSF"]
    + test_df["1stFlrSF"]
    + test_df["2ndFlrSF"]
)

test_df["TotalBathrooms"] = (
    test_df["FullBath"]
    + test_df["HalfBath"] * 0.5
    + test_df["BsmtFullBath"]
    + test_df["BsmtHalfBath"] * 0.5
)

porch_features = [
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch"
]

test_df["TotalPorchSF"] = (
    test_df[porch_features].sum(axis=1)
)

test_df["TotalOutdoorSF"] = (
    test_df["TotalPorchSF"]
    + test_df["WoodDeckSF"]
)

test_df["TotalBsmtFinSF"] = (
    test_df["BsmtFinSF1"]
    + test_df["BsmtFinSF2"]
)

test_df["HouseAge"] = (
    test_df["YrSold"]
    - test_df["YearBuilt"]
)

test_df["RemodAge"] = (
    test_df["YrSold"]
    - test_df["YearRemodAdd"]
)

test_df["GarageAge"] = (
    test_df["YrSold"]
    - test_df["GarageYrBlt"]
)

test_df["HasGarage"] = (
    test_df["GarageArea"] > 0
).astype(int)

test_df["HasBasement"] = (
    test_df["TotalBsmtSF"] > 0
).astype(int)

test_df["IsRemodeled"] = (
    test_df["YearRemodAdd"]
    > test_df["YearBuilt"]
).astype(int)


# --------------------------------------------------
# 6. PREPARE FEATURES
# --------------------------------------------------

X_test = test_df.drop(columns=["Id"])

print(f"Prediction feature count: {X_test.shape[1]}")


# --------------------------------------------------
# 7. CHECK FEATURE COMPATIBILITY
# --------------------------------------------------

expected_features = (
    model.regressor_.named_steps[
        "preprocessor"
    ].feature_names_in_
)

missing_features = set(expected_features) - set(X_test.columns)
extra_features = set(X_test.columns) - set(expected_features)

if missing_features:
    raise ValueError(
        f"Missing features: {sorted(missing_features)}"
    )

if extra_features:
    raise ValueError(
        f"Unexpected features: {sorted(extra_features)}"
    )

X_test = X_test[expected_features]


# --------------------------------------------------
# 8. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating house price predictions...")

predictions = model.predict(X_test)

if not np.isfinite(predictions).all():
    raise ValueError(
        "Predictions contain invalid numeric values."
    )

if (predictions < 0).any():
    raise ValueError(
        "Some predicted house prices are negative."
    )


# --------------------------------------------------
# 9. CREATE PREDICTION REPORT
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})

prediction_path = (
    REPORT_DIR / "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# --------------------------------------------------
# 10. CREATE KAGGLE SUBMISSION
# --------------------------------------------------

submission_path = (
    REPORT_DIR / "submission.csv"
)

prediction_df.to_csv(
    submission_path,
    index=False
)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("PREDICTION SUMMARY")
print("=" * 55)

print(f"Number of predictions: {len(predictions)}")
print(f"Mean predicted price: ${predictions.mean():,.2f}")
print(f"Median predicted price: ${np.median(predictions):,.2f}")
print(f"Minimum predicted price: ${predictions.min():,.2f}")
print(f"Maximum predicted price: ${predictions.max():,.2f}")

print("\nFirst 10 predictions:")

print(
    prediction_df.head(10).to_string(index=False)
)

print("\nFiles saved successfully:")
print(f"Predictions: {prediction_path}")
print(f"Kaggle submission: {submission_path}")

print("\nPrediction process completed!")
