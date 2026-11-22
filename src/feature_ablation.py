
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
from sklearn.ensemble import GradientBoostingRegressor


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# 3. Feature engineering
df["TotalSF"] = (
    df["TotalBsmtSF"]
    + df["1stFlrSF"]
    + df["2ndFlrSF"]
)

df["TotalBathrooms"] = (
    df["FullBath"]
    + df["HalfBath"] * 0.5
    + df["BsmtFullBath"]
    + df["BsmtHalfBath"] * 0.5
)

porch_features = [
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch"
]

df["TotalPorchSF"] = df[porch_features].sum(axis=1)

df["TotalOutdoorSF"] = (
    df["TotalPorchSF"]
    + df["WoodDeckSF"]
)

df["TotalBsmtFinSF"] = (
    df["BsmtFinSF1"]
    + df["BsmtFinSF2"]
)

df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]

df["HasGarage"] = (
    df["GarageArea"] > 0
).astype(int)

df["HasBasement"] = (
    df["TotalBsmtSF"] > 0
).astype(int)

df["IsRemodeled"] = (
    df["YearRemodAdd"] > df["YearBuilt"]
).astype(int)


# 4. Define engineered features
engineered_features = [
    "TotalSF",
    "TotalBathrooms",
    "TotalPorchSF",
    "TotalOutdoorSF",
    "TotalBsmtFinSF",
    "HouseAge",
    "RemodAge",
    "GarageAge",
    "HasGarage",
    "HasBasement",
    "IsRemodeled"
]


# 5. Prepare X and y
X = df.drop(
    columns=["Id", "SalePrice"]
)

y = df["SalePrice"]

print("Total features:", X.shape[1])


# 6. Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 7. Define evaluation function
def evaluate_features(X_data, experiment_name):

    print(f"\nRunning experiment: {experiment_name}")

    predictions = np.zeros(len(X_data))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_data), start=1
    ):

        X_train = X_data.iloc[train_idx]
        X_val = X_data.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        numeric_features = X_train.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_features = X_train.select_dtypes(
            exclude="number"
        ).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ])

        regressor = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ))
        ])

        model = TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        pred = np.maximum(pred, 0)

        predictions[val_idx] = pred

        fold_metrics.append({
            "Experiment": experiment_name,
            "Fold": fold,
            "MAE": mean_absolute_error(y_val, pred),
            "RMSE": root_mean_squared_error(y_val, pred),
            "R2": r2_score(y_val, pred)
        })

        print(f"Fold {fold} completed")

    pooled_metrics = {
        "Experiment": experiment_name,
        "Feature Count": X_data.shape[1],
        "Pooled MAE": mean_absolute_error(y, predictions),
        "Pooled RMSE": root_mean_squared_error(y, predictions),
        "Pooled R2": r2_score(y, predictions)
    }

    return fold_metrics, pooled_metrics


# 8. Run baseline experiment
all_fold_results = []
all_pooled_results = []

baseline_folds, baseline_pooled = evaluate_features(
    X,
    "All Features"
)

all_fold_results.extend(baseline_folds)
all_pooled_results.append(baseline_pooled)


# 9. Remove one engineered feature at a time
for feature in engineered_features:

    reduced_X = X.drop(columns=[feature])

    fold_results, pooled_results = evaluate_features(
        reduced_X,
        f"Without {feature}"
    )

    all_fold_results.extend(fold_results)
    all_pooled_results.append(pooled_results)


# 10. Create result dataframes
fold_df = pd.DataFrame(all_fold_results)
pooled_df = pd.DataFrame(all_pooled_results)


# 11. Calculate mean fold metrics
mean_fold_df = (
    fold_df
    .groupby("Experiment")
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "R2": "mean"
    })
    .reset_index()
)


# 12. Compare each experiment with baseline
baseline = pooled_df.loc[
    pooled_df["Experiment"] == "All Features"
].iloc[0]

pooled_df["MAE Change"] = (
    pooled_df["Pooled MAE"] - baseline["Pooled MAE"]
)

pooled_df["RMSE Change"] = (
    pooled_df["Pooled RMSE"] - baseline["Pooled RMSE"]
)

pooled_df["R2 Change"] = (
    pooled_df["Pooled R2"] - baseline["Pooled R2"]
)


# 13. Sort by MAE impact
comparison = pooled_df.sort_values(
    by="MAE Change",
    ascending=False
)


# 14. Display results
print("\n========== POOLED ABLATION RESULTS ==========")

print(
    comparison.to_string(index=False)
)

print("\n========== MEAN FOLD RESULTS ==========")

print(
    mean_fold_df.to_string(index=False)
)


# 15. Save reports
fold_df.to_csv(
    REPORT_DIR / "feature_ablation_folds.csv",
    index=False
)

pooled_df.to_csv(
    REPORT_DIR / "feature_ablation_pooled.csv",
    index=False
)

mean_fold_df.to_csv(
    REPORT_DIR / "feature_ablation_mean_folds.csv",
    index=False
)


print("\nFeature ablation completed!")
print("Reports saved successfully.")
