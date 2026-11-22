
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


# 2. Load data
df = pd.read_csv(DATA_PATH)

print("Original dataset:", df.shape)


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
    df["TotalPorchSF"] + df["WoodDeckSF"]
)

df["TotalBsmtFinSF"] = (
    df["BsmtFinSF1"] + df["BsmtFinSF2"]
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


# 4. Prepare target and original features
y = df["SalePrice"]

X = df.drop(
    columns=["Id", "SalePrice"]
)


# 5. Define feature groups
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

selected_engineered = [
    "TotalSF",
    "TotalBathrooms",
    "HouseAge"
]

area_components = [
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF"
]


# 6. Create four feature sets
original_features = X.drop(
    columns=engineered_features
)

all_features = X.copy()

selected_features = original_features.join(
    X[selected_engineered]
)

selected_without_components = selected_features.drop(
    columns=area_components
)

datasets = {
    "Original Features": original_features,
    "All Engineered Features": all_features,
    "Selected Features": selected_features,
    "Selected Without Area Components": selected_without_components
}


# 7. Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 8. Result storage
fold_results = []
pooled_results = []


# 9. Evaluate each feature set
for dataset_name, X_data in datasets.items():

    print("\n" + "=" * 60)
    print(dataset_name)
    print("Feature count:", X_data.shape[1])
    print("=" * 60)

    predictions = np.zeros(len(X_data))

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

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ))
        ])

        model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        pred = np.maximum(pred, 0)

        predictions[val_idx] = pred

        fold_results.append({
            "Dataset": dataset_name,
            "Fold": fold,
            "MAE": mean_absolute_error(y_val, pred),
            "RMSE": root_mean_squared_error(y_val, pred),
            "R2": r2_score(y_val, pred)
        })

        print(f"Fold {fold} completed")

    pooled_results.append({
        "Dataset": dataset_name,
        "Feature Count": X_data.shape[1],
        "Pooled MAE": mean_absolute_error(y, predictions),
        "Pooled RMSE": root_mean_squared_error(y, predictions),
        "Pooled R2": r2_score(y, predictions)
    })


# 10. Create result dataframes
fold_df = pd.DataFrame(fold_results)
pooled_df = pd.DataFrame(pooled_results)

mean_fold_df = (
    fold_df
    .groupby("Dataset")
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "R2": "mean"
    })
    .reset_index()
)


# 11. Display results
print("\n========== POOLED RESULTS ==========")
print(
    pooled_df.sort_values(
        "Pooled MAE"
    ).to_string(index=False)
)

print("\n========== MEAN FOLD RESULTS ==========")
print(
    mean_fold_df.to_string(index=False)
)


# 12. Save reports
fold_df.to_csv(
    REPORT_DIR / "feature_selection_folds.csv",
    index=False
)

pooled_df.to_csv(
    REPORT_DIR / "feature_selection_pooled.csv",
    index=False
)

mean_fold_df.to_csv(
    REPORT_DIR / "feature_selection_mean_folds.csv",
    index=False
)


print("\nFeature selection comparison completed!")
print("Reports saved successfully.")
