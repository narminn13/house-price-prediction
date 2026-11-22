
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

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# 2. Load original dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# 3. Create engineered features
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

df["HouseAge"] = (
    df["YrSold"]
    - df["YearBuilt"]
)

df["RemodAge"] = (
    df["YrSold"]
    - df["YearRemodAdd"]
)

df["GarageAge"] = (
    df["YrSold"]
    - df["GarageYrBlt"]
)

df["HasGarage"] = (
    df["GarageArea"] > 0
).astype(int)

df["HasBasement"] = (
    df["TotalBsmtSF"] > 0
).astype(int)

df["IsRemodeled"] = (
    df["YearRemodAdd"]
    > df["YearBuilt"]
).astype(int)


# 4. Define engineered feature names
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


# 5. Prepare original and engineered datasets
original_X = df.drop(
    columns=["Id", "SalePrice"] + engineered_features,
    errors="ignore"
)

engineered_X = df.drop(
    columns=["Id", "SalePrice"]
)

y = df["SalePrice"]


datasets = {
    "Original Features": original_X,
    "Engineered Features": engineered_X
}


print("\nOriginal feature count:", original_X.shape[1])
print("Engineered feature count:", engineered_X.shape[1])


# 6. Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# 7. Cross-validation setup
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 8. Result storage
fold_results = []
oof_results = []


# 9. Run experiments
for dataset_name, X in datasets.items():

    print(f"\n{'=' * 60}")
    print(dataset_name)
    print("Feature count:", X.shape[1])
    print(f"{'=' * 60}")

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    for model_name, model in models.items():

        print(f"\nModel: {model_name}")

        # Store OOF predictions for each model and dataset
        predictions = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X), start=1
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

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
                ("model", model)
            ])

            transformed_model = TransformedTargetRegressor(
                regressor=pipeline,
                func=np.log1p,
                inverse_func=np.expm1
            )

            transformed_model.fit(
                X_train,
                y_train
            )

            pred = transformed_model.predict(X_val)

            pred = np.maximum(pred, 0)

            predictions[val_idx] = pred

            fold_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Fold": fold,
                "MAE": mean_absolute_error(y_val, pred),
                "RMSE": root_mean_squared_error(y_val, pred),
                "R2": r2_score(y_val, pred)
            })

            print(f"Fold {fold} completed")

        # 10. Pooled OOF metrics
        pooled_mae = mean_absolute_error(
            y, predictions
        )

        pooled_rmse = root_mean_squared_error(
            y, predictions
        )

        pooled_r2 = r2_score(
            y, predictions
        )

        oof_results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Feature Count": X.shape[1],
            "Pooled MAE": pooled_mae,
            "Pooled RMSE": pooled_rmse,
            "Pooled R2": pooled_r2
        })


# 11. Fold results dataframe
fold_df = pd.DataFrame(fold_results)


# 12. Mean fold metrics
summary_df = (
    fold_df
    .groupby(["Dataset", "Model"])
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "R2": "mean"
    })
    .reset_index()
)


# 13. Pooled results dataframe
pooled_df = pd.DataFrame(oof_results)


# 14. Display results
print("\n========== MEAN FOLD RESULTS ==========")
print(summary_df.to_string(index=False))


print("\n========== POOLED OOF RESULTS ==========")
print(pooled_df.to_string(index=False))


# 15. Save results
fold_df.to_csv(
    REPORT_DIR / "feature_comparison_folds.csv",
    index=False
)

summary_df.to_csv(
    REPORT_DIR / "feature_comparison_summary.csv",
    index=False
)

pooled_df.to_csv(
    REPORT_DIR / "feature_comparison_pooled.csv",
    index=False
)


print("\nFeature comparison completed!")
print("Reports saved successfully.")
