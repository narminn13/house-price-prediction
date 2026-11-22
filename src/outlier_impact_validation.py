
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

outlier_ids = [524, 1299]

outlier_mask = df["Id"].isin(outlier_ids)


# 3. Define models
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


# 4. Define preprocessing
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


def create_preprocessor():

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

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


# 5. Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 6. Results storage
fold_results = []
oof_results = []


# 7. Run experiments
for model_name, model in models.items():

    print(f"\n{'=' * 60}")
    print(model_name)
    print(f"{'=' * 60}")

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X), start=1
    ):

        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()

        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()

        train_ids = df.iloc[train_idx]["Id"]

        # Experiment A: Keep all training observations
        model_a = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", clone(model))
        ])

        model_a.fit(X_train, np.log1p(y_train))

        pred_a = np.expm1(model_a.predict(X_val))
        pred_a = np.maximum(pred_a, 0)

        # Experiment B: Remove outliers only from training
        keep_train_mask = ~train_ids.isin(outlier_ids)

        X_train_filtered = X_train.loc[
            keep_train_mask.values
        ]

        y_train_filtered = y_train.loc[
            keep_train_mask.values
        ]

        model_b = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", clone(model))
        ])

        model_b.fit(
            X_train_filtered,
            np.log1p(y_train_filtered)
        )

        pred_b = np.expm1(model_b.predict(X_val))
        pred_b = np.maximum(pred_b, 0)

        # Evaluate both experiments on the SAME validation fold
        for experiment, predictions in [
            ("All Training Data", pred_a),
            ("Outliers Removed From Training", pred_b)
        ]:

            fold_results.append({
                "Model": model_name,
                "Experiment": experiment,
                "Fold": fold,
                "MAE": mean_absolute_error(y_val, predictions),
                "RMSE": root_mean_squared_error(y_val, predictions),
                "R2": r2_score(y_val, predictions)
            })

            for idx, actual, predicted in zip(
                val_idx, y_val, predictions
            ):
                oof_results.append({
                    "Model": model_name,
                    "Experiment": experiment,
                    "Id": df.iloc[idx]["Id"],
                    "Actual": actual,
                    "Predicted": predicted
                })

        print(f"Fold {fold} completed")


# 8. Fold-level results
fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    REPORT_DIR / "outlier_validation_folds.csv",
    index=False
)


# 9. Mean fold metrics
summary_df = (
    fold_df
    .groupby(["Model", "Experiment"])
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "R2": "mean"
    })
    .reset_index()
)

print("\n========== MEAN FOLD RESULTS ==========")
print(summary_df.to_string(index=False))


# 10. Pooled out-of-fold evaluation
oof_df = pd.DataFrame(oof_results)

pooled_results = []

for (model_name, experiment), group in oof_df.groupby(
    ["Model", "Experiment"]
):

    pooled_results.append({
        "Model": model_name,
        "Experiment": experiment,
        "Pooled MAE": mean_absolute_error(
            group["Actual"], group["Predicted"]
        ),
        "Pooled RMSE": root_mean_squared_error(
            group["Actual"], group["Predicted"]
        ),
        "Pooled R2": r2_score(
            group["Actual"], group["Predicted"]
        )
    })

pooled_df = pd.DataFrame(pooled_results)

print("\n========== POOLED OOF RESULTS ==========")
print(pooled_df.to_string(index=False))


# 11. Save reports
summary_df.to_csv(
    REPORT_DIR / "outlier_validation_summary.csv",
    index=False
)

pooled_df.to_csv(
    REPORT_DIR / "outlier_validation_pooled.csv",
    index=False
)

oof_df.to_csv(
    REPORT_DIR / "outlier_validation_predictions.csv",
    index=False
)


print("\nOutlier validation completed!")
print("All reports saved successfully.")
