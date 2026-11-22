
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import KFold, cross_validate

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)


# 3. Create dataset versions
outlier_ids = [524, 1299]

datasets = {
    "Original Dataset": df.copy(),

    "Filtered Dataset": df[
        ~df["Id"].isin(outlier_ids)
    ].copy()
}


# 4. Define models
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


# 5. Cross-validation configuration
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}


# 6. Store results
results = []


# 7. Evaluate each dataset version
for dataset_name, data in datasets.items():

    print(f"\n{'=' * 55}")
    print(dataset_name)
    print(f"Dataset size: {len(data)}")
    print(f"{'=' * 55}")

    X = data.drop(columns=["Id", "SalePrice"])
    y = data["SalePrice"]

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Create a fresh preprocessor for each model
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

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", model)
        ])

        full_model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

        scores = cross_validate(
            full_model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        mean_mae = -scores["test_MAE"].mean()
        mean_rmse = -scores["test_RMSE"].mean()
        mean_r2 = scores["test_R2"].mean()

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Observations": len(data),
            "Mean MAE": mean_mae,
            "Mean RMSE": mean_rmse,
            "Mean R2": mean_r2,
            "R2 Std": scores["test_R2"].std()
        })

        print(f"\n{model_name}")
        print(f"Mean MAE: {mean_mae:.2f}")
        print(f"Mean RMSE: {mean_rmse:.2f}")
        print(f"Mean R2: {mean_r2:.4f}")


# 8. Compare results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Model", "Dataset"]
)

print("\n========== OUTLIER IMPACT COMPARISON ==========")
print(results_df.to_string(index=False))


# 9. Save report
results_df.to_csv(
    REPORT_DIR / "outlier_impact_comparison.csv",
    index=False
)


print("\nOutlier impact analysis completed!")
print("Report saved: reports/outlier_impact_comparison.csv")
