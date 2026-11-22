
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import KFold, cross_validate

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["Id"])

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]


# 3. Identify feature types
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 4. Create preprocessing pipeline
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


# 5. Define models
models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=10.0),

    "Lasso": Lasso(
        alpha=0.0005,
        max_iter=10000
    ),

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


# 6. Configure cross-validation
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


# 7. Train and evaluate models
results = []

for name, model in models.items():

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

    fold_results = []

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")

    for fold_number, (mae, rmse, r2) in enumerate(
        zip(
            -scores["test_MAE"],
            -scores["test_RMSE"],
            scores["test_R2"]
        ),
        start=1
    ):

        fold_results.append({
            "Model": name,
            "Fold": fold_number,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(
            f"Fold {fold_number}: "
            f"MAE = {mae:.2f} | "
            f"RMSE = {rmse:.2f} | "
            f"R2 = {r2:.4f}"
        )

    results.append({
        "Model": name,
        "Mean MAE": -scores["test_MAE"].mean(),
        "Mean RMSE": -scores["test_RMSE"].mean(),
        "Mean R2": scores["test_R2"].mean(),
        "R2 Std": scores["test_R2"].std()
    })


# 8. Compare models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Mean RMSE"
)

fold_results_df = pd.DataFrame(fold_results)

print("\n========== CROSS-VALIDATION SUMMARY ==========")
print(results_df.to_string(index=False))


# 9. Save reports
results_df.to_csv(
    REPORT_DIR / "cross_validation_results.csv",
    index=False
)

fold_results_df.to_csv(
    REPORT_DIR / "fold_results.csv",
    index=False
)

print("\nReports saved successfully!")
print("Files:")
print("- reports/cross_validation_results.csv")
print("- reports/fold_results.csv")
