
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import joblib


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# 2. Load dataset
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


# 4. Prepare X and y
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

print("Feature count:", X.shape[1])


# 5. Identify feature types
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 6. Preprocessing pipelines
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


# 7. Create model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)


# 8. Hyperparameter search space
param_distributions = {
    "regressor__model__n_estimators": [
        100, 200, 300, 400
    ],
    "regressor__model__learning_rate": [
        0.03, 0.05, 0.07, 0.1
    ],
    "regressor__model__max_depth": [
        2, 3, 4
    ],
    "regressor__model__min_samples_split": [
        2, 5, 10
    ],
    "regressor__model__min_samples_leaf": [
        1, 2, 4
    ],
    "regressor__model__subsample": [
        0.7, 0.8, 0.9, 1.0
    ]
}


# 9. Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 10. Randomized search
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=25,
    scoring="neg_mean_absolute_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
    refit=True
)


# 11. Fit search
print("\nStarting hyperparameter tuning...")

search.fit(X, y)


# 12. Best results
best_mae = -search.best_score_

print("\n========== BEST RESULTS ==========")
print("Best CV MAE:", round(best_mae, 2))
print("\nBest parameters:")

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


# 13. Save all search results
results_df = pd.DataFrame(search.cv_results_)

results_df["Mean CV MAE"] = (
    -results_df["mean_test_score"]
)

results_df["Std CV MAE"] = (
    results_df["std_test_score"]
)

results_df = results_df.sort_values(
    by="rank_test_score"
)

results_df.to_csv(
    REPORT_DIR / "hyperparameter_tuning_results.csv",
    index=False
)


# 14. Save best parameters
best_params = {
    key: (
        value.item()
        if isinstance(value, np.generic)
        else value
    )
    for key, value in search.best_params_.items()
}

with open(
    REPORT_DIR / "best_hyperparameters.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        best_params,
        file,
        indent=4
    )


# 15. Save tuned model
joblib.dump(
    search.best_estimator_,
    MODEL_DIR / "tuned_gradient_boosting.joblib"
)


print("\nTuning completed successfully!")
print("Best CV MAE:", round(best_mae, 2))
print("Results saved in reports/")
print("Tuned model saved in models/")
