
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
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


# 4. Prepare X and y
X = df.drop(
    columns=["Id", "SalePrice"]
)

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


# 7. Define model
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gb_model)
])

model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)


# 8. Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 9. Store importance results
importance_results = []


# 10. Calculate permutation importance
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X), start=1
):

    print(f"\n========== FOLD {fold} ==========")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model.fit(
        X_train,
        y_train
    )

    print("Model trained.")

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    print("Permutation importance calculated.")

    fold_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
        "Fold": fold
    })

    importance_results.append(fold_df)


# 11. Combine fold results
all_importance = pd.concat(
    importance_results,
    ignore_index=True
)


# 12. Aggregate importance across folds
summary = (
    all_importance
    .groupby("Feature")
    .agg(
        Importance_Mean=("Importance_Mean", "mean"),
        Importance_Std=("Importance_Mean", "std"),
        Fold_Stability=("Importance_Mean", "count")
    )
    .reset_index()
    .sort_values(
        by="Importance_Mean",
        ascending=False
    )
)


# 13. Display results
print("\n========== TOP 25 FEATURES ==========")

print(
    summary.head(25).to_string(index=False)
)


print("\n========== LOWEST 15 FEATURES ==========")

print(
    summary.tail(15).to_string(index=False)
)


# 14. Save reports
all_importance.to_csv(
    REPORT_DIR / "feature_importance_folds.csv",
    index=False
)

summary.to_csv(
    REPORT_DIR / "feature_importance_summary.csv",
    index=False
)


print("\nFeature importance analysis completed!")

print(
    "Reports saved in:",
    REPORT_DIR
)
