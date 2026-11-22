
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
VISUAL_DIR = BASE_DIR / "visualizations"

REPORT_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)


# 2. Load data
df = pd.read_csv(DATA_PATH)


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


# 5. Preprocessor function
def create_preprocessor(X_train):

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

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


# 6. Model creation
def create_model(X_train, estimator):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(X_train)),
        ("model", estimator)
    ])

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )


# 7. Models
def get_models(X_train):

    return {
        "Gradient Boosting": create_model(
            X_train,
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        ),

        "Tuned Gradient Boosting": create_model(
            X_train,
            GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=1,
                subsample=0.7,
                random_state=42
            )
        )
    }


# 8. Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 9. Importance storage
importance_results = []


# 10. Permutation importance
for model_name in [
    "Gradient Boosting",
    "Tuned Gradient Boosting"
]:

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X), start=1
    ):

        print(f"\nProcessing fold {fold}...")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = get_models(X_train)[model_name]

        model.fit(X_train, y_train)

        result = permutation_importance(
            estimator=model,
            X=X_val,
            y=y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

        for feature_index, feature_name in enumerate(X.columns):

            importance_results.append({
                "Model": model_name,
                "Fold": fold,
                "Feature": feature_name,
                "Importance Mean": (
                    result.importances_mean[feature_index]
                ),
                "Importance Std": (
                    result.importances_std[feature_index]
                )
            })

        print(f"Fold {fold} completed")


# 11. Create importance dataframe
importance_df = pd.DataFrame(importance_results)


# 12. Aggregate across folds
summary_df = (
    importance_df
    .groupby(["Model", "Feature"])
    .agg(
        Importance_Mean=("Importance Mean", "mean"),
        Importance_Std=("Importance Mean", "std"),
        Mean_Permutation_Std=("Importance Std", "mean"),
        Fold_Stability=(
            "Importance Mean",
            lambda values: (values > 0).sum()
        )
    )
    .reset_index()
)


# 13. Save complete results
importance_df.to_csv(
    REPORT_DIR / "explainability_fold_results.csv",
    index=False
)

summary_df.to_csv(
    REPORT_DIR / "explainability_summary.csv",
    index=False
)


# 14. Plot top 20 features for each model
for model_name in summary_df["Model"].unique():

    model_summary = (
        summary_df[
            summary_df["Model"] == model_name
        ]
        .sort_values(
            "Importance_Mean",
            ascending=False
        )
        .head(20)
        .sort_values(
            "Importance_Mean",
            ascending=True
        )
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        model_summary["Feature"],
        model_summary["Importance_Mean"],
        xerr=model_summary["Importance_Std"],
        capsize=3
    )

    plt.xlabel("Increase in MAE after permutation ($)")
    plt.ylabel("Feature")
    plt.title(f"{model_name}: Permutation Importance")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        VISUAL_DIR / f"{safe_name}_permutation_importance.png",
        dpi=300
    )

    plt.close()


# 15. Display top features
for model_name in summary_df["Model"].unique():

    print("\n" + "=" * 60)
    print(f"TOP FEATURES: {model_name}")
    print("=" * 60)

    top_features = (
        summary_df[
            summary_df["Model"] == model_name
        ]
        .sort_values(
            "Importance_Mean",
            ascending=False
        )
        .head(20)
    )

    print(
        top_features.round(3).to_string(index=False)
    )


print("\nExplainability analysis completed!")
print("Reports and plots saved successfully.")
