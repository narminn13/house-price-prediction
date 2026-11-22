
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


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


# 4. Prepare features and target
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

print("Feature count:", X.shape[1])


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


# 6. Model creation function
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


# 7. Outer and inner cross-validation
outer_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# 8. Hyperparameter search space
param_distributions = {
    "regressor__model__n_estimators": [
        200, 300, 400
    ],
    "regressor__model__learning_rate": [
        0.03, 0.05, 0.07
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


# 9. Model definitions
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

        "Random Forest": create_model(
            X_train,
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        ),

        "Extra Trees": create_model(
            X_train,
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
    }


# 10. Results storage
fold_results = []
pooled_results = []
prediction_results = []
tuning_results = []


# 11. Outer cross-validation
for fold, (train_idx, val_idx) in enumerate(
    outer_cv.split(X), start=1
):

    print("\n" + "=" * 60)
    print(f"OUTER FOLD {fold}")
    print("=" * 60)

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    models = get_models(X_train)

    # 11A. Evaluate standard models
    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        predictions = np.maximum(predictions, 0)

        fold_results.append({
            "Model": model_name,
            "Fold": fold,
            "MAE": mean_absolute_error(
                y_val, predictions
            ),
            "RMSE": root_mean_squared_error(
                y_val, predictions
            ),
            "R2": r2_score(
                y_val, predictions
            )
        })

        for index, actual, predicted in zip(
            val_idx, y_val, predictions
        ):
            prediction_results.append({
                "Model": model_name,
                "Fold": fold,
                "Dataset Index": index,
                "Actual Price": actual,
                "Predicted Price": predicted
            })

        print(f"{model_name} completed")

    # 11B. Tune Gradient Boosting inside outer fold
    print("\nTuning Gradient Boosting inside outer fold...")

    tuned_base_model = create_model(
        X_train,
        GradientBoostingRegressor(random_state=42)
    )

    search = RandomizedSearchCV(
        estimator=tuned_base_model,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    search.fit(X_train, y_train)

    tuned_predictions = search.predict(X_val)
    tuned_predictions = np.maximum(
        tuned_predictions, 0
    )

    fold_results.append({
        "Model": "Tuned Gradient Boosting",
        "Fold": fold,
        "MAE": mean_absolute_error(
            y_val, tuned_predictions
        ),
        "RMSE": root_mean_squared_error(
            y_val, tuned_predictions
        ),
        "R2": r2_score(
            y_val, tuned_predictions
        )
    })

    for index, actual, predicted in zip(
        val_idx, y_val, tuned_predictions
    ):
        prediction_results.append({
            "Model": "Tuned Gradient Boosting",
            "Fold": fold,
            "Dataset Index": index,
            "Actual Price": actual,
            "Predicted Price": predicted
        })

    tuning_results.append({
        "Fold": fold,
        "Best Inner CV MAE": -search.best_score_,
        "Best Parameters": json.dumps(
            search.best_params_
        )
    })

    print(
        "Tuned Gradient Boosting completed. "
        f"Best inner MAE: {-search.best_score_:.2f}"
    )


# 12. Create result dataframes
fold_df = pd.DataFrame(fold_results)
predictions_df = pd.DataFrame(prediction_results)
tuning_df = pd.DataFrame(tuning_results)


# 13. Mean fold results
mean_fold_df = (
    fold_df
    .groupby("Model")
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "R2": "mean"
    })
    .reset_index()
    .sort_values("MAE")
)


# 14. Pooled out-of-fold results
for model_name, group in predictions_df.groupby("Model"):

    actual = group["Actual Price"]
    predicted = group["Predicted Price"]

    pooled_results.append({
        "Model": model_name,
        "Pooled MAE": mean_absolute_error(
            actual, predicted
        ),
        "Pooled RMSE": root_mean_squared_error(
            actual, predicted
        ),
        "Pooled R2": r2_score(
            actual, predicted
        )
    })

pooled_df = pd.DataFrame(pooled_results).sort_values(
    "Pooled MAE"
)


# 15. Display results
print("\n========== MEAN FOLD RESULTS ==========")
print(mean_fold_df.to_string(index=False))

print("\n========== POOLED RESULTS ==========")
print(pooled_df.to_string(index=False))


# 16. Save reports
fold_df.to_csv(
    REPORT_DIR / "final_model_comparison_folds.csv",
    index=False
)

mean_fold_df.to_csv(
    REPORT_DIR / "final_model_comparison_mean_folds.csv",
    index=False
)

pooled_df.to_csv(
    REPORT_DIR / "final_model_comparison_pooled.csv",
    index=False
)

predictions_df.to_csv(
    REPORT_DIR / "final_model_predictions.csv",
    index=False
)

tuning_df.to_csv(
    REPORT_DIR / "nested_tuning_parameters.csv",
    index=False
)


print("\nFinal model comparison completed!")
print("Reports saved successfully.")
