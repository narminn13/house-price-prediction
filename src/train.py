
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

def create_features(data):

    data = data.copy()

    data["TotalSF"] = (
        data["TotalBsmtSF"]
        + data["1stFlrSF"]
        + data["2ndFlrSF"]
    )

    data["TotalBathrooms"] = (
        data["FullBath"]
        + data["HalfBath"] * 0.5
        + data["BsmtFullBath"]
        + data["BsmtHalfBath"] * 0.5
    )

    porch_features = [
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch"
    ]

    data["TotalPorchSF"] = data[porch_features].sum(axis=1)

    data["TotalOutdoorSF"] = (
        data["TotalPorchSF"] + data["WoodDeckSF"]
    )

    data["TotalBsmtFinSF"] = (
        data["BsmtFinSF1"] + data["BsmtFinSF2"]
    )

    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]

    data["RemodAge"] = (
        data["YrSold"] - data["YearRemodAdd"]
    )

    data["GarageAge"] = (
        data["YrSold"] - data["GarageYrBlt"]
    )

    data["HasGarage"] = (
        data["GarageArea"] > 0
    ).astype(int)

    data["HasBasement"] = (
        data["TotalBsmtSF"] > 0
    ).astype(int)

    data["IsRemodeled"] = (
        data["YearRemodAdd"] > data["YearBuilt"]
    ).astype(int)

    return data


print("\nCreating engineered features...")

df = create_features(df)


# --------------------------------------------------
# 4. PREPARE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

print(f"Feature count: {X.shape[1]}")
print(f"Total samples: {X.shape[0]}")


# --------------------------------------------------
# 5. TRAIN-VALIDATION SPLIT
# --------------------------------------------------

print("\nSplitting dataset...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_valid)}")


# --------------------------------------------------
# 6. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# --------------------------------------------------
# 7. CREATE MODEL PIPELINE
# --------------------------------------------------

def create_model():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=1,
        subsample=0.7,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", gradient_boosting)
    ])

    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    return model


# --------------------------------------------------
# 8. TRAIN VALIDATION MODEL
# --------------------------------------------------

print("\nTraining validation model...")

validation_model = create_model()

validation_model.fit(X_train, y_train)

print("Validation model training completed!")


# --------------------------------------------------
# 9. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating validation predictions...")

y_pred = validation_model.predict(X_valid)


# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_valid, y_pred)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

r2 = r2_score(y_valid, y_pred)

validation_metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_valid)),
    "validation_size": 0.2,
    "random_state": 42
}

print("\n" + "=" * 55)
print("MODEL EVALUATION RESULTS")
print("=" * 55)

print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2:   {r2:.4f}")

print("=" * 55)


# --------------------------------------------------
# 11. SAVE VALIDATION METRICS
# --------------------------------------------------

metrics_path = REPORT_DIR / "validation_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(validation_metrics, file, indent=4)


# --------------------------------------------------
# 12. SAVE VALIDATION PREDICTIONS
# --------------------------------------------------

validation_results = pd.DataFrame({
    "Actual SalePrice": y_valid.values,
    "Predicted SalePrice": y_pred,
    "Absolute Error": np.abs(y_valid.values - y_pred),
    "Residual": y_valid.values - y_pred
})

validation_results.to_csv(
    REPORT_DIR / "validation_predictions.csv",
    index=False
)


# --------------------------------------------------
# 13. ACTUAL VS PREDICTED PLOT
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    y_valid,
    y_pred,
    alpha=0.6
)

min_value = min(y_valid.min(), y_pred.min())
max_value = max(y_valid.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 14. RESIDUAL ANALYSIS
# --------------------------------------------------

residuals = y_valid.values - y_pred

plt.figure(figsize=(9, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sale Price")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "residual_analysis.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 15. TRAIN FINAL MODEL ON FULL DATASET
# --------------------------------------------------

print("\nTraining final model on full dataset...")

final_model = create_model()

final_model.fit(X, y)

print("Final model training completed!")


# --------------------------------------------------
# 16. FEATURE IMPORTANCE
# --------------------------------------------------

print("\nCalculating feature importance...")

fitted_pipeline = final_model.regressor_

preprocessor = fitted_pipeline.named_steps["preprocessor"]
regressor = fitted_pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": regressor.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

top_features = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 20 Most Important Features")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "feature_importance.png",
    dpi=300
)

plt.close()

print("\nTop 10 Important Features:")

print(
    importance_df.head(10).to_string(index=False)
)


# --------------------------------------------------
# 17. SAVE FINAL MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "final_model.joblib"

joblib.dump(final_model, model_path)

print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# 18. SAVE MODEL METADATA
# --------------------------------------------------

metadata = {
    "model_name": "Tuned Gradient Boosting Regressor",
    "dataset": "Kaggle House Prices - Ames Housing",
    "training_samples": int(X.shape[0]),
    "original_feature_count": int(X.shape[1]),
    "numeric_feature_count": len(numeric_features),
    "categorical_feature_count": len(categorical_features),
    "target": "SalePrice",
    "target_transformation": "log1p",
    "inverse_target_transformation": "expm1",
    "validation_metrics": validation_metrics,
    "hyperparameters": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_split": 10,
        "min_samples_leaf": 1,
        "subsample": 0.7,
        "random_state": 42
    }
}

metadata_path = REPORT_DIR / "final_model_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)


# --------------------------------------------------
# 19. SAVE FEATURE INFORMATION
# --------------------------------------------------

feature_info = pd.DataFrame({
    "Feature": X.columns,
    "Data Type": X.dtypes.astype(str).values,
    "Feature Type": [
        "Numeric" if feature in numeric_features
        else "Categorical"
        for feature in X.columns
    ]
})

feature_info.to_csv(
    REPORT_DIR / "final_model_features.csv",
    index=False
)


# --------------------------------------------------
# 20. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL MODEL TRAINING SUMMARY")
print("=" * 55)

print("Model: Tuned Gradient Boosting Regressor")
print(f"Training samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

print("\nValidation Performance:")
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2: {r2:.4f}")

print("\nGenerated files:")
print(f"Model: {model_path}")
print(f"Metadata: {metadata_path}")
print(f"Metrics: {metrics_path}")

print("\nFinal model is ready for prediction!")
