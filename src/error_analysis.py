
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

ids = df["Id"]


# 3. Identify feature types
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 4. Preprocessing
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
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


# 5. Define models
models = {
    "Linear Regression": LinearRegression(),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# 6. Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 7. Analyze prediction errors
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

    predictions = cross_val_predict(
        full_model,
        X,
        y,
        cv=cv,
        n_jobs=1
    )

    errors = pd.DataFrame({
        "Id": ids,
        "Actual Price": y,
        "Predicted Price": predictions,
        "Residual": y.to_numpy() - predictions,
        "Absolute Error": np.abs(y.to_numpy() - predictions)
    })

    errors["Absolute Percentage Error"] = (
        errors["Absolute Error"] /
        errors["Actual Price"]
    ) * 100

    # Add original house features
    errors = pd.concat(
        [
            errors.reset_index(drop=True),
            X.reset_index(drop=True)
        ],
        axis=1
    )

    errors = errors.sort_values(
        by="Absolute Error",
        ascending=False
    )

    output_path = REPORT_DIR / (
        f"{name.lower().replace(' ', '_')}_error_analysis.csv"
    )

    errors.to_csv(output_path, index=False)

    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")

    selected_columns = [
        "Id",
        "Actual Price",
        "Predicted Price",
        "Absolute Error",
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "Neighborhood"
    ]

    print(
        errors[selected_columns].head(15).to_string(index=False)
    )

    print(f"\nSaved: {output_path}")


print("\nError analysis completed!")
