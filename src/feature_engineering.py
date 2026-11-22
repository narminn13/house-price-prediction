
from pathlib import Path

import pandas as pd


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# 3. Total living area
df["TotalSF"] = (
    df["TotalBsmtSF"]
    + df["1stFlrSF"]
    + df["2ndFlrSF"]
)


# 4. Total bathrooms
df["TotalBathrooms"] = (
    df["FullBath"]
    + df["HalfBath"] * 0.5
    + df["BsmtFullBath"]
    + df["BsmtHalfBath"] * 0.5
)


# 5. Total porch area
porch_features = [
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch"
]

df["TotalPorchSF"] = df[porch_features].sum(axis=1)


# 6. Total outdoor area
df["TotalOutdoorSF"] = (
    df["TotalPorchSF"]
    + df["WoodDeckSF"]
)


# 7. Total finished basement area
df["TotalBsmtFinSF"] = (
    df["BsmtFinSF1"]
    + df["BsmtFinSF2"]
)


# 8. House age
df["HouseAge"] = (
    df["YrSold"]
    - df["YearBuilt"]
)


# 9. Remodeling age
df["RemodAge"] = (
    df["YrSold"]
    - df["YearRemodAdd"]
)


# 10. Garage age
df["GarageAge"] = (
    df["YrSold"]
    - df["GarageYrBlt"]
)


# 11. Garage existence
df["HasGarage"] = (
    df["GarageArea"] > 0
).astype(int)


# 12. Basement existence
df["HasBasement"] = (
    df["TotalBsmtSF"] > 0
).astype(int)


# 13. Remodeling indicator
df["IsRemodeled"] = (
    df["YearRemodAdd"]
    > df["YearBuilt"]
).astype(int)


# 14. New features
new_features = [
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


# 15. Feature statistics
print("\n========== ENGINEERED FEATURES ==========")

print(
    df[new_features].describe().round(2)
)


# 16. Missing values
print("\n========== MISSING VALUES ==========")

print(
    df[new_features]
    .isnull()
    .sum()
    .sort_values(ascending=False)
)


# 17. Find negative age values
age_features = [
    "HouseAge",
    "RemodAge",
    "GarageAge"
]

print("\n========== AGE VALIDATION ==========")

for feature in age_features:

    negative_count = (
        df[feature] < 0
    ).sum()

    print(f"{feature}: {negative_count} negative values")


# 18. Investigate negative remodeling age
negative_remod = df.loc[
    df["RemodAge"] < 0,
    [
        "Id",
        "YearBuilt",
        "YearRemodAdd",
        "YrSold",
        "RemodAge",
        "SalePrice",
        "Neighborhood"
    ]
].copy()

print("\n========== NEGATIVE REMOD AGE RECORDS ==========")

if negative_remod.empty:
    print("No negative remodeling ages found.")
else:
    print(negative_remod.to_string(index=False))

negative_remod.to_csv(
    OUTPUT_DIR / "negative_remod_age_records.csv",
    index=False
)


# 19. Investigate missing garage age
missing_garage_age = df.loc[
    df["GarageAge"].isnull(),
    [
        "Id",
        "GarageType",
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "HasGarage",
        "SalePrice"
    ]
].copy()

print("\n========== MISSING GARAGE AGE RECORDS ==========")
print("Number of records:", len(missing_garage_age))

missing_garage_age.to_csv(
    OUTPUT_DIR / "missing_garage_age_records.csv",
    index=False
)


# 20. Save engineered dataset
output_path = OUTPUT_DIR / "ames_engineered.csv"

df.to_csv(
    output_path,
    index=False
)


# 21. Save feature summary
feature_summary = pd.DataFrame({
    "Feature": new_features,
    "Description": [
        "Total basement and above-ground living area",
        "Total bathrooms including half bathrooms",
        "Total porch area",
        "Total porch and deck area",
        "Total finished basement area",
        "House age at sale",
        "Years since remodeling at sale",
        "Garage age at sale",
        "Whether the house has a garage",
        "Whether the house has a basement",
        "Whether the house was remodeled"
    ]
})

feature_summary.to_csv(
    OUTPUT_DIR / "engineered_feature_summary.csv",
    index=False
)


# 22. Final output
print("\n========== FINAL ==========")

print("Dataset shape:", df.shape)
print("New features created:", len(new_features))

print("\nFeature engineering completed!")
print(f"Dataset saved to: {output_path}")
print("Validation reports saved successfully.")
