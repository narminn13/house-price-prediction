
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"
VISUAL_DIR = BASE_DIR / "visualizations"

REPORT_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)


# 3. Analyze important numerical features
features = [
    "SalePrice",
    "GrLivArea",
    "OverallQual",
    "TotalBsmtSF",
    "GarageArea"
]

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df[features].describe().round(2))


# 4. IQR outlier detection
def get_iqr_bounds(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound


outlier_summary = []

for feature in features:

    lower, upper = get_iqr_bounds(df[feature])

    mask = (
        (df[feature] < lower) |
        (df[feature] > upper)
    )

    outlier_summary.append({
        "Feature": feature,
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Outlier Count": int(mask.sum())
    })

    print(f"\n{feature}")
    print(f"Lower bound: {lower:.2f}")
    print(f"Upper bound: {upper:.2f}")
    print(f"Outlier count: {mask.sum()}")


outlier_summary_df = pd.DataFrame(outlier_summary)

outlier_summary_df.to_csv(
    REPORT_DIR / "outlier_summary.csv",
    index=False
)


# 5. Identify extreme living area observations
large_area_mask = df["GrLivArea"] > 4000

large_area_houses = df.loc[
    large_area_mask,
    [
        "Id",
        "SalePrice",
        "GrLivArea",
        "OverallQual",
        "TotalBsmtSF",
        "GarageCars",
        "YearBuilt",
        "Neighborhood"
    ]
].copy()

large_area_houses.to_csv(
    REPORT_DIR / "large_area_houses.csv",
    index=False
)

print("\n========== HOUSES WITH GRLIVAREA > 4000 ==========")

if large_area_houses.empty:
    print("No houses found.")
else:
    print(
        large_area_houses.to_string(index=False)
    )


# 6. Flag possible outliers without deleting them
df["LargeAreaFlag"] = df["GrLivArea"] > 4000

df["HighPriceFlag"] = (
    df["SalePrice"] > get_iqr_bounds(df["SalePrice"])[1]
)

df["PotentialOutlier"] = (
    df["LargeAreaFlag"] |
    df["HighPriceFlag"]
)

candidates = df.loc[
    df["PotentialOutlier"],
    [
        "Id",
        "SalePrice",
        "GrLivArea",
        "OverallQual",
        "YearBuilt",
        "Neighborhood",
        "LargeAreaFlag",
        "HighPriceFlag"
    ]
]

candidates.to_csv(
    REPORT_DIR / "outlier_candidates.csv",
    index=False
)


# 7. Visualize living area and sale price
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice",
    hue="LargeAreaFlag",
    palette={False: "steelblue", True: "red"},
    alpha=0.7
)

plt.axvline(
    4000,
    color="black",
    linestyle="--",
    label="4000 sq ft threshold"
)

plt.title("Living Area vs Sale Price")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.legend()
plt.tight_layout()

plt.savefig(
    VISUAL_DIR / "living_area_outlier_analysis.png"
)

plt.close()


# 8. Boxplots
for feature in features:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[feature])

    plt.title(f"Outlier Analysis - {feature}")
    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / f"{feature.lower()}_boxplot.png"
    )

    plt.close()


print("\nOutlier analysis completed!")
print("Reports saved successfully.")
print("Visualizations saved successfully.")
