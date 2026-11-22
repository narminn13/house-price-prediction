
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")


# 1. Missing values percentage

missing = df.isnull().sum()
missing = missing[missing > 0]

missing_percentage = (
    missing / len(df) * 100
).sort_values(ascending=False)

print("\nMissing values percentage:")
print(missing_percentage.round(2))


plt.figure(figsize=(12, 8))

missing_percentage.sort_values().plot(
    kind="barh"
)

plt.title("Percentage of Missing Values")
plt.xlabel("Missing Values (%)")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "missing_values.png")
plt.show()


# 2. SalePrice distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    df["SalePrice"],
    kde=True,
    bins=40
)

plt.title("SalePrice Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "saleprice_distribution.png")
plt.show()


# 3. SalePrice boxplot

plt.figure(figsize=(10, 4))

sns.boxplot(x=df["SalePrice"])

plt.title("SalePrice Boxplot")
plt.xlabel("Sale Price")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "saleprice_boxplot.png")
plt.show()


# 4. Correlation with SalePrice

numeric_df = df.select_dtypes(include="number")

correlations = (
    numeric_df
    .drop(columns=["Id"])
    .corr()["SalePrice"]
    .drop("SalePrice")
    .sort_values(ascending=False)
)

print("\nTop 15 positively correlated features:")
print(correlations.head(15))

print("\nTop 15 negatively correlated features:")
print(correlations.tail(15))


# 5. Correlation heatmap

top_features = correlations.abs().sort_values(
    ascending=False
).head(15).index

plt.figure(figsize=(12, 9))

sns.heatmap(
    df[list(top_features) + ["SalePrice"]]
    .corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.show()

# 6. Living area vs SalePrice

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Living Area vs Sale Price")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "living_area_vs_price.png")
plt.show()


# 7. Overall Quality vs SalePrice

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("Overall Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "quality_vs_price.png")
plt.show()


# 8. Target skewness

print("\nSalePrice skewness:")
print(df["SalePrice"].skew())


# 9. Identify extreme observations

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Outlier Analysis")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "outlier_analysis.png")
plt.show()


# Investigate unusually large houses
outliers = df[
    (df["GrLivArea"] > 4000) &
    (df["SalePrice"] < 300000)
]

print("\nPotential outliers:")
print(outliers[["Id", "GrLivArea", "SalePrice"]])