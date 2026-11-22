from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
VISUALIZATION_DIR = BASE_DIR / "visualizations"

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

# 3. Dataset dimensions
print("\nDataset shape:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# 4. First observations
print("\nFirst 5 rows:")
print(df.head())


# 5. Dataset information
print("\nDataset information:")
df.info()


# 6. Statistical summary
print("\nStatistical summary:")
print(df.describe())


# 7. Missing values
print("\nMissing values:")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print(missing_values)


# 8. Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())


# 9. Column types
numeric_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(exclude="number").columns

print("\nNumeric columns:", len(numeric_columns))
print("Categorical columns:", len(categorical_columns))


# 10. Target variable analysis
print("\nSalePrice statistics:")
print(df["SalePrice"].describe())


# 11. SalePrice distribution
plt.figure(figsize=(10, 6))

sns.histplot(df["SalePrice"], kde=True)

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(VISUALIZATION_DIR / "saleprice_distribution.png")
plt.show()