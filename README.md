#House Price Prediction

An end-to-end Machine Learning project that predicts residential property prices using the Ames Housing dataset.

## Project Overview

EstateIQ is an interactive house price prediction application developed using Python, Scikit-learn, and Streamlit.

The project implements a complete Machine Learning workflow, including data preprocessing, feature engineering, model development, evaluation, and interactive prediction.

## Key Features

- Data preprocessing and missing-value handling
- Feature engineering
- Numerical scaling and categorical encoding
- Gradient Boosting regression
- Logarithmic target transformation
- Model evaluation and error analysis
- Feature importance analysis
- Interactive Streamlit application

## Dataset

**Dataset:** Ames Housing — Kaggle House Prices

- Training observations: 1,460
- Original columns: 81
- Engineered features: 11
- Total model input features: 90
- Target variable: `SalePrice`

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Joblib
- Streamlit

## Machine Learning Workflow

### 1. Data Preprocessing

- Identified numerical and categorical features.
- Applied median imputation to numerical features.
- Handled missing categorical values using constant imputation.
- Applied StandardScaler to numerical features.
- Used OneHotEncoder for categorical features.

### 2. Feature Engineering

Created 11 additional features:

- TotalSF
- TotalBathrooms
- TotalPorchSF
- TotalOutdoorSF
- TotalBsmtFinSF
- HouseAge
- RemodAge
- GarageAge
- HasGarage
- HasBasement
- IsRemodeled

### 3. Model Development

**Algorithm:** Gradient Boosting Regressor

The model was developed using a Scikit-learn Pipeline and TransformedTargetRegressor.

Target transformation:
- `log1p`
- `expm1`

### 4. Model Evaluation

The dataset was split into training and validation subsets using an 80/20 split.

| Metric | Result |
|---|---:|
| MAE | $15,153.26 |
| RMSE | $25,689.40 |
| R² | 0.9140 |

The model achieved an R² score of 0.9140 on the validation subset.

### 5. Feature Importance

The top features identified by the model were:

1. OverallQual
2. TotalSF
3. TotalBathrooms
4. GrLivArea
5. OverallCond
6. GarageCars
7. LotArea
8. HouseAge
9. GarageArea
10. RemodAge

## Project Structure

```text
house-price-prediction/
│
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
│
├── models/
│   └── final_model.joblib
│
├── reports/
│   ├── validation_metrics.json
│   ├── validation_predictions.csv
│   ├── actual_vs_predicted.png
│   ├── residual_analysis.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── final_model_metadata.json
│   └── final_model_features.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/narminn13/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the model

```bash
python src/train.py
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

## Application

EstateIQ provides an interactive interface where users can enter property characteristics and receive estimated house prices.

## Limitations

The Ames Housing dataset represents properties in Ames, Iowa. Therefore, the application is intended for educational and demonstration purposes and does not provide real-time market valuations.


**Narmin Hasanova**

