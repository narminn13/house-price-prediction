
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EstateIQ | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"


# --------------------------------------------------
# 3. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f5f7fb;
    color: #243247;
}

/* Main content */
.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] li {
    color: #243247 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #14213d !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
}

.sidebar-title {
    font-size: 27px;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    color: #cbd5e1 !important;
    font-size: 13px;
}

.sidebar-heading {
    color: #ffffff !important;
    font-size: 18px;
    font-weight: 700;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #14213d, #243b64);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #e2e8f0 !important;
    font-size: 15px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5eaf2;
    box-shadow: 0 3px 12px rgba(20, 33, 61, 0.04);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #64748b !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #14213d !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #243247 !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #2563eb !important;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox input,
.stTextArea textarea {
    color: #14213d !important;
    background-color: #ffffff !important;
}

[data-baseweb="select"] div {
    color: #14213d !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}

/* Prediction */
.prediction-card {
    background: linear-gradient(120deg, #14213d, #243b64);
    padding: 32px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 24px rgba(20, 33, 61, 0.15);
}

.prediction-card h3 {
    color: #e2e8f0 !important;
    font-size: 18px;
    font-weight: 600;
}

.prediction-card h1 {
    color: #ffffff !important;
    font-size: 42px;
    font-weight: 800;
    margin: 8px 0;
}

.prediction-card p {
    color: #cbd5e1 !important;
    font-size: 14px;
}

/* Headings */
h2, h3, h4 {
    color: #14213d !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5eaf2;
}

/* Footer */
.footer {
    color: #64748b !important;
    text-align: center;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    train_data = load_data()

except FileNotFoundError:
    st.error(
        "Model or dataset file not found. "
        "Check the models/ and data/raw/ folders."
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# --------------------------------------------------
# 5. FEATURE ENGINEERING
# Same logic as train.py
# --------------------------------------------------

def create_features(df):

    df = df.copy()

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

    df["RemodAge"] = (
        df["YrSold"] - df["YearRemodAdd"]
    )

    df["GarageAge"] = (
        df["YrSold"] - df["GarageYrBlt"]
    )

    df["HasGarage"] = (
        df["GarageArea"] > 0
    ).astype(int)

    df["HasBasement"] = (
        df["TotalBsmtSF"] > 0
    ).astype(int)

    df["IsRemodeled"] = (
        df["YearRemodAdd"] > df["YearBuilt"]
    ).astype(int)

    return df


# --------------------------------------------------
# 6. DEFAULT VALUES
# --------------------------------------------------

numeric_defaults = train_data.median(numeric_only=True)

categorical_defaults = train_data.select_dtypes(
    include="object"
).mode().iloc[0]


def get_numeric_default(column):

    value = numeric_defaults.get(column, 0)

    if pd.isna(value):
        return 0

    return value


def get_category_default(column):

    value = categorical_defaults.get(column, "None")

    if pd.isna(value):
        return "None"

    return value


# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🏡 EstateIQ</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'House Price Prediction Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">About the project</div>',
        unsafe_allow_html=True
    )

    st.write(
        "EstateIQ uses a Machine Learning model "
        "to estimate residential property prices."
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">Model information</div>',
        unsafe_allow_html=True
    )

    st.markdown("**Algorithm:** Gradient Boosting")
    st.markdown("**Dataset:** Ames Housing")
    st.markdown("**Training samples:** 1,460")

    st.divider()

    st.markdown(
        '<p style="color:#cbd5e1 !important; font-size:12px;">'
        'Developed as a Machine Learning portfolio project.'
        '</p>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 8. HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Discover Your Property's Value</h1>
    <p>
        Estimate house prices using property characteristics
        and a trained Machine Learning model.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 9. DATASET OVERVIEW
# --------------------------------------------------

st.markdown("## Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Properties",
        f"{len(train_data):,}"
    )

with col2:
    st.metric(
        "Average Price",
        f"${train_data['SalePrice'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Median Price",
        f"${train_data['SalePrice'].median():,.0f}"
    )

with col4:
    st.metric(
        "Features",
        f"{train_data.drop(columns=['Id', 'SalePrice']).shape[1] + 11}"
    )


st.divider()


# --------------------------------------------------
# 10. PROPERTY INPUT FORM
# --------------------------------------------------

st.markdown("## Property Information")

st.write(
    "Enter the property information below to estimate its value."
)

with st.form("property_form"):

    tab1, tab2, tab3 = st.tabs([
        "📍 Location & Quality",
        "📐 Size & Rooms",
        "🏗️ Construction & Garage"
    ])

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            neighborhood_options = sorted(
                train_data["Neighborhood"].dropna().unique()
            )

            default_neighborhood = get_category_default("Neighborhood")

            neighborhood = st.selectbox(
                "Neighborhood",
                neighborhood_options,
                index=(
                    neighborhood_options.index(default_neighborhood)
                    if default_neighborhood in neighborhood_options
                    else 0
                )
            )

            overall_qual = st.slider(
                "Overall Quality",
                1,
                10,
                int(get_numeric_default("OverallQual"))
            )

            overall_cond = st.slider(
                "Overall Condition",
                1,
                10,
                int(get_numeric_default("OverallCond"))
            )

        with col2:

            house_style_options = sorted(
                train_data["HouseStyle"].dropna().unique()
            )

            house_style = st.selectbox(
                "House Style",
                house_style_options
            )

            exter_qual_options = sorted(
                train_data["ExterQual"].dropna().unique()
            )

            exter_qual = st.selectbox(
                "Exterior Quality",
                exter_qual_options
            )

    with tab2:

        col1, col2 = st.columns(2)

        with col1:

            living_area = st.number_input(
                "Above Ground Living Area (sq ft)",
                min_value=300,
                max_value=10000,
                value=int(get_numeric_default("GrLivArea"))
            )

            basement_area = st.number_input(
                "Basement Area (sq ft)",
                min_value=0,
                max_value=5000,
                value=int(get_numeric_default("TotalBsmtSF"))
            )

            lot_area = st.number_input(
                "Lot Area (sq ft)",
                min_value=500,
                max_value=100000,
                value=int(get_numeric_default("LotArea"))
            )

        with col2:

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                value=int(get_numeric_default("BedroomAbvGr"))
            )

            full_bathrooms = st.number_input(
                "Full Bathrooms",
                min_value=0,
                max_value=5,
                value=int(get_numeric_default("FullBath"))
            )

            half_bathrooms = st.number_input(
                "Half Bathrooms",
                min_value=0,
                max_value=5,
                value=int(get_numeric_default("HalfBath"))
            )

    with tab3:

        col1, col2 = st.columns(2)

        with col1:

            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2026,
                value=int(get_numeric_default("YearBuilt"))
            )

            year_remodeled = st.number_input(
                "Year Remodeled",
                min_value=1800,
                max_value=2026,
                value=int(get_numeric_default("YearRemodAdd"))
            )

        with col2:

            garage_area = st.number_input(
                "Garage Area (sq ft)",
                min_value=0,
                max_value=2000,
                value=int(get_numeric_default("GarageArea"))
            )

            garage_cars = st.number_input(
                "Garage Capacity (cars)",
                min_value=0,
                max_value=5,
                value=int(get_numeric_default("GarageCars"))
            )

    submitted = st.form_submit_button(
        "Estimate Property Price"
    )


# --------------------------------------------------
# 11. PREDICTION
# --------------------------------------------------

if submitted:

    input_data = train_data.drop(
        columns=["SalePrice", "Id"],
        errors="ignore"
    ).iloc[[0]].copy()

    numeric_columns = train_data.select_dtypes(
        include="number"
    ).columns

    for column in input_data.columns:

        if column in numeric_columns:
            input_data[column] = get_numeric_default(column)

        else:
            input_data[column] = get_category_default(column)

    # User inputs
    input_data["Neighborhood"] = neighborhood
    input_data["OverallQual"] = overall_qual
    input_data["OverallCond"] = overall_cond
    input_data["HouseStyle"] = house_style
    input_data["ExterQual"] = exter_qual

    input_data["GrLivArea"] = living_area
    input_data["TotalBsmtSF"] = basement_area
    input_data["LotArea"] = lot_area
    input_data["BedroomAbvGr"] = bedrooms
    input_data["FullBath"] = full_bathrooms
    input_data["HalfBath"] = half_bathrooms

    input_data["YearBuilt"] = year_built
    input_data["YearRemodAdd"] = year_remodeled
    input_data["GarageArea"] = garage_area
    input_data["GarageCars"] = garage_cars

    # Feature engineering
    input_data = create_features(input_data)

    try:

        expected_features = (
            model.regressor_.named_steps[
                "preprocessor"
            ].feature_names_in_
        )

        missing_features = (
            set(expected_features) - set(input_data.columns)
        )

        extra_features = (
            set(input_data.columns) - set(expected_features)
        )

        if missing_features or extra_features:

            raise ValueError(
                f"Feature mismatch. Missing: {sorted(missing_features)}. "
                f"Extra: {sorted(extra_features)}"
            )

        input_data = input_data[expected_features]

        prediction = model.predict(input_data)[0]

        if not pd.notna(prediction) or prediction < 0:
            raise ValueError("The model returned an invalid prediction.")

        st.divider()

        st.markdown("## Your Estimated Property Value")

        st.markdown(
            f"""
            <div class="prediction-card">
                <h3>Estimated Sale Price</h3>
                <h1>${prediction:,.0f}</h1>
                <p>Machine Learning prediction</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Price comparison
        st.markdown("### Price Comparison")

        average_price = train_data["SalePrice"].mean()

        comparison = pd.DataFrame({
            "Category": [
                "Your Property",
                "Dataset Average"
            ],
            "Price": [
                prediction,
                average_price
            ]
        })

        fig, ax = plt.subplots(figsize=(8, 3.5))

        bars = ax.bar(
            comparison["Category"],
            comparison["Price"]
        )

        ax.set_ylabel("Price (USD)")
        ax.set_title("Estimated Price vs Dataset Average")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"${height:,.0f}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)
        plt.close(fig)

        # Property details
        with st.expander("View submitted property details"):

            details = pd.DataFrame({
                "Feature": [
                    "Neighborhood",
                    "Overall Quality",
                    "Overall Condition",
                    "Living Area",
                    "Basement Area",
                    "Lot Area",
                    "Bedrooms",
                    "Full Bathrooms",
                    "Half Bathrooms",
                    "Year Built",
                    "Year Remodeled",
                    "Garage Area",
                    "Garage Capacity"
                ],
                "Value": [
                    neighborhood,
                    overall_qual,
                    overall_cond,
                    living_area,
                    basement_area,
                    lot_area,
                    bedrooms,
                    full_bathrooms,
                    half_bathrooms,
                    year_built,
                    year_remodeled,
                    garage_area,
                    garage_cars
                ]
            })

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'EstateIQ | House Price Prediction | Powered by Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
