
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

MODEL_CLASSIFIER = MODEL_DIR / "investment_classifier.joblib"
MODEL_REGRESSOR = MODEL_DIR / "future_price_regressor.joblib"
FEATURE_FILE = MODEL_DIR / "feature_columns.json"

REAL_DATA = DATA_DIR / "india_housing_prices.csv"
DEMO_DATA = DATA_DIR / "demo_housing_data.csv"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    if REAL_DATA.exists():
        return pd.read_csv(REAL_DATA), "Provided dataset"

    if DEMO_DATA.exists():
        return pd.read_csv(DEMO_DATA), "Synthetic demo dataset"

    raise FileNotFoundError(
        "No housing dataset was found in the data folder."
    )


try:
    df, data_source = load_data()
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# --------------------------------------------------
# 3. Validate dataset
# --------------------------------------------------

required_columns = {
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "Dataset is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

if df.empty:
    st.error("The housing dataset contains no records.")
    st.stop()

df = df.copy()

df["Price_in_Lakhs"] = pd.to_numeric(
    df["Price_in_Lakhs"], errors="coerce"
)

df["Size_in_SqFt"] = pd.to_numeric(
    df["Size_in_SqFt"], errors="coerce"
)

df = df.dropna(
    subset=["Price_in_Lakhs", "Size_in_SqFt"]
)

df = df[
    (df["Price_in_Lakhs"] > 0)
    & (df["Size_in_SqFt"] > 0)
]

if df.empty:
    st.error("No valid property records were found.")
    st.stop()


# --------------------------------------------------
# 4. Application heading
# --------------------------------------------------

st.title("🏠 Real Estate Investment Advisor")

st.caption(
    "Property investment classification and "
    "five-year price forecasting."
)

if data_source == "Synthetic demo dataset":
    st.warning(
        "Demo mode uses synthetic housing data. "
        "Replace it with the assigned india_housing_prices.csv "
        "before presenting results as real."
    )

st.info(
    f"Dataset loaded: {data_source} · {len(df):,} rows"
)


# --------------------------------------------------
# 5. Load trained models
# --------------------------------------------------

@st.cache_resource
def load_models():
    required_files = [
        MODEL_CLASSIFIER,
        MODEL_REGRESSOR,
        FEATURE_FILE
    ]

    missing_files = [
        file.name for file in required_files
        if not file.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing_files)
        )

    classifier = joblib.load(MODEL_CLASSIFIER)
    regressor = joblib.load(MODEL_REGRESSOR)

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    if not feature_columns:
        raise ValueError("The feature columns file is empty.")

    return classifier, regressor, feature_columns


try:
    classifier, regressor, feature_cols = load_models()

except Exception as error:
    st.error(f"Unable to load trained models: {error}")
    st.info(
        "Run python train.py from the project folder "
        "to train and save the models."
    )
    st.stop()


# --------------------------------------------------
# 6. Tabs
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Property prediction",
        "Market insights",
        "About"
    ]
)


# --------------------------------------------------
# 7. Property prediction
# --------------------------------------------------

with tab1:

    st.subheader("Analyze a property")

    st.write(
        "Enter the property details to estimate its "
        "investment classification and five-year price."
    )

    c1, c2, c3 = st.columns(3)

    states = sorted(
        df["State"].dropna().astype(str).unique()
    )

    cities = sorted(
        df["City"].dropna().astype(str).unique()
    )

    property_types = sorted(
        df["Property_Type"].dropna().astype(str).unique()
    )

    with c1:

        state = st.selectbox("State", states)

        state_cities = sorted(
            df.loc[
                df["State"].astype(str) == state,
                "City"
            ].dropna().astype(str).unique()
        )

        city = st.selectbox(
            "City",
            state_cities if state_cities else cities
        )

        locality = st.text_input(
            "Locality",
            value="Central"
        )

    with c2:

        ptype = st.selectbox(
            "Property type",
            property_types
        )

        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=10,
            value=2
        )

        size = st.number_input(
            "Size (sq ft)",
            min_value=100,
            value=1000,
            step=50
        )

    with c3:

        price = st.number_input(
            "Current price (₹ lakh)",
            min_value=1.0,
            value=50.0,
            step=1.0
        )

        year = st.number_input(
            "Year built",
            min_value=1950,
            max_value=2026,
            value=2018
        )

        schools = st.number_input(
            "Nearby schools",
            min_value=0,
            max_value=100,
            value=3
        )

    if st.button(
        "Analyze property",
        type="primary"
    ):

        # Default values for all training features
        row = {column: 0 for column in feature_cols}

        values = {
            "State": state,
            "City": city,
            "Locality": locality,
            "Property_Type": ptype,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price,
            "Price_per_SqFt": price * 100000 / max(size, 1),
            "Year_Built": year,
            "Age_of_Property": max(0, 2026 - year),
            "Nearby_Schools": schools,
            "Nearby_Hospitals": 2,
            "Floor_No": 2,
            "Total_Floors": 8,
            "Parking_Space": 1,
            "Furnished_Status": "Semi",
            "Public_Transport_Accessibility": "Medium",
            "Security": "Gated",
            "Amenities": "Basic",
            "Facing": "East",
            "Owner_Type": "Individual",
            "Availability_Status": "Available"
        }

        for column, value in values.items():
            if column in row:
                row[column] = value

        X = pd.DataFrame(
            [row],
            columns=feature_cols
        )

        try:

            prediction = int(
                classifier.predict(X)[0]
            )

            probability = None

            if hasattr(classifier, "predict_proba"):
                probability = float(
                    classifier.predict_proba(X)[0][1]
                )

            future_price = float(
                regressor.predict(X)[0]
            )

            if not pd.notna(future_price):
                raise ValueError(
                    "The model returned an invalid price."
                )

            if future_price < 0:
                future_price = 0

            st.divider()

            st.subheader("Prediction results")

            a, b = st.columns(2)

            a.metric(
                "Investment assessment",
                "Good investment"
                if prediction == 1
                else "Needs caution"
            )

            if probability is not None:
                a.metric(
                    "Model confidence",
                    f"{probability:.1%}"
                )

            b.metric(
                "Estimated price after 5 years",
                f"₹{future_price:.2f} lakh"
            )

            st.caption(
                "These are model estimates, not financial advice "
                "or guaranteed investment returns."
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------------------------
# 8. Market insights
# --------------------------------------------------

with tab2:

    st.subheader("Dataset overview")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Properties",
        f"{len(df):,}"
    )

    m2.metric(
        "Median price",
        f"₹{df['Price_in_Lakhs'].median():.1f} lakh"
    )

    m3.metric(
        "Median size",
        f"{df['Size_in_SqFt'].median():,.0f} sq ft"
    )

    st.plotly_chart(
        px.histogram(
            df,
            x="Price_in_Lakhs",
            nbins=30,
            title="Property price distribution",
            labels={
                "Price_in_Lakhs": "Price (₹ lakh)"
            }
        ),
        use_container_width=True
    )

    city_prices = (
        df.groupby("City", as_index=False)["Price_in_Lakhs"]
        .mean()
        .sort_values(
            "Price_in_Lakhs",
            ascending=False
        )
        .head(15)
    )

    st.plotly_chart(
        px.bar(
            city_prices,
            x="City",
            y="Price_in_Lakhs",
            title="Average price by city",
            labels={
                "Price_in_Lakhs": "Average price (₹ lakh)"
            }
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            df,
            x="Size_in_SqFt",
            y="Price_in_Lakhs",
            color="Property_Type",
            title="Property size vs price",
            labels={
                "Size_in_SqFt": "Size (sq ft)",
                "Price_in_Lakhs": "Price (₹ lakh)"
            }
        ),
        use_container_width=True
    )


# --------------------------------------------------
# 9. About
# --------------------------------------------------

with tab3:

    st.subheader("Project information")

    st.markdown("""
    **Workflow**

    Data cleaning → Feature engineering → Exploratory data
    analysis → Classification and regression → MLflow tracking
    → Streamlit application.

    **Investment classification**

    A property is labeled a good investment when its price is
    at or below the median price in the training dataset.

    **Five-year price forecast**

    The training target uses an assumed annual appreciation
    rate of 8%. This is a demonstration assumption, not a
    validated market forecast.

    **Technology stack**

    Python, Pandas, NumPy, Scikit-learn, MLflow, Joblib,
    Plotly and Streamlit.
    """)

    st.write(
        "Dataset source:",
        data_source
    )

    st.write(
        "Number of model features:",
        len(feature_cols)
    )

    st.info(
        "For real-world use, replace the synthetic demo dataset "
        "with verified housing data and retrain the models."
    )
