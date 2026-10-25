"""
House Price Prediction App - King County, Washington (Seattle area)

Developer : Shaiza Butt (FA24-BED-066)
Course    : Machine Learning - Assignment 3
GitHub    : https://github.com/ShaizaButt

The app loads the best model found by 10-fold cross validation in
House_price_prediction_kfold.ipynb (saved as model.pkl). The saved file is a
complete scikit-learn pipeline (imputer + scaler + one-hot encoder + algorithm),
so the raw house details typed in the form can be passed to it directly.

Run locally:   streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_FILE = APP_DIR / "model.pkl"
DATA_FILE = APP_DIR / "kc_house_data.csv"

# Mean 10-fold cross-validation RMSE of the saved model (Step 10.1 of the notebook)
CV_RMSE = 116745

# Column order must be identical to the one used for training in the notebook
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15",
]
CATEGORICAL_FEATURES = ["zipcode"]

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_reference_data():
    """Same cleaning as Step 3.2 of the notebook; used for valid ranges and typical values."""
    df = pd.read_csv(DATA_FILE).drop(columns=["id", "date"])
    df["zipcode"] = df["zipcode"].astype(str)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[(df["bathrooms"] > 0) & (df["bedrooms"] != 33)].reset_index(drop=True)
    return df


st.title("🏠 House Price Predictor")
st.write(
    "Estimate the sale price of a house in **King County, Washington** (the Seattle area). "
    "Enter the details below and click **Predict price**."
)

try:
    model = load_model()
    data = load_reference_data()
except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}. "
        "Make sure `model.pkl` and `kc_house_data.csv` are in the same folder as `app.py`."
    )
    st.stop()

zip_codes = sorted(data["zipcode"].unique())
zip_typical = data.groupby("zipcode")[["lat", "long", "sqft_living15", "sqft_lot15"]].median()
floor_options = sorted(data["floors"].unique().tolist())
renovation_years = data.loc[data["yr_renovated"] > 0, "yr_renovated"]


def low(column):
    return data[column].min()


def high(column):
    return data[column].max()


# ---------------------------------------------------------------- 1. location and size
st.subheader("1. Location and size")
col1, col2 = st.columns(2)

zipcode = col1.selectbox("ZIP code", zip_codes, index=zip_codes.index("98052"))
sqft_living = col2.number_input(
    "Living area (sq ft)", min_value=int(low("sqft_living")), max_value=int(high("sqft_living")),
    value=int(data["sqft_living"].median()), step=50,
)
bedrooms = col1.number_input(
    "Bedrooms", min_value=int(low("bedrooms")), max_value=int(high("bedrooms")),
    value=int(data["bedrooms"].median()), step=1,
)
bathrooms = col2.number_input(
    "Bathrooms", min_value=float(low("bathrooms")), max_value=float(high("bathrooms")),
    value=float(data["bathrooms"].median()), step=0.25,
)
floors = col1.selectbox("Floors", floor_options, index=floor_options.index(1.5) if 1.5 in floor_options else 0)
yr_built = col2.number_input(
    "Year built", min_value=int(low("yr_built")), max_value=int(high("yr_built")),
    value=int(data["yr_built"].median()), step=1,
)

# ---------------------------------------------------------------- 2. quality and surroundings
st.subheader("2. Quality and surroundings")
grade = st.slider(
    "Grade: construction and design quality (1 = lowest, 13 = highest)",
    int(low("grade")), int(high("grade")), int(data["grade"].median()),
)
condition = st.slider(
    "Condition (1 = poor, 5 = very good)",
    int(low("condition")), int(high("condition")), int(data["condition"].median()),
)
view = st.slider("View (0 = no view, 4 = excellent view)", int(low("view")), int(high("view")), 0)

col3, col4 = st.columns(2)
waterfront = col3.checkbox("Waterfront property")
renovated = col4.checkbox("The house has been renovated")
if renovated:
    yr_renovated = int(st.number_input(
        "Year of last renovation", min_value=int(renovation_years.min()),
        max_value=int(renovation_years.max()), value=2000, step=1,
    ))
else:
    yr_renovated = 0  # the dataset uses 0 for "never renovated"

# ---------------------------------------------------------------- 3. optional details
with st.expander("3. More details (optional - pre-filled with typical values)"):
    st.caption(
        "Location values are pre-filled with the typical (median) values of the selected ZIP code "
        "and update when you change the ZIP code."
    )
    sqft_lot = st.number_input(
        "Lot size (sq ft)", min_value=int(low("sqft_lot")), max_value=int(high("sqft_lot")),
        value=int(data["sqft_lot"].median()), step=100,
    )
    max_basement = max(int(sqft_living) - int(low("sqft_above")), 0)
    sqft_basement = st.number_input(
        "Basement area (sq ft, 0 = no basement)", min_value=0, max_value=max_basement, value=0, step=50,
    )
    sqft_above = int(sqft_living) - int(sqft_basement)
    st.caption(f"Area above ground (calculated): {sqft_above:,} sq ft")

    typical = zip_typical.loc[zipcode]
    lat = st.number_input(
        "Latitude", min_value=float(low("lat")), max_value=float(high("lat")),
        value=float(typical["lat"]), step=0.001, format="%.4f", key=f"lat_{zipcode}",
    )
    long = st.number_input(
        "Longitude", min_value=float(low("long")), max_value=float(high("long")),
        value=float(typical["long"]), step=0.001, format="%.4f", key=f"long_{zipcode}",
    )
    sqft_living15 = st.number_input(
        "Average living area of the 15 nearest houses (sq ft)",
        min_value=int(low("sqft_living15")), max_value=int(high("sqft_living15")),
        value=int(typical["sqft_living15"]), step=50, key=f"living15_{zipcode}",
    )
    sqft_lot15 = st.number_input(
        "Average lot size of the 15 nearest houses (sq ft)",
        min_value=int(low("sqft_lot15")), max_value=int(high("sqft_lot15")),
        value=int(typical["sqft_lot15"]), step=100, key=f"lot15_{zipcode}",
    )

# ---------------------------------------------------------------- prediction
if st.button("Predict price", type="primary"):
    house = {
        "bedrooms": int(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft_living": int(sqft_living),
        "sqft_lot": int(sqft_lot),
        "floors": float(floors),
        "waterfront": int(waterfront),
        "view": int(view),
        "condition": int(condition),
        "grade": int(grade),
        "sqft_above": int(sqft_above),
        "sqft_basement": int(sqft_basement),
        "yr_built": int(yr_built),
        "yr_renovated": int(yr_renovated),
        "lat": float(lat),
        "long": float(long),
        "sqft_living15": int(sqft_living15),
        "sqft_lot15": int(sqft_lot15),
        "zipcode": str(zipcode),
    }
    feature_vector = pd.DataFrame([house])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predicted_price = float(model.predict(feature_vector)[0])

    st.metric("Estimated sale price", f"${predicted_price:,.0f}")
    st.caption(
        f"Typical error of this model: about ±${CV_RMSE:,.0f} "
        "(mean RMSE from 10-fold cross validation). Use it as an estimate, not an exact valuation."
    )
    with st.expander("Show the values sent to the model"):
        st.dataframe(feature_vector.T.rename(columns={0: "value"}).astype(str))

st.divider()
algorithm_name = type(model.named_steps["regressor"]).__name__
st.caption(
    f"Model: {algorithm_name}, trained on {len(data):,} King County house sales "
    "(May 2014 - May 2015). Developed by Shaiza Butt (FA24-BED-066) for Machine Learning Assignment 3. "
    "GitHub: https://github.com/ShaizaButt"
)
