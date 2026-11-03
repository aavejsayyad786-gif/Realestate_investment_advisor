
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "india_housing_prices.csv"

if not DATA.exists():
    DATA = ROOT / "data" / "demo_housing_data.csv"

if not DATA.exists():
    raise FileNotFoundError(
        "Dataset not found. Place india_housing_prices.csv "
        "inside the data folder."
    )

df = pd.read_csv(DATA)

print("Dataset:", DATA.name)
print("Original shape:", df.shape)

needed = [
    "State", "City", "Locality", "Property_Type",
    "BHK", "Size_in_SqFt", "Price_in_Lakhs"
]

missing = set(needed) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# --------------------------------------------------
# 2. Data preprocessing
# --------------------------------------------------

df = df.drop_duplicates().copy()

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
    (df["Price_in_Lakhs"] > 0) &
    (df["Size_in_SqFt"] > 0)
].copy()

if len(df) < 10:
    raise ValueError("Dataset must contain at least 10 valid records.")

# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

df["Price_per_SqFt"] = (
    df["Price_in_Lakhs"] * 100000 /
    df["Size_in_SqFt"]
)

if "Year_Built" in df.columns:
    df["Year_Built"] = pd.to_numeric(
        df["Year_Built"], errors="coerce"
    )
    df["Age_of_Property"] = (
        2026 - df["Year_Built"]
    ).clip(lower=0)
else:
    df["Year_Built"] = np.nan
    df["Age_of_Property"] = np.nan

numeric_columns = [
    "BHK", "Nearby_Schools", "Nearby_Hospitals",
    "Floor_No", "Total_Floors", "Parking_Space"
]

for col in numeric_columns:
    if col not in df.columns:
        df[col] = np.nan
    else:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

categorical_columns = [
    "State", "City", "Locality", "Property_Type",
    "Furnished_Status", "Public_Transport_Accessibility",
    "Security", "Amenities", "Facing", "Owner_Type",
    "Availability_Status"
]

for col in categorical_columns:
    if col not in df.columns:
        df[col] = "Unknown"

# --------------------------------------------------
# 4. Create target variables
# --------------------------------------------------

median_price = df["Price_in_Lakhs"].median()

df["Good_Investment"] = (
    df["Price_in_Lakhs"] <= median_price
).astype(int)

# Project brief's example: 8% annual appreciation
growth_rate = 0.08

df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"] *
    (1 + growth_rate) ** 5
)

print("Valid records:", len(df))
print("Investment class distribution:")
print(df["Good_Investment"].value_counts())

# --------------------------------------------------
# 5. Select features
# --------------------------------------------------

features = [
    "State", "City", "Locality", "Property_Type",
    "BHK", "Size_in_SqFt", "Price_in_Lakhs",
    "Price_per_SqFt", "Year_Built", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals",
    "Floor_No", "Total_Floors", "Parking_Space",
    "Furnished_Status", "Public_Transport_Accessibility",
    "Security", "Amenities", "Facing", "Owner_Type",
    "Availability_Status"
]

features = [
    col for col in features if col in df.columns
]

X = df[features]

y_classification = df["Good_Investment"]
y_regression = df["Future_Price_5Y"]

# Use explicit dtype selection to avoid Pandas warnings
categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

print("Number of features:", len(features))

# --------------------------------------------------
# 6. Preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_features
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        )
    ]
)

# --------------------------------------------------
# 7. Model creation
# --------------------------------------------------

classifier = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])

regressor = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1
        )
    )
])

# --------------------------------------------------
# 8. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X,
    y_regression,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# 9. MLflow experiment tracking
# --------------------------------------------------

mlflow.set_experiment(
    "Real_Estate_Investment_Advisor"
)

with mlflow.start_run():

    print("\nTraining classification model...")
    classifier.fit(X_train, y_train)

    print("Training regression model...")
    regressor.fit(Xr_train, yr_train)

    # Predictions
    class_predictions = classifier.predict(X_test)
    price_predictions = regressor.predict(Xr_test)

    # Evaluation metrics
    metrics = {
        "classification_accuracy": accuracy_score(
            y_test, class_predictions
        ),
        "classification_f1": f1_score(
            y_test,
            class_predictions,
            zero_division=0
        ),
        "regression_mae": mean_absolute_error(
            yr_test, price_predictions
        ),
        "regression_rmse": np.sqrt(
            mean_squared_error(
                yr_test, price_predictions
            )
        ),
        "regression_r2": r2_score(
            yr_test, price_predictions
        )
    }

    mlflow.log_params({
        "classifier": "RandomForestClassifier",
        "regressor": "RandomForestRegressor",
        "n_estimators": 250,
        "test_size": 0.2,
        "growth_rate": growth_rate
    })

    mlflow.log_metrics(metrics)

    # FIX: Use cloudpickle instead of the default skops format
    mlflow.sklearn.log_model(
        classifier,
        name="investment_classifier",
        serialization_format="cloudpickle"
    )

    mlflow.sklearn.log_model(
        regressor,
        name="future_price_regressor",
        serialization_format="cloudpickle"
    )

    print("\nModel Evaluation Results")
    print("-" * 40)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# --------------------------------------------------
# 10. Save trained models
# --------------------------------------------------

model_dir = ROOT / "models"
model_dir.mkdir(exist_ok=True)

joblib.dump(
    classifier,
    model_dir / "investment_classifier.joblib"
)

joblib.dump(
    regressor,
    model_dir / "future_price_regressor.joblib"
)

with open(
    model_dir / "feature_columns.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(features, file, indent=2)

print("\nTraining completed successfully!")
print("Models saved in:", model_dir)
