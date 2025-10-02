# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess(filepath):
    # === Load dataset ===
    df = pd.read_csv(filepath)

    # Drop identifiers (to avoid data leakage)
    df = df.drop(["encounter_id", "patient_nbr"], axis=1, errors="ignore")

    # === Define target variable ===
    if "readmitted" in df.columns:
        df = df[df["readmitted"] != "30"]  # remove ambiguous category
        df["target"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
        df = df.drop("readmitted", axis=1)

    # === Handle missing values ===
    df.replace("?", np.nan, inplace=True)
    df.fillna("Unknown", inplace=True)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # === Build preprocessing pipeline ===
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor
