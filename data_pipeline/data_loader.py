import pandas as pd
import os

from data_pipeline.data_cleaning import clean_data
from data_pipeline.feature_engineering import (
    generate_customer_features,
    customer_segmentation,
    rfm_scoring,
    create_churn_label
)

DATA_PATH = "datasets/online_retail_raw.xlsx"
CACHE_PATH = "datasets/processed_data.csv"


# --------------------------
# PRIORITY SCORE FUNCTION
# --------------------------
def add_priority_score(df):
    df["Monetary"] = df["Monetary"].fillna(0)
    df["Frequency"] = df["Frequency"].fillna(0)
    df["Recency"] = df["Recency"].fillna(0)

    df["Priority_Score"] = (
        df["Monetary"] * 0.5 +
        df["Frequency"] * 0.3 -
        df["Recency"] * 0.2
    )

    return df


# --------------------------
# PRIORITY LEVEL FUNCTION
# --------------------------
def add_priority_level(df):
    def get_priority_level(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        else:
            return "Low"

    df["Priority_Level"] = df["Priority_Score"].apply(get_priority_level)

    return df


# --------------------------
# MAIN LOADER
# --------------------------
def load_data():

    # --------------------------
    # LOAD FROM CACHE
    # --------------------------
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)

        if "CustomerID" not in df.columns:
            df["CustomerID"] = range(1000, 1000 + len(df))

        # ✅ Always recompute (important)
        df = add_priority_score(df)
        df = add_priority_level(df)

        return df

    print("🔄 Processing data from scratch...")

    # --------------------------
    # LOAD RAW DATA
    # --------------------------
    df = pd.read_excel(DATA_PATH)

    # --------------------------
    # PIPELINE
    # --------------------------
    df = clean_data(df)
    df = generate_customer_features(df)
    df = customer_segmentation(df)
    df = rfm_scoring(df)
    df = create_churn_label(df)

    # --------------------------
    # ADD CUSTOMER ID
    # --------------------------
    if "CustomerID" not in df.columns:
        df["CustomerID"] = range(1000, 1000 + len(df))

    # --------------------------
    # ADD PRIORITY FEATURES
    # --------------------------
    df = add_priority_score(df)
    df = add_priority_level(df)

    # --------------------------
    # SAVE PROCESSED DATA
    # --------------------------
    df.to_csv(CACHE_PATH, index=False)

    return df