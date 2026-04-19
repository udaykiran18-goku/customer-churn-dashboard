import pandas as pd
import numpy as np


# ===============================
# 🧠 FEATURE ENGINEERING (RFM)
# ===============================
def generate_customer_features(data):

    print("\nStarting Feature Engineering...")

    # Convert date safely
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")

    # Remove rows with missing important values
    data = data.dropna(subset=["CustomerID", "InvoiceDate"])

    # Create total price
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    # Snapshot date (latest date in dataset)
    snapshot_date = data["InvoiceDate"].max()

    # RFM aggregation
    rfm = data.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    print("Customer Features Created")
    print("Total Customers:", rfm.shape[0])

    return rfm


# ===============================
# 🎯 CUSTOMER SEGMENTATION (FAST)
# ===============================
def customer_segmentation(rfm):

    print("\nStarting Customer Segmentation...")

    conditions = [
        (rfm["Recency"] <= 30) & (rfm["Frequency"] >= 5),
        (rfm["Recency"] <= 60) & (rfm["Frequency"] >= 3),
        (rfm["Recency"] <= 120),
        (rfm["Recency"] <= 180),
        (rfm["Recency"] <= 365)
    ]

    choices = ["VIP", "Loyal", "Regular", "At Risk", "Churn Likely"]

    rfm["Segment"] = np.select(conditions, choices, default="Lost")

    print("Segmentation Complete\n")
    print(rfm["Segment"].value_counts())

    return rfm


# ===============================
# 📊 RFM SCORING (SAFE VERSION)
# ===============================
def rfm_scoring(rfm):

    print("\nStarting RFM Scoring...")

    try:
        # Recency (lower = better → reverse)
        rfm["R_Score"] = pd.qcut(
            rfm["Recency"],
            q=5,
            labels=[5, 4, 3, 2, 1],
            duplicates="drop"
        )

        # Frequency (higher = better)
        rfm["F_Score"] = pd.qcut(
            rfm["Frequency"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        )

        # Monetary (higher = better)
        rfm["M_Score"] = pd.qcut(
            rfm["Monetary"],
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        )

    except Exception as e:
        print("⚠️ qcut failed, using fallback scoring:", e)

        # Fallback scoring (manual bins)
        rfm["R_Score"] = pd.cut(rfm["Recency"], bins=5, labels=[5,4,3,2,1])
        rfm["F_Score"] = pd.cut(rfm["Frequency"], bins=5, labels=[1,2,3,4,5])
        rfm["M_Score"] = pd.cut(rfm["Monetary"], bins=5, labels=[1,2,3,4,5])

    # Combine scores
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) +
        rfm["F_Score"].astype(str) +
        rfm["M_Score"].astype(str)
    )

    print("RFM Scoring Complete\n")
    print(rfm.head())

    return rfm


# ===============================
# 🔥 CHURN LABEL CREATION
# ===============================
def create_churn_label(rfm):

    print("\nCreating Churn Labels...")

    # Simple rule: inactive > 180 days = churn
    rfm["Churn"] = (rfm["Recency"] > 180).astype(int)

    print("Churn Label Created\n")
    print(rfm["Churn"].value_counts())

    return rfm