import sys
import os
import matplotlib.pyplot as plt

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from security.auth_system import login

import streamlit as st

# --------------------------
# AUTH SYSTEM
# --------------------------
# -----------------------------

# LOGIN CONTROL
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.warning("🔒 Please login to access dashboard")
    st.stop()

# 👇 ADD LOGOUT HERE
st.sidebar.markdown("### 🔐 Session")
st.sidebar.success("✅ Logged in as Admin")

if st.sidebar.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# -----------------------------
# DASHBOARD STARTS HERE
# -----------------------------


import requests
import pandas as pd


from data_pipeline.data_loader import load_data

# --------------------------
# PAGE SETTINGS
# --------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def get_processed_data():
    return load_data()

with st.spinner("Loading data..."):
    full_df = get_processed_data()   # 🔥 ORIGINAL DATA
    df = full_df.copy()              # 🔥 FILTERED VERSION

# --------------------------
# 🎯 FILTER SECTION
# --------------------------
st.sidebar.header("🎯 Filters")

priority_filter = st.sidebar.selectbox(
    "Select Priority Level",
    ["All", "High", "Medium", "Low"]
)

if priority_filter != "All":
    df = df[df["Priority_Level"] == priority_filter]


# --------------------------
# METRICS
# --------------------------
col1, col2, col3 = st.columns(3)

total_customers = len(df)

if "Churn" in df.columns:
    churn_count = int(df["Churn"].sum())
    churn_rate = round((churn_count / total_customers) * 100, 2)
else:
    churn_count = 0
    churn_rate = 0

col1.metric("Total Customers", total_customers)
col2.metric("Churn Customers", churn_count)
col3.metric("Churn Rate", f"{churn_rate}%")

# --------------------------
# 🧠 SMART INSIGHTS
# --------------------------
# --------------------------
st.subheader("🧠 Key Insights")

if "Churn" in df.columns:

    churn_rate_val = round((df["Churn"].sum() / len(df)) * 100, 2)

    avg_recency = int(df["Recency"].mean())
    avg_frequency = round(df["Frequency"].mean(), 2)
    avg_monetary = round(df["Monetary"].mean(), 2)

    # 💰 BUSINESS METRICS
    revenue_at_risk = df[df["Churn"] == 1]["Monetary"].sum()

    if "Priority_Level" in df.columns:
        high_risk_revenue = full_df[
            (full_df["Churn"] == 1) & (full_df["Priority_Level"] == "High")
        ]["Monetary"].sum()
    else:
        high_risk_revenue = 0

    # 📊 Layout (Improved UI)
    col1, col2, col3 = st.columns(3)

    # LEFT PANEL
    with col1:
        st.info(f"📉 Overall churn rate is **{churn_rate_val}%**")

        if churn_rate_val > 30:
            st.warning("⚠️ High churn detected → retention strategy needed")
        else:
            st.success("✅ Churn is under control")

    # MIDDLE PANEL
    with col2:
        st.write(f"🕒 Avg Recency: {avg_recency} days")
        st.write(f"🔁 Avg Frequency: {avg_frequency}")
        st.write(f"💰 Avg Monetary: {avg_monetary}")

    # RIGHT PANEL (🔥 BUSINESS VALUE)
    with col3:
        st.metric(
            label="💰 Revenue at Risk",
            value=f"${round(revenue_at_risk, 2)}"
        )

        st.metric(
            label="🔥 High Risk Revenue",
            value=f"${round(high_risk_revenue, 2)}"
        )

# --------------------------
# SIDEBAR INPUTS
# --------------------------
st.sidebar.header("Enter Customer Details")

recency = st.sidebar.number_input("Recency (days)", min_value=0, value=100)
frequency = st.sidebar.number_input("Frequency", min_value=1, value=5)
monetary = st.sidebar.number_input("Monetary Value", min_value=0.0, value=500.0)

# --------------------------
# --------------------------
# 🚀 PREDICT BUTTON (FIXED)
# --------------------------
predict_clicked = st.sidebar.button("Predict")

if predict_clicked:

    url = "http://127.0.0.1:8000/predict"

    payload = {
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()

            # --------------------------
            # Prediction result
            # --------------------------
            if result["prediction"] == "Churn Likely":
                st.error(f"🚨 {result['prediction']}")
            else:
                st.success(f"✅ {result['prediction']}")

            # Confidence
            st.metric("Confidence", f"{result['confidence']}%")

            # Estimated Loss
            estimated_loss = monetary * (result["confidence"] / 100)
            st.metric("💰 Estimated Loss (This Customer)", f"${estimated_loss:,.2f}")

            # Risk alert
            if estimated_loss > 1000:
                st.error("🔥 High value customer at risk! Take immediate action")
            elif estimated_loss > 200:
                st.warning("⚠️ Medium value risk customer")
            else:
                st.info("ℹ️ Low impact customer")

            # Risk Level
            st.markdown("## 🚦 Risk Level")

            if result["risk_level"] == "High Risk":
                st.error("🔴 HIGH RISK")
            elif result["risk_level"] == "Medium Risk":
                st.warning("🟠 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

            # --------------------------
            # 💡 Retention Suggestion
            # --------------------------
            st.markdown("## 💡 Recommended Action")

            suggestion = result.get("suggestion", "")

            if "discount" in suggestion.lower():
                st.error(f"🔥 {suggestion}")
            elif "email" in suggestion.lower():
                st.warning(f"📧 {suggestion}")
            else:
                st.success(f"✅ {suggestion}")

        else:
            st.error(f"❌ API Error: {response.status_code}")

    except Exception as e:
        st.error(f"❌ Connection Error: {e}")


# --------------------------
# SEGMENT ANALYSIS
# --------------------------
st.subheader("📊 Customer Segments Distribution")

if "Segment" in df.columns:

    segment_counts = df["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    chart_type = st.selectbox("Choose Chart Type", ["Pie Chart", "Bar Chart"])

    fig, ax = plt.subplots(figsize=(6, 4))

    if chart_type == "Pie Chart":
        ax.pie(segment_counts["Count"], labels=segment_counts["Segment"], autopct="%1.1f%%")
        ax.set_title("Customer Segments (Pie)")
    else:
        ax.bar(segment_counts["Segment"], segment_counts["Count"])
        ax.set_title("Customer Segments (Bar)")
        plt.xticks(rotation=30)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📌 Segment Insight")
    top_segment = df["Segment"].value_counts().idxmax()
    top_count = df["Segment"].value_counts().max()
    st.success(f"🏆 Most customers belong to: **{top_segment}** ({top_count})")

else:
    st.error("❌ Segment column missing")

# --------------------------
st.subheader("📊 Priority Level Distribution")

if "Priority_Level" in df.columns:

    priority_counts = df["Priority_Level"].value_counts()

    col1, col2 = st.columns(2)

    # BAR CHART
    with col1:
        st.write("📊 Bar Chart")

        fig, ax = plt.subplots()
        ax.bar(priority_counts.index, priority_counts.values)
        ax.set_xlabel("Priority Level")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Priority Distribution")

        st.pyplot(fig)

    # PIE CHART
    with col2:
        st.write("🥧 Pie Chart")

        fig2, ax2 = plt.subplots()
        ax2.pie(
            priority_counts.values,
            labels=priority_counts.index,
            autopct="%1.1f%%"
        )
        ax2.set_title("Priority Distribution")

        st.pyplot(fig2)

# --------------------------
# CHURN ANALYSIS
# --------------------------
st.subheader("📉 Churn Analysis")

if "Churn" in df.columns:

    col1, col2 = st.columns(2)

    with col1:
        fig2, ax2 = plt.subplots()
        churn_counts = df["Churn"].value_counts().sort_index()
        ax2.bar(["Active", "Churn"], churn_counts.values)
        ax2.set_title("Churn Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots()
        df.groupby("Churn")["Recency"].mean().plot(kind="bar", ax=ax3)
        ax3.set_title("Avg Recency by Churn")
        ax3.set_xticklabels(["Active", "Churn"], rotation=0)
        plt.tight_layout()
        st.pyplot(fig3)

else:
    st.warning("Churn data not available")

# --------------------------
# FEATURE IMPORTANCE
# --------------------------
st.subheader("🔍 Feature Importance (Why Customers Churn)")

try:
    response = requests.get("http://127.0.0.1:8000/feature-importance")

    if response.status_code == 200:
        importance_data = response.json()

        fig4, ax4 = plt.subplots()
        ax4.bar(importance_data.keys(), importance_data.values())
        ax4.set_title("Feature Importance")
        plt.tight_layout()

        st.pyplot(fig4)
    else:
        st.warning("Feature importance API not working")

except:
    st.warning("API not connected — showing fallback")

    fallback = {"Recency": 0.5, "Frequency": 0.3, "Monetary": 0.2}

    fig4, ax4 = plt.subplots()
    ax4.bar(fallback.keys(), fallback.values())
    ax4.set_title("Feature Importance (Fallback)")
    plt.tight_layout()

    st.pyplot(fig4)

# --------------------------
# HIGH RISK CUSTOMERS
# --------------------------
st.subheader("⚠️ High Risk Customers")

if "Churn" in df.columns:
    high_risk = df[df["Churn"] == 1].sort_values(by="Recency", ascending=False).head(10)
    st.dataframe(high_risk)
else:
    st.warning("No churn data available")

st.subheader("📋 Full High-Risk Customer List")

if "Churn" in df.columns:
    full_high_risk = df[df["Churn"] == 1]

    search = st.text_input("Search Customer (by segment or value)")

    if search:
        full_high_risk = full_high_risk[
            full_high_risk.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)
        ]

    st.dataframe(full_high_risk, width="stretch")

    st.download_button(
        "Download High Risk Customers",
        data=full_high_risk.to_csv(index=False),
        file_name="high_risk_customers.csv"
    )

# -----------------------------
# -----------------------------
# PRIORITY CUSTOMERS
# -----------------------------
# -----------------------------
# PRIORITY CUSTOMERS (SAFE VERSION)
# -----------------------------
st.subheader("🔥 Top Customers to Save")

# Safe sorting
if "Priority_Score" in df.columns:
    top_customers = df[df["Churn"] == 1].sort_values(
        by="Priority_Score", ascending=False
    ).head(10)
else:
    top_customers = df[df["Churn"] == 1].head(10)

# Safe column selection
columns_to_show = ["CustomerID", "Recency", "Frequency", "Monetary"]

if "Priority_Score" in df.columns:
    columns_to_show.append("Priority_Score")

if "Priority_Level" in df.columns:
    columns_to_show.append("Priority_Level")

# Display
st.dataframe(top_customers[columns_to_show], width="stretch")

# -----------------------------
# 🎯 SMART ACTIONS (ADD HERE)
# -----------------------------
st.subheader("🎯 Smart Action Recommendations")

for _, row in top_customers.iterrows():

    level = row["Priority_Level"]

    if level == "High":
        st.error(f"🔴 Customer {row['CustomerID']} → HIGH PRIORITY")
        st.write("👉 Give 20% discount immediately")
        st.write("👉 Call customer personally")
        st.write("👉 Offer premium support")

    elif level == "Medium":
        st.warning(f"🟠 Customer {row['CustomerID']} → MEDIUM PRIORITY")
        st.write("👉 Send personalized email")
        st.write("👉 Offer loyalty points")

    else:
        st.success(f"🟢 Customer {row['CustomerID']} → LOW PRIORITY")
        st.write("👉 Monitor activity")
        st.write("👉 Send occasional offers")

    st.write("---")
# --------------------------
# --------------------------
# ACTION PLAN PER CUSTOMER
# --------------------------
st.subheader("🎯 Action Plan for High Risk Customers")

if "Churn" in df.columns:

    sample = df[df["Churn"] == 1].head(5)

    for i, row in sample.iterrows():

        # ✅ Correct indentation + safe label
        customer_label = row["CustomerID"] if "CustomerID" in df.columns else i

        st.markdown(f"### 👤 Customer: {customer_label}")

        if row.get("Recency", 0) > 200:
            st.write("👉 Send re-engagement email")

        if row.get("Frequency", 0) <= 2:
            st.write("👉 Offer loyalty reward")

        if row.get("Monetary", 0) < 500:
            st.write("👉 Give discount offer")

        st.write("---")

# --------------------------
# RETENTION STRATEGIES
# --------------------------
st.subheader("🎯 Recommended Retention Strategies")

if "Churn" in df.columns:

    high_risk_count = df[df["Churn"] == 1].shape[0]

    if high_risk_count > 0:

        st.warning(f"⚠️ {high_risk_count} customers are at risk of churn")

        st.markdown("### 📌 General Strategies:")
        st.write("💡 Offer discounts to inactive customers")
        st.write("📧 Send re-engagement emails")
        st.write("🎁 Provide loyalty rewards")
        st.write("📞 Follow up with high-value customers")

        st.markdown("### 🎯 Smart Recommendations:")

        if df["Recency"].mean() > df["Recency"].quantile(0.75):
            st.write("👉 High inactivity → Launch campaigns")

        if df["Frequency"].mean() < df["Frequency"].quantile(0.25):
            st.write("👉 Low frequency → Loyalty programs")

        if df["Monetary"].mean() < df["Monetary"].quantile(0.25):
            st.write("👉 Low spending → Discounts")

    else:
        st.success("✅ No major churn risk")

# --------------------------
# DOWNLOAD
# --------------------------
st.download_button(
    label="Download Processed Data",
    data=df.to_csv(index=False),
    file_name="churn_data.csv",
    mime="text/csv"
)