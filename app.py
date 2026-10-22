import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SmartCart", page_icon="🛒", layout="wide")
st.title("🛒 SmartCart")
st.subheader("Customer Segmentation & Intelligence Dashboard")

@st.cache_data
def load_data():
    lines = Path("smartcart_customers.csv").read_text(encoding="utf-8-sig").splitlines()
    cleaned = "\n".join(
        line[1:-1] if line.startswith('"') and line.endswith('"') else line
        for line in lines
    )
    return pd.read_csv(StringIO(cleaned))

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.sidebar.header("Settings")
k = st.sidebar.slider("Number of customer segments", 2, 8, 4)

spending_cols = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
purchase_cols = ["NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases"]

for col in ["Income"] + spending_cols + purchase_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Income"] + spending_cols + purchase_cols).copy()
df["Total_Spending"] = df[spending_cols].sum(axis=1)
df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

features = df[["Income", "Total_Spending", "Total_Purchases"]]
scaled = StandardScaler().fit_transform(features)
df["Segment"] = KMeans(n_clusters=min(k, len(df)), random_state=42, n_init=10).fit_predict(scaled).astype(str)

a, b, c = st.columns(3)
a.metric("Customers", f"{len(df):,}")
b.metric("Average Income", f"₹{df['Income'].mean():,.0f}")
c.metric("Average Spending", f"₹{df['Total_Spending'].mean():,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Income vs Spending")
    fig = px.scatter(df, x="Income", y="Total_Spending", color="Segment",
                     hover_data=["ID"], title="Customer Segments")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Customers per Segment")
    counts = df["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
    st.plotly_chart(px.bar(counts, x="Segment", y="Customers", color="Segment"),
                    use_container_width=True)

st.subheader("Spending by Product Category")
spending = df[spending_cols].sum().rename({
    "MntWines": "Wines", "MntFruits": "Fruits",
    "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
    "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"
}).rename_axis("Category").reset_index(name="Total Spending")
st.plotly_chart(px.bar(spending, x="Category", y="Total Spending", color="Category"),
                use_container_width=True)

st.subheader("Segment Summary")
summary = df.groupby("Segment").agg(
    Customers=("ID", "count"),
    Average_Income=("Income", "mean"),
    Average_Spending=("Total_Spending", "mean"),
    Average_Purchases=("Total_Purchases", "mean")
).round(2)
st.dataframe(summary, use_container_width=True)

st.subheader("Customer Records")
st.dataframe(df, use_container_width=True)
st.download_button("Download segmented data", df.to_csv(index=False),
                   file_name="smartcart_segmented_customers.csv", mime="text/csv")
