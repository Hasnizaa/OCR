import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import re

st.set_page_config(page_title="Bank Statement Analysis", layout="wide")
st.title("💳 Bank Statement Analysis Dashboard")

# -----------------------------
# 1️⃣ Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload your bank statement CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['debit'] = df['debit'].fillna(0)
    df['credit'] = df['credit'].fillna(0)
    st.success("CSV loaded successfully!")

    # -----------------------------
    # 2️⃣ Automatic Merchant Detection
    # -----------------------------
    df_debit = df[df['debit'] > 0].copy()
    df_debit['description'] = df_debit['description'].fillna("UNKNOWN").str.upper()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform(df_debit['description'])
    k = min(10, len(df_debit))
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_debit['merchant_cluster'] = kmeans.fit_predict(X)

    # -----------------------------
    # 3️⃣ Transaction Category Labeling
    # -----------------------------
    def transaction_category(desc, merchant, note, credit):
        text = " ".join([
            str(desc).upper(),
            str(merchant).upper() if merchant else "",
            str(note).upper() if note else ""
        ])
        if re.search(r'MAXIS|CELCOM|TNB|TM|TELEKOM|JABATAN|WATER|INSURANCE', text):
            return "BILL PAYMENT"
        elif re.search(r'SHOPEE|AEON|FOODPANDA', text):
            return "SHOPPING"
        elif re.search(r'MAKAN|KFC|MCD|MC DONALD|CAFE|NASI|RESTORAN', text):
            return "FOOD & BEVERAGE"
        elif re.search(r'PETRONAS|PETRON|SHELL', text):
            return "FUEL"
        elif re.search(r'ATM|CASH', text):
            return "CASH WITHDRAWAL"
        elif re.search(r'LOAN|HP', text):
            return "LOAN PAYMENT"
        elif credit > 0:
            return "MONEY IN"
        else:
            return "OTHERS"

    df_debit['transaction_category'] = df_debit.apply(
        lambda x: transaction_category(
            x['description'], x.get('merchant', ''), x.get('note', ''), x['credit']
        ),
        axis=1
    )

    # -----------------------------
    # 4️⃣ Transaction-Level Anomaly Detection
    # -----------------------------
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_debit['anomaly_flag'] = iso.fit_predict(df_debit[['debit','merchant_cluster']])
    df_debit['is_anomaly'] = df_debit['anomaly_flag'].apply(lambda x: True if x==-1 else False)

    # -----------------------------
    # 5️⃣ Merchant Cluster Summary (fixed)
    # -----------------------------
    merchant_summary = df_debit.groupby('merchant_cluster').agg(
        merchant_name=('description', lambda x: x.mode()[0]),
        debit=('debit', 'sum'),
        transaction_count=('description', 'count')
    ).reset_index()

    # -----------------------------
    # 6️⃣ Visualizations
    # -----------------------------
    # -----------------------------
# 5️⃣ Spending by Transaction Category
# -----------------------------
    st.subheader("📊 Spending by Transaction Category")
    category_summary = df_debit.groupby('transaction_category')['debit'].sum().reset_index()
    fig1 = px.pie(
        category_summary,
        names='transaction_category',
        values='debit',
        title='Spending Distribution by Category'
    )
    st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# 6️⃣ Top Merchant Clusters
# -----------------------------
    st.subheader("💰 Top Merchant Clusters")
    merchant_summary = df_debit.groupby('merchant_cluster').agg(
        merchant_name=('merchant', lambda x: x.mode()[0] if not x.mode().empty else "UNKNOWN"),
        debit=('debit', 'sum'),
        transaction_count=('merchant', 'count')
    ).reset_index()

    fig2 = px.bar(
        merchant_summary,
        x='merchant_cluster',
        y='debit',
        text='merchant_name',  # shows the most common merchant in each cluster
        title='Top Merchant Clusters by Spending'
    )
    st.plotly_chart(fig2, use_container_width=True)


    st.subheader("⚠️ Detected Anomalies")
    anomalies = df_debit[df_debit['is_anomaly']]
    st.dataframe(anomalies[['date','description','debit','merchant_cluster','transaction_category']])

    st.success("Dashboard analysis completed!")
