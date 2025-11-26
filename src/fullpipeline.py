import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import re

# -----------------------------
# 1️⃣ Load and clean data
# -----------------------------
df = pd.read_csv("bank_statement_clean.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['debit'] = df['debit'].fillna(0)
df['credit'] = df['credit'].fillna(0)

# -----------------------------
# 2️⃣ RFM Segmentation
# -----------------------------
snapshot_date = df['date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('account_number').agg({
    'date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'statement_id': 'count',                            # Frequency
    'debit': 'sum'                                     # Monetary (spending)
}).reset_index()

rfm.rename(columns={
    'date': 'Recency',
    'statement_id': 'Frequency',
    'debit': 'Monetary'
}, inplace=True)

def rfm_segment(row):
    if row['Recency'] <= 30 and row['Frequency'] >= 10 and row['Monetary'] > 1000:
        return "Top Spender"
    elif row['Recency'] <= 60 and row['Frequency'] >= 5:
        return "Active"
    elif row['Recency'] > 90:
        return "Dormant"
    else:
        return "Low Activity"

rfm['Segment'] = rfm.apply(rfm_segment, axis=1)
rfm.to_csv("rfm_segmented.csv", index=False)
print("✅ RFM segmentation done")

# -----------------------------
# 3️⃣ Automatic Merchant Detection (TF-IDF + KMeans)
# -----------------------------
df_debit = df[df['debit'] > 0].copy()
df_debit['description'] = df_debit['description'].fillna("UNKNOWN").str.upper()

vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(df_debit['description'])

k = min(10, len(df_debit))
kmeans = KMeans(n_clusters=k, random_state=42)
df_debit['merchant_cluster'] = kmeans.fit_predict(X)

def transaction_category(desc, merchant, note, credit):
    # Combine all text columns
    text = " ".join([
        str(desc).upper(),
        str(merchant).upper() if merchant else "",
        str(note).upper() if note else ""
    ])
    
    # Rule-based categorization
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

# Apply row-wise
df_debit['transaction_category'] = df_debit.apply(
    lambda x: transaction_category(x['description'], x['merchant'], x['note'], x['credit']),
    axis=1
)

# -----------------------------
# 5️⃣ Transaction-Level Anomaly Detection (Isolation Forest)
# -----------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
df_debit['anomaly_flag'] = iso.fit_predict(df_debit[['debit','merchant_cluster']])
df_debit['is_anomaly'] = df_debit['anomaly_flag'].apply(lambda x: True if x==-1 else False)

# -----------------------------
# 6️⃣ Merchant Cluster Summary
# -----------------------------
cluster_summary = df_debit.groupby('merchant_cluster').agg({
    'description': lambda x: x.mode()[0],
    'debit': 'sum',
    'description': 'count'
}).rename(columns={'description':'transaction_count'}).reset_index()

# -----------------------------
# 7️⃣ Save Results
# -----------------------------
df_debit.to_csv("transactions_full_pipeline.csv", index=False)
cluster_summary.to_csv("merchant_summary.csv", index=False)

print("✅ Full pipeline completed!")
print("Merchant cluster summary:")
print(cluster_summary)
print("Anomalies detected:")
print(df_debit[df_debit['is_anomaly']][['date','description','debit','merchant_cluster']])
