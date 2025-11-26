import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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

rfm.to_csv("rfm_segmented_single_account.csv", index=False)
print("RFM segmentation done:")
print(rfm)

# -----------------------------
# 3️⃣ Automatic Merchant Detection
# -----------------------------
# Only consider debit transactions for merchant clustering
df_debit = df[df['debit'] > 0].copy()
df_debit['description'] = df_debit['description'].fillna("UNKNOWN").str.upper()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(df_debit['description'])

# KMeans clustering
k = min(10, len(df_debit))  # number of clusters <= number of transactions
kmeans = KMeans(n_clusters=k, random_state=42)
df_debit['merchant_cluster'] = kmeans.fit_predict(X)

# Aggregate cluster info
cluster_summary = df_debit.groupby('merchant_cluster').agg({
    'description': lambda x: x.mode()[0],  # most common description in cluster
    'debit': 'sum',
    'description': 'count'
}).rename(columns={'description':'transaction_count'}).reset_index()

df_debit.to_csv("auto_merchant_clusters.csv", index=False)
cluster_summary.to_csv("merchant_summary.csv", index=False)

print("Automatic merchant detection done:")
print(cluster_summary)
