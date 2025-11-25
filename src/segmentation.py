import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

sns.set(style="whitegrid")


# --------------------- LOAD JSON ------------------------
def load_from_json_folder(folder):
    """Load multiple OCR JSON files into one DataFrame."""
    rows = []

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle list or dict JSON
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                print(f"⚠️ Skipped {fname} — list but no dict")
                continue

        customer = data.get("customer_info", {})
        trans = data.get("transactions", [])

        for t in trans:
            rows.append({
                "file": fname,
                "name": customer.get("name"),
                "ic": customer.get("ic"),
                "account_no": customer.get("account_no"),
                "date": t.get("date"),
                "desc": t.get("description"),
                "amount": t.get("amount"),
                "balance": t.get("balance")
            })

    df = pd.DataFrame(rows)

    # --- Ensure customer_id exists ---
    df["customer_id"] = (
        df["name"].fillna("unknown") + "_" + df["ic"].fillna("0000")
    )
    df.loc[df["customer_id"] == "unknown_0000", "customer_id"] = df["account_no"]

    return df


# ---------------- CLEAN + FEATURE ENGINEER ----------------
def clean_and_enrich(df):
    df = df.copy()

    # Convert date safely
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert amount to numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Optional: convert balance to numeric
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0)

    return df


def build_customer_features(df):
    """Aggregate transactions per customer"""
    agg = df.groupby("customer_id").agg(
        amount_mean=("amount", "mean"),
        amount_median=("amount", "median"),
        amount_std=("amount", "std"),
        amount_min=("amount", "min"),
        amount_max=("amount", "max"),
        total_spend=("amount", "sum"),
        txn_count=("amount", "count"),
        last_date=("date", "max"),
    ).reset_index()

    agg["days_since_last"] = (
        (pd.Timestamp.now() - agg["last_date"]).dt.days.fillna(9999)
    )
    agg = agg.fillna(0)
    return agg


# ---------------------- K SELECTION -----------------------
def choose_k(Xs, k_min=2, k_max=8):
    best_k = 3
    best_score = -1

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        try:
            score = silhouette_score(Xs, labels)
        except:
            score = -1
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


# ---------------------- MAIN PIPELINE ---------------------
def pipeline(json_folder, outdir="results_segmentation"):
    os.makedirs(outdir, exist_ok=True)

    print("📂 Loading OCR JSON…")
    df = load_from_json_folder(json_folder)

    print("🧹 Cleaning & feature engineering…")
    df = clean_and_enrich(df)

    print("🔹 Building customer features…")
    cust = build_customer_features(df)

    feature_cols = [c for c in cust.columns if c not in ["customer_id", "last_date"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(cust[feature_cols].values)

    print("🔸 Selecting best K…")
    k = choose_k(Xs)
    print(f"✅ Selected K = {k}")

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    cust["cluster"] = labels

    # Save results
    cust.to_csv(os.path.join(outdir, "customer_clusters.csv"), index=False)
    joblib.dump(km, os.path.join(outdir, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.joblib"))

    # PCA Visualization
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xs)
    plt.figure(figsize=(7, 6))
    import seaborn as sns
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=labels, palette="tab10")
    plt.title("Customer Segmentation (PCA)")
    plt.savefig(os.path.join(outdir, "clusters_pca.png"))
    plt.close()

    print(f"🎉 Segmentation completed! Results saved in {outdir}")


# --------------------------- CLI --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output_json", help="Folder with JSON files")
    parser.add_argument("--outdir", default="results_segmentation", help="Output folder")
    args = parser.parse_args()

    pipeline(args.input, args.outdir)
