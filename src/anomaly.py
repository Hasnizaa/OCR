import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import seaborn as sns

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

    # Convert balance to numeric
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


# ---------------- ANOMALY DETECTION -----------------------
def detect_anomalies(df, contamination=0.05):
    """Detect anomalies per customer using Isolation Forest"""
    feature_cols = [c for c in df.columns if c not in ["customer_id", "last_date"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[feature_cols].values)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    df["anomaly"] = iso.fit_predict(Xs)
    # Convert: -1 = anomaly, 1 = normal -> True/False
    df["is_anomaly"] = df["anomaly"] == -1

    return df, iso, scaler


# ---------------------- MAIN PIPELINE ---------------------
def pipeline(json_folder, outdir="results_anomaly", contamination=0.05):
    os.makedirs(outdir, exist_ok=True)

    print("📂 Loading OCR JSON…")
    df = load_from_json_folder(json_folder)

    print("🧹 Cleaning & feature engineering…")
    df = clean_and_enrich(df)

    print("🔹 Building customer features…")
    cust = build_customer_features(df)

    print("⚡ Detecting anomalies…")
    cust, iso_model, scaler_model = detect_anomalies(cust, contamination=contamination)

    # Save results
    cust.to_csv(os.path.join(outdir, "customer_anomalies.csv"), index=False)
    joblib.dump(iso_model, os.path.join(outdir, "isolation_forest_model.joblib"))
    joblib.dump(scaler_model, os.path.join(outdir, "scaler.joblib"))

    # Visualization (PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(scaler_model.transform(cust.drop(columns=["customer_id", "last_date", "anomaly", "is_anomaly"])))

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=cust["is_anomaly"], palette={True:"red", False:"green"})
    plt.title("Customer Anomalies (PCA)")
    plt.savefig(os.path.join(outdir, "anomalies_pca.png"))
    plt.close()

    print(f"🎉 Anomaly detection completed! Results saved in {outdir}")


# --------------------------- CLI --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output_json", help="Folder with JSON files")
    parser.add_argument("--outdir", default="results_anomaly", help="Output folder")
    parser.add_argument("--contamination", type=float, default=0.05, help="Proportion of anomalies")
    args = parser.parse_args()

    pipeline(args.input, args.outdir, args.contamination)
