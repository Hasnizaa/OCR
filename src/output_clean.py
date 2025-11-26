import json
import re
import pandas as pd

# -----------------------------
# 1️⃣ Helper functions
# -----------------------------
def clean_str(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("  ", " ").strip()
    s = re.sub(r"[^\x20-\x7E]", "", s)  # Remove non-ASCII
    return s

def clean_number(num):
    if isinstance(num, (int, float)):
        return float(num)
    if not isinstance(num, str):
        return 0.0
    num = num.replace(",", "").strip()
    if num.endswith("-"):  # Handle negative
        try:
            return -float(num[:-1])
        except:
            return 0.0
    try:
        return float(num)
    except:
        return 0.0

def clean_date(date_str):
    if pd.isna(date_str):
        return ""
    match = re.search(r"(\d{2}/\d{2}/\d{2})", date_str)
    if match:
        return match.group(1)
    return ""

# -----------------------------
# 1.1️⃣ Description Correction Dictionary
# -----------------------------
DESCRIPTION_MAP = {
    # SNHO12J variations
    r"S[NH]+O?12[\)\]I3]": "SNHO12J",

    # FPX PAYMENT variations
    r"FPX PAYMENT FR A[fj]": "FPX PAYMENT FR A/",
    r"FPX PAYMENT RR A[j/]": "FPX PAYMENT FR A/",
    r"FPX PAYMENT RR A/": "FPX PAYMENT FR A/",
    r"FPX PAYMENT FR A$": "FPX PAYMENT FR A/",
    r"FPX PAYMENT FR A[\s\.-]": "FPX PAYMENT FR A/",

    # IBK FUND TRANSFER variations
    r"IBK FUND T[RF]+R F?R?A/?C": "IBK FUND TFR FR A/C",
    r"IBK FUND TRR FR A/C": "IBK FUND TFR FR A/C",
    r"IBK FUND TR FRA/C": "IBK FUND TFR FR A/C",
    r"IBK FUND TFR FRA/C": "IBK FUND TFR FR A/C",

    # FUND TRANSFER TO A/ (OCR enhanced)
    r"FUND TRANSFER TO A[fj1l\-]?": "FUND TRANSFER TO A/",
    r"FUND TRANSFER TO A$": "FUND TRANSFER TO A/",
    r"FUND TRANSFER T0 A/?": "FUND TRANSFER TO A/",
    r"FUND TRANSFER TO A/.*": "FUND TRANSFER TO A/",
    r"FUND TRANSFER TO A\.": "FUND TRANSFER TO A/",
    r"FUND TRANSFER TO A\|": "FUND TRANSFER TO A/",
    r"FUND TRANSFER TO Aj\.?": "FUND TRANSFER TO A/",
    r"FUND TRANSFER TO Af\.?": "FUND TRANSFER TO A/",
    r"FUND TRANSFER TO A,$": "FUND TRANSFER TO A/",

    # Pre-auth
    r"PRE-AUTH MYDEBIT\.?": "PRE-AUTH MYDEBIT",
    r"REV PREAUTH MYDEBIT\.?": "REV PREAUTH MYDEBIT",

    # Cash Withdrawal
    r"CASH WITHDRAWAL\.?": "CASH WITHDRAWAL",
}

# -----------------------------
# 1.2️⃣ Clean description
# -----------------------------
def clean_description(desc):
    if not isinstance(desc, str):
        return ""

    desc = desc.replace("|", "").strip()
    desc = re.sub(r"\s+", " ", desc)

    # Apply OCR corrections
    for pattern, replacement in DESCRIPTION_MAP.items():
        if re.search(pattern, desc, re.IGNORECASE):
            return replacement

    return desc


def extract_merchant(desc, current_merchant=""):
    desc = clean_str(desc)

    if current_merchant.strip() != "":
        return clean_str(current_merchant).upper()

    parts = re.split(r"-|,", desc)
    if len(parts) > 1:
        merchant = parts[-1].strip()
        return merchant.upper() if merchant else "UNKNOWN"

    return "UNKNOWN"

def clean_merchant_and_note(raw_merchant):
    """
    Clean merchant column & extract note after '/'.
    Rules:
      - If merchant starts with numbers → UNKNOWN
      - If contains 'PERHATIAN' or 'NOTA' → UNKNOWN
      - If note starts with numbers → UNKNOWN
      - If no note → UNKNOWN
    """
    if not isinstance(raw_merchant, str):
        return "UNKNOWN", "UNKNOWN"

    m = raw_merchant.strip()

    # Rule: PERHATIAN or NOTA → UNKNOWN
    if re.search(r"\bPERHATIAN\b|\bNOTA\b", m, re.IGNORECASE):
        return "UNKNOWN", "UNKNOWN"

    # Split merchant / note
    if "/" in m:
        merchant_part, note_part = m.split("/", 1)
        merchant_part = merchant_part.strip()
        note_part = note_part.strip()
    else:
        merchant_part = m.strip()
        note_part = "UNKNOWN"

    # Rule: merchant starts with number -> UNKNOWN
    if re.match(r"^\d", merchant_part):
        merchant_part = "UNKNOWN"

    # Rule: note starts with number -> UNKNOWN
    if isinstance(note_part, str) and re.match(r"^\d", note_part):
        note_part = "UNKNOWN"

    # Clean up '*' or trailing punctuation
    merchant_part = re.sub(r"[*]+$", "", merchant_part).strip().upper()
    note_part = re.sub(r"[*]+$", "", note_part).strip().upper()

    # Final cleaning
    if merchant_part == "":
        merchant_part = "UNKNOWN"
    if note_part == "":
        note_part = "UNKNOWN"

    return merchant_part, note_part


# -----------------------------
# 2️⃣ Clean single transaction
# -----------------------------
def clean_transaction(t, customer_info, statement_id):
    name = clean_str(customer_info.get("name", ""))
    account_number = clean_str(customer_info.get("account_number", ""))
    statement_date = clean_str(customer_info.get("statement_date", ""))

    raw_description = clean_str(t.get("transaction_type", ""))
    description = clean_description(raw_description)

        # Merchant + Note extraction and cleanup
    raw_merchant = clean_str(t.get("merchant", ""))
    merchant, note = clean_merchant_and_note(raw_merchant)

    debit = clean_number(t.get("debit", 0.0))
    credit = clean_number(t.get("credit", 0.0))
    balance = clean_number(t.get("balance", 0.0))
    date = clean_date(t.get("date", ""))

    return {
        "statement_id": statement_id,
        "date": date,
        "description": description,
        "merchant": merchant,
        "note": note,
        "debit": debit,
        "credit": credit,
        "balance": balance,
        "name": name,
        "account_number": account_number,
        "statement_date": statement_date
    }

# -----------------------------
# 3️⃣ Clean entire statement
# -----------------------------
def clean_statement(entry, statement_id):
    customer_info = entry.get("customer_info", {})
    transactions_raw = entry.get("transactions", [])
    return [clean_transaction(t, customer_info, statement_id) for t in transactions_raw]

# -----------------------------
# 4️⃣ MAIN EXECUTION
# -----------------------------
input_file = "output_json/ALL_RESULTS.json"
output_csv_file = "bank_statement_clean.csv"

try:
    with open(input_file, "r") as f:
        data = json.load(f)

    all_clean_transactions = []
    for i, entry in enumerate(data):
        statement_id = f"statement_{i+1}"
        all_clean_transactions.extend(clean_statement(entry, statement_id))

    # Remove broken rows
    all_clean_transactions = [
        t for t in all_clean_transactions
        if t["date"] != "" and (t["debit"] != 0.0 or t["credit"] != 0.0)
    ]

    # Save CSV
    df = pd.DataFrame(all_clean_transactions)
    df.to_csv(output_csv_file, index=False)

    print("✅ Clean CSV saved to:", output_csv_file)

except Exception as e:
    print("❌ Error:", e)
