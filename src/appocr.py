import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
from io import BytesIO

# -------------------------------
# 1️⃣ OCR Functions
# -------------------------------
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    if coords.shape[0] == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=30)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

def extract_text(image_path):
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_img, config="--psm 6")
    return text

# -------------------------------
# 2️⃣ Cleaning Functions
# -------------------------------
DESCRIPTION_MAP = {
    r"S[NH]+O?12[\)\]I3]": "SNHO12J",
    r"FPX PAYMENT FR A[fj]": "FPX PAYMENT FR A/",
    r"IBK FUND T[RF]+R F?R?A/?C": "IBK FUND TFR FR A/C",
    r"FUND TRANSFER TO A[fj1l\-]?": "FUND TRANSFER TO A/",
    r"PRE-AUTH MYDEBIT\.?": "PRE-AUTH MYDEBIT",
    r"REV PREAUTH MYDEBIT\.?": "REV PREAUTH MYDEBIT",
    r"CASH WITHDRAWAL\.?": "CASH WITHDRAWAL",
}

def clean_str(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("  ", " ").strip()
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return s

def clean_description(desc):
    desc = clean_str(desc)
    for pattern, replacement in DESCRIPTION_MAP.items():
        if re.search(pattern, desc, re.IGNORECASE):
            return replacement
    return desc

def clean_merchant_and_note(raw_merchant):
    if not isinstance(raw_merchant, str):
        return "UNKNOWN", "UNKNOWN"

    m = raw_merchant.strip()
    if re.search(r"\bPERHATIAN\b|\bNOTA\b|\bPERTATIAN\b", m, re.IGNORECASE):
        return "UNKNOWN", "UNKNOWN"

    if "/" in m:
        merchant_part, note_part = m.split("/", 1)
        merchant_part = merchant_part.strip()
        note_part = note_part.strip()
    else:
        merchant_part = m.strip()
        note_part = "UNKNOWN"

    if re.match(r"^\d", merchant_part):
        merchant_part = "UNKNOWN"
    if re.match(r"^\d", note_part):
        note_part = "UNKNOWN"

    merchant_part = re.sub(r"[*]+$", "", merchant_part).strip().upper()
    note_part = re.sub(r"[*]+$", "", note_part).strip().upper()

    if merchant_part == "":
        merchant_part = "UNKNOWN"
    if note_part == "":
        note_part = "UNKNOWN"

    return merchant_part, note_part

# -------------------------------
# 3️⃣ Transaction Extraction
# -------------------------------
def extract_transactions(text):
    lines = text.split("\n")
    transactions = []

    pattern = re.compile(
        r"(\d{2}/\d{2}/\d{2,4})\s+"      # Date
        r"(.+?)\s+"                        # Transaction type / description
        r"([0-9,]+\.\d{2}[+-]?)\s+"       # Debit/Credit
        r"([0-9,]+\.\d{2})"               # Balance
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        m = pattern.search(line)
        if m:
            date, transaction_type, amt, bal = m.groups()
            amt = amt.replace(",", "")
            debit, credit = 0.0, 0.0
            if amt.endswith("+"):
                credit = float(amt[:-1])
            elif amt.endswith("-"):
                debit = float(amt[:-1])
            else:
                debit = float(amt)

            # Capture 1–2 lines for merchant
            merchant_lines = []
            for j in range(1, 3):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    if next_line and not re.match(r"\d{2}/\d{2}/\d{2}", next_line):
                        merchant_lines.append(next_line)
            merchant_raw = "|".join(merchant_lines) if merchant_lines else "UNKNOWN"
            merchant, note = clean_merchant_and_note(merchant_raw)

            transactions.append({
                "date": date,
                "transaction_type": clean_description(transaction_type),
                "merchant": merchant,
                "note": note,
                "debit": debit,
                "credit": credit,
                "balance": float(bal.replace(",", ""))
            })
            i += len(merchant_lines)
        i += 1

    return pd.DataFrame(transactions)

# -------------------------------
# 4️⃣ Streamlit UI
# -------------------------------
st.set_page_config(page_title="OCR Transaction Scanner", layout="wide")
st.title("🖼️ OCR Bank Statement Scanner & Cleaner")

uploaded_files = st.file_uploader(
    "Upload bank statement images (JPG/PNG/JPEG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    all_transactions = []

    for file in uploaded_files:
        st.subheader(f"Processing: {file.name}")
        image = Image.open(file)
        st.image(image, use_container_width=True)

        # Save temp file
        temp_path = f"temp_{file.name}"
        image.save(temp_path)

        # OCR
        st.text("⏳ Running OCR...")
        text = extract_text(temp_path)

        # Extract + Clean Transactions
        df_transactions = extract_transactions(text)
        st.subheader("📋 Extracted & Cleaned Transactions")
        st.dataframe(df_transactions)

        df_transactions["filename"] = file.name
        all_transactions.append(df_transactions)

    if all_transactions:
        df_all = pd.concat(all_transactions, ignore_index=True)
        st.subheader("💼 All Transactions Combined")
        st.dataframe(df_all)

        # Download CSV
        csv_buffer = BytesIO()
        df_all.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned Transactions CSV",
            data=csv_buffer.getvalue(),
            file_name="ocr_transactions_cleaned.csv",
            mime="text/csv"
        )
