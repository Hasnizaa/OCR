import pytesseract
import cv2
import numpy as np
import re
import json
import pandas as pd
from PIL import Image

# Tesseract path (UPDATE if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------------
# 1. Image Preprocessing
# --------------------------------------------------------
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

# --------------------------------------------------------
# 2. Extract Raw Text
# --------------------------------------------------------
def extract_text(image_path):
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_img, config="--psm 6")
    return text

# --------------------------------------------------------
# 3. Extract Customer Fields
# --------------------------------------------------------
def extract_customer_info(text):
    account_number = None
    name = None
    statement_date = None

    match_ac = re.search(r"\b\d{6}-\d{6}\b", text)
    if match_ac:
        account_number = match_ac.group(0)

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if account_number and account_number in line and i >= 2:
            name = lines[i-2].strip()
            break

    match_date = re.search(r"(?:Statement Date[:\s]*)(\d{2}/\d{2}/\d{2,4})", text, re.IGNORECASE)
    if match_date:
        statement_date = match_date.group(1)
    else:
        for line in lines[:10]:
            m = re.search(r"\b\d{2}/\d{2}/\d{2,4}\b", line)
            if m:
                statement_date = m.group(0)
                break

    return {
        "name": name if name else "Unknown",
        "account_number": account_number if account_number else "Unknown",
        "statement_date": statement_date if statement_date else "Unknown"
    }

# --------------------------------------
# 4. Extract Transactions (Supports 2-line merchant)
# --------------------------------------
def extract_transactions(text):
    lines = text.split("\n")
    transactions = []

    pattern = re.compile(
        r"(\d{2}/\d{2}/\d{2})\s+"           
        r"(.+?)\s+"                         
        r"([0-9,]+\.\d{2}[+-]?)\s+"          
        r"([0-9,]+\.\d{2})"                  
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

            # Capture 2 merchant description lines
            merchant_lines = []
            for j in range(1, 3):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    if next_line and not re.match(r"\d{2}/\d{2}/\d{2}", next_line):
                        merchant_lines.append(next_line)

            merchant = " | ".join(merchant_lines)
            i += len(merchant_lines)

            transactions.append({
                "date": date,
                "transaction_type": transaction_type.strip(),
                "merchant": merchant.strip(),
                "debit": debit,
                "credit": credit,
                "balance": float(bal.replace(",", ""))
            })

        i += 1

    return pd.DataFrame(transactions)

# --------------------------------------------------------
# 6. Convert Everything to JSON
# --------------------------------------------------------
def build_json(fields, table_df):
    output = {
        "customer_info": fields,
        "transactions": table_df.to_dict(orient="records"),
        "total_transactions": len(table_df)
    }
    return json.dumps(output, indent=4)

# --------------------------------------------------------
# 7. Main Runner (Single Image)
# --------------------------------------------------------
if __name__ == "__main__":
    image_path = r"C:\Users\user\Documents\INTERNSHIP\PORTFOLIO\hackaton\OCR\data1.jpg"

    print("⏳ Extracting text...")
    text = extract_text(image_path)

    print("\n📌 Raw OCR Extracted!\n")
    print(text)

    fields = extract_customer_info(text)
    df = extract_transactions(text)

    print("\n📌 Extracted Customer Info:", fields)
    print("\n📌 Transaction Table:")
    print(df)

    final_json = build_json(fields, df)

    print("\n📦 FINAL JSON OUTPUT")
    print(final_json)

    with open("bank_statement_output.json", "w") as f:
        f.write(final_json)

    print("\n✅ Saved as bank_statement_output.json")
