import os
import pytesseract
import cv2
import numpy as np
import re
import json
import pandas as pd
from PIL import Image

# ----- TESSERACT PATH -----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------------
# 1. Preprocessing
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
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

# --------------------------------------------------------
# 2. OCR Text Extraction
# --------------------------------------------------------
def extract_text(image_path):
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)
    return pytesseract.image_to_string(pil_img, config="--psm 6")

# --------------------------------------------------------
# 3. Extract Maybank Header (Account Number + Name)
# --------------------------------------------------------
def extract_customer_info(text):
    account_number = None
    name = None
    statement_date = None

    # Account number pattern: numbers with dash, e.g., 112473-091156
    match_ac = re.search(r"\b\d{6}-\d{6}\b", text)
    if match_ac:
        account_number = match_ac.group(0)

    # Name: lines above account number (simplified assumption)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if account_number and account_number in line:
            # Usually the name is 2-3 lines above
            if i >= 2:
                name = lines[i-2].strip()
            break

    # Statement date pattern: look for dd/mm/yyyy or dd/mm/yy
    match_date = re.search(r"(?:Statement Date[:\s]*)(\d{2}/\d{2}/\d{2,4})", text, re.IGNORECASE)
    if match_date:
        statement_date = match_date.group(1)
    else:
        # fallback: first date in the top 10 lines
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
# 3. Merge Multi-line Transactions
# --------------------------------------
def merge_transactions_lines(text):
    lines = text.split("\n")
    merged_lines = []
    buffer = ""
    
    for line in lines:
        # If line starts with a date (dd/mm/yy)
        if re.match(r"\d{2}/\d{2}/\d{2}", line):
            if buffer:
                merged_lines.append(buffer.strip())
            buffer = line
        else:
            # Append as merchant/extra description
            buffer += " || " + line.strip()
    if buffer:
        merged_lines.append(buffer.strip())
    
    return merged_lines

# --------------------------------------
# 4. Extract Transactions
# --------------------------------------
def extract_transactions(text):
    lines = text.split("\n")
    transactions = []

    # Regex to detect transaction line (date + type + amount + balance)
    pattern = re.compile(
        r"(\d{2}/\d{2}/\d{2})\s+"       # Date
        r"(.+?)\s+"                      # Transaction type
        r"([0-9,]+\.\d{2}[+-]?)\s+"      # Amount
        r"([0-9,]+\.\d{2})"              # Balance
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        m = pattern.search(line)
        if m:
            date, transaction_type, amt, bal = m.groups()
            amt = amt.replace(",", "")
            debit, credit = 0.0, 0.0

            # Determine debit or credit
            if amt.endswith("+"):
                credit = float(amt[:-1])
            elif amt.endswith("-"):
                debit = float(amt[:-1])
            else:
                debit = float(amt)

            merchant_lines = []
            # Look at next 2 lines for merchant info
            for j in range(1, 3):  # next 2 lines
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    if next_line and not re.match(r"\d{2}/\d{2}/\d{2}", next_line):
                        merchant_lines.append(next_line)
            merchant = " / ".join(merchant_lines)

            # Skip the merchant lines in the loop
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
# 5. Build JSON
# --------------------------------------------------------
def build_json(customer_info, df):
    return {
        "customer_info": customer_info,
        "transactions": df.to_dict(orient="records"),
        "total_transactions": len(df)
    }

# --------------------------------------------------------
# 6. Process folder of images
# --------------------------------------------------------
def process_folder(folder_path, output_folder="output_json"):
    os.makedirs(output_folder, exist_ok=True)
    all_results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, file)
            print(f"📸 Processing: {file}")
            text = extract_text(image_path)
            customer_info = extract_customer_info(text)
            df = extract_transactions(text)
            json_data = build_json(customer_info, df)

            # Save per file
            output_path = os.path.join(output_folder, file.rsplit(".", 1)[0]+".json")
            with open(output_path, "w") as f:
                json.dump(json_data, f, indent=4)
            print(f"✅ Saved: {output_path}")
            all_results.append(json_data)

    # Save combined JSON
    with open(os.path.join(output_folder, "ALL_RESULTS.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    print("\n🎉 DONE! All images processed.")
    print(f"📦 Combined JSON saved as: {output_folder}/ALL_RESULTS.json")

# --------------------------------------------------------
# 7. MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    input_folder = r"C:\Users\user\Documents\INTERNSHIP\PORTFOLIO\hackaton\OCR\img"
    process_folder(input_folder)
