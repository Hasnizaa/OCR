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

    thresh = cv2.adaptiveThreshold(gray, 255,
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

    text = pytesseract.image_to_string(
        pil_img,
        config="--psm 6"
    )
    return text


# --------------------------------------------------------
# 3. Extract Fields (Name, IC, Statement Date)
# --------------------------------------------------------
def extract_fields(text):

    name = None
    ic = None
    statement_date = None

    # Name: look for capital letter patterns (simple)
    match_name = re.search(r"[A-Z ]+ BINTI [A-Z ]+", text)
    if match_name:
        name = match_name.group(0).strip()

    # IC detection (usually 12 digits)
    match_ic = re.search(r"\b\d{12}\b", text)
    if match_ic:
        ic = match_ic.group(0)

    # Statement Date
    match_date = re.search(r"\d{2}/\d{2}/\d{2,4}", text)
    if match_date:
        statement_date = match_date.group(0)

    return {
        "name": name,
        "ic": ic,
        "statement_date": statement_date
    }


# --------------------------------------------------------
# 4. Extract Transaction Table
# --------------------------------------------------------
def extract_table(text):
    lines = text.split("\n")
    data = []

    transaction_pattern = re.compile(
        r"(\d{2}/\d{2}/\d{2})\s+\|(.+?)\s+([0-9,]+\.\d{2}[+-])\s+([0-9,]+\.\d{2})"
    )

    for line in lines:
        m = transaction_pattern.search(line)
        if m:
            date, desc, amount, balance = m.groups()
            data.append({
                "date": date,
                "description": desc.strip(),
                "amount": amount.replace(",", ""),
                "balance": balance.replace(",", "")
            })

    df = pd.DataFrame(data)
    return df


# --------------------------------------------------------
# 5. Convert Everything to JSON
# --------------------------------------------------------
def build_json(fields, table_df):
    output = {
        "customer_info": fields,
        "transactions": table_df.to_dict(orient="records"),
        "total_transactions": len(table_df)
    }
    return json.dumps(output, indent=4)


# --------------------------------------------------------
# 6. Main Runner
# --------------------------------------------------------
if __name__ == "__main__":
    image_path = r"C:\Users\user\Documents\INTERNSHIP\PORTFOLIO\hackaton\OCR\data1.jpg"

    print("⏳ Extracting text...")
    text = extract_text(image_path)

    print("\n📌 Raw OCR Extracted!\n")

    fields = extract_fields(text)
    df = extract_table(text)

    print("\n📌 Extracted Fields:", fields)
    print("\n📌 Transaction Table:")
    print(df)

    final_json = build_json(fields, df)

    print("\n📦 FINAL JSON OUTPUT")
    print(final_json)

    with open("bank_statement_output.json", "w") as f:
        f.write(final_json)

    print("\n✅ Saved as bank_statement_output.json")