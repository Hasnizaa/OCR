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
# 1. Preprocessing (same as before)
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

    return pytesseract.image_to_string(
        pil_img,
        config="--psm 6"
    )


# --------------------------------------------------------
# 3. Field Extraction
# --------------------------------------------------------
def extract_fields(text):
    name = None
    ic = None
    statement_date = None

    match_name = re.search(r"[A-Z ]+ BINTI [A-Z ]+", text)
    if match_name:
        name = match_name.group(0).strip()

    match_ic = re.search(r"\b\d{12}\b", text)
    if match_ic:
        ic = match_ic.group(0)

    match_date = re.search(r"\d{2}/\d{2}/\d{2,4}", text)
    if match_date:
        statement_date = match_date.group(0)

    return {
        "name": name,
        "ic": ic,
        "statement_date": statement_date
    }


# --------------------------------------------------------
# 4. Extract Table
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

    return pd.DataFrame(data)


# --------------------------------------------------------
# 5. JSON Builder
# --------------------------------------------------------
def build_json(fields, df):
    return {
        "customer_info": fields,
        "transactions": df.to_dict(orient="records"),
        "total_transactions": len(df)
    }


# --------------------------------------------------------
# 6. Run OCR on ALL IMAGES in FOLDER
# --------------------------------------------------------
def process_folder(folder_path, output_folder="output_json"):
    os.makedirs(output_folder, exist_ok=True)

    all_results = []

    for file in os.listdir(folder_path):

        if file.lower().endswith((".jpg", ".png", ".jpeg")):

            image_path = os.path.join(folder_path, file)
            print(f"📸 Processing: {file}")

            text = extract_text(image_path)
            fields = extract_fields(text)
            df = extract_table(text)
            json_data = build_json(fields, df)

            # Save per file
            output_path = os.path.join(output_folder, file.replace(".jpg", ".json").replace(".png", ".json"))
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
# 7. PROGRAM START
# --------------------------------------------------------
if __name__ == "__main__":
    input_folder = r"C:\Users\user\Documents\INTERNSHIP\PORTFOLIO\hackaton\OCR\img"
    process_folder(input_folder)
