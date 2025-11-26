OCR & Data Analysis Pipeline

This repository provides scripts for performing OCR on images, as well as anomaly detection and segmentation on the extracted data. You can also run a web interface using Streamlit or explore the original OCR implementation in Google Colab.

---

## ⚡ Prerequisites

1. **Python 3.8+** installed  
2. **Tesseract OCR** installed: [Download here](https://github.com/tesseract-ocr/tesseract)  
   - Ensure `tesseract.exe` is added to your system PATH.  
3. Optional: **Streamlit** for running the web app:
pip install streamlit

🛠 Setup
Clone this repository:
git clone https://github.com/yourusername/ocr-project.git
cd ocr-project

Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m venv venv
.\venv\Scripts\activate.bat

Install dependencies:
pip install -r requirements.txt
Tip: requirements.txt should include pytesseract, opencv-python, numpy, pandas, streamlit, and other necessary packages.

📂 Scripts Overview
Script	Description
src/batch_ocr.py	Scan multiple images in a folder and extract text using OCR
src/ocr_extract.py	Scan a single image at a time
src/anomaly_detection.py	Detect anomalies in extracted JSON data
src/segmentation.py	Segment extracted JSON data
src/app_streamlit.py	Run a Streamlit web app for OCR processing

🚀 Usage

1️⃣ Batch OCR
Scan all images in a folder:
python src/batch_ocr.py

2️⃣ Single Image OCR
Scan one image at a time:
python src/ocr_extract.py

3️⃣ Anomaly Detection
Detect anomalies in JSON output:
python src/anomaly_detection.py --input output_json --outdir results_anomaly --contamination 0.05

4️⃣ Segmentation
Segment JSON output:
python src/segmentation.py --input output_json --outdir results_segmentation

5️⃣ Streamlit Web App
Run the OCR web interface:
streamlit run src/app_streamlit.py

6️⃣ Original OCR in Colab
You can also explore the original OCR implementation in Google Colab
