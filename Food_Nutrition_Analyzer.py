import streamlit as st
import numpy as np
import pickle
import os
import re
import cv2
import pytesseract
from PIL import Image

from utils.model_wrapper import FoodHealthModel

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Food Nutrition Analyzer", layout="centered")

st.title("🥗 Food Nutrition Analyzer")
st.write("Upload a food label image or enter nutrition values manually.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "food_health_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- OCR FUNCTION ----------------
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
    return text

# ---------------- PARSE NUTRIENTS ----------------
def parse_nutrients(text):
    text = text.lower()

    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    nutrients = {
        "energy-kcal_100g": grab(r"energy.*?([\d]+\.?\d*)"),
        "fat_100g": grab(r"(?:total fat|fat).*?([\d]+\.?\d*)"),
        "saturated-fat_100g": grab(r"saturated.*?([\d]+\.?\d*)"),
        "carbohydrates-total_100g": grab(r"carbohydrate.*?([\d]+\.?\d*)"),
        "sugars_100g": grab(r"(?:total sugars|added sugars|sugars).*?([\d]+\.?\d*)"),
        "fiber_100g": grab(r"(?:fiber|fibre).*?([\d]+\.?\d*)"),
        "proteins_100g": grab(r"protein.*?([\d]+\.?\d*)"),
        "salt_100g": grab(r"salt.*?([\d]+\.?\d*)"),
        "sodium_100g": grab(r"sodium.*?([\d]+\.?\d*)") / 1000,
        "additives_n": 0,
        "nova_group": 3
    }

    # 🚨 OCR sanity limits
    nutrients["fiber_100g"] = min(nutrients["fiber_100g"], 50)
    nutrients["proteins_100g"] = min(nutrients["proteins_100g"], 80)
    nutrients["sugars_100g"] = min(nutrients["sugars_100g"], 100)
    nutrients["fat_100g"] = min(nutrients["fat_100g"], 100)

    return nutrients

# ---------------- IMAGE UPLOAD ----------------
st.subheader("📸 Image Upload")
uploaded_image = st.file_uploader(
    "Upload Nutrition Label Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", width=300)

    text = extract_text_from_image(uploaded_image)
    st.subheader("📝 OCR Extracted Text")
    st.text(text)

    nutrients = parse_nutrients(text)

    st.subheader("🥗 Parsed Nutrients (per 100g)")
    st.json(nutrients)

    label, confidence = model.predict(nutrients)

    st.subheader("✅ Final Result")
    st.success(f"Prediction: {label}")
    st.info(f"Healthy confidence: {confidence*100:.2f}%")

# ---------------- MANUAL INPUT ----------------
st.subheader("🧪 Manual Nutrition Test (per 100g)")

sample_food = {}
for f in model.features:
    sample_food[f] = st.number_input(f.replace("_", " "), min_value=0.0, value=0.0)

if st.button("Analyze Manual Input"):
    label, confidence = model.predict(sample_food)

    st.subheader("✅ Result")
    st.success(f"Prediction: {label}")
    st.info(f"Healthy confidence: {confidence*100:.2f}%")
