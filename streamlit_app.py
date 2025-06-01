import streamlit as st
import os
import gdown
from PIL import Image
from ultralytics import YOLO
import numpy as np



# Google Drive file ID for YOLO model trained on caterpillars
file_id = "1oHLmjRhWXaA-meKxxZlD2CLJ97I5719R"  # Make sure this is the caterpillar model
model_path = "best.pt"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load YOLO model
model = YOLO(model_path)

# Title for the app
st.title("🐛 Caterpillar Detection with YOLO")

# --- SECTION 1: Snapshot Detection (st.camera_input) ---
st.subheader("📸 Detect Caterpillars from Your Camera (Snapshot)")

img_file_buffer = st.camera_input("Take a picture using your webcam")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.image(image, caption="Captured Image", use_column_width=True)

    img_np = np.array(image.convert("RGB"))

    results = model(img_np)

    st.image(results[0].plot(), caption="Detection Result", use_column_width=True)

# --- SECTION 2: Upload Image Detection ---
st.subheader("🖼️ Upload an Image for Detection")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(img.convert("RGB"))

    results = model(img_np)

    st.image(results[0].plot(), caption="Detected Image", use_column_width=True)
    results = model(img_np)

    # Display result
    st.image(results[0].plot(), caption="Detected Image", use_column_width=True)
