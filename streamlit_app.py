import streamlit as st
import os
import gdown
from PIL import Image
from ultralytics import YOLO
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# PAGE SETUP
st.set_page_config(page_title="Caterpillar Detection", layout="wide")
st.title("🐛 Caterpillar Detection using YOLO")
st.caption("PROJECT CAO SECTION 3 GROUP 4. Members: Angelina Goh, Lee Siew Shuen, Ching Li Ban, Oliver Wong, Tan De Hang")

# MODEL DOWNLOAD
file_id = "1bSUm1mJSnqEOMZ6IEpLLJTLgG9lToqIq"  # Replace with your own file ID if needed
model_path = "best.pt"

if not os.path.exists(model_path):
    with st.spinner("Downloading YOLO model..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# LOAD MODEL
model = YOLO(model_path)

# SIDEBAR OPTIONS
st.subheader("🛠️ Detection Settings")
#confidence = st.sidebar.slider("Detection Confidence", 0.0,1.0, 0.25)
use_realtime = st.checkbox("Use Real-Time Webcam", value=False)
use_snapshot = st.checkbox("Use Snapshot Camera", value=False)

# ------------------ SNAPSHOT CAMERA INPUT ------------------
if use_snapshot:
    st.subheader("📸 Detect from Snapshot (Camera)")

    img_file_buffer = st.camera_input("Take a picture using your webcam")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Image", use_container_width=True)
        img_np = np.array(image.convert("RGB"))

        # Inference
        results = model.predict(img_np, conf=confidence)
        st.image(results[0].plot(), caption="Detection Result", use_container_width=True)



# ------------------ REAL-TIME WEBCAM STREAM ------------------
if use_realtime:
    st.subheader("🎥 Real-Time Webcam Detection")

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.confidence = confidence

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(img, conf=self.confidence)
            return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

    webrtc_streamer(
        key="realtime",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )


# ------------------ FILE UPLOAD INPUT ------------------
st.subheader("🖼️ Upload an Image for Caterpillar Detection")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    img_np = np.array(img.convert("RGB"))

    results = model.predict(img_np, conf=confidence)
    st.image(results[0].plot(), caption="Detected Image", use_container_width=True)
