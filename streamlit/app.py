import streamlit as st
from yolo_model import load_yolo_model, detect_objects
from webcam_handler import handle_webcam
from image_handler import handle_image_upload

# Load YOLO model
model = load_yolo_model()

# Streamlit title
st.title("Real-time Object Detection with YOLOv8")

# Sidebar to choose between webcam and photo upload
option = st.sidebar.selectbox("Select Input Method", ("Use Webcam", "Upload Photo"))

# Sidebar slider to adjust confidence threshold
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Handle webcam or image upload
if option == "Use Webcam":
    handle_webcam(model, confidence_threshold)
elif option == "Upload Photo":
    handle_image_upload(model, confidence_threshold)
