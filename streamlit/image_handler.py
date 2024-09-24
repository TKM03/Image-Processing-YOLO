import streamlit as st
import numpy as np
from PIL import Image
import cv2
from yolo_model import detect_objects

def handle_image_upload(model, confidence_threshold):
    """Handle image upload and run YOLO detection."""
    
    # Upload image using Streamlit's file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Ensure the image is in RGB format (convert from RGBA if necessary)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Store the original image for comparison
        original_image_np = image_np.copy()

        # Run YOLO detection on the uploaded image
        results = detect_objects(model, image_np, confidence_threshold)

        # Process the results and draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                class_id = box.cls.item()

                # Draw the bounding box on the image
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{result.names[class_id]}: {conf:.2f}'
                cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the original and detection images side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image_np, caption="Original Image")
        with col2:
            st.image(image_np, caption="Detection Image")
