import streamlit as st
import cv2
from yolo_model import detect_objects

def handle_webcam(model, confidence_threshold):
    """Handle the webcam stream and perform YOLO detection."""
    
    # Initialize webcam running state if not present in session state
    if 'run' not in st.session_state:
        st.session_state.run = False

    # Show the "Start Webcam" button only if the webcam is not running
    if not st.session_state.run:
        if st.button('Start Webcam'):
            st.session_state.run = True  # Set the webcam to running

    # Show the "Stop Webcam" button only if the webcam is running
    if st.session_state.run:
        if st.button('Stop Webcam'):
            st.session_state.run = False  # Set the webcam to stopped

    # Webcam logic: If the webcam is running, capture frames and display them
    if st.session_state.run:
        # Initialize video capture object
        cap = cv2.VideoCapture(0)

        # Placeholder for displaying the real-time detection stream
        frame_window = st.image([])

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            # Perform YOLO detection
            results = detect_objects(model, frame, confidence_threshold)

            # Process the results and draw bounding boxes
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    class_id = box.cls.item()

                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{result.names[class_id]}: {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert the frame to RGB (OpenCV uses BGR format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the real-time image in Streamlit
            frame_window.image(frame_rgb)

        cap.release()
        st.write("Webcam stopped.")
