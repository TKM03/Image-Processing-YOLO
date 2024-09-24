from ultralytics import YOLO

def load_yolo_model(model_path='../trained_yolov8_model.pt'):
    """Load YOLOv8 model from the given path."""
    model = YOLO(model_path)
    return model

def detect_objects(model, source, confidence_threshold):
    """Run YOLO detection on the source with the specified confidence threshold."""
    results = model.predict(source=source, conf=confidence_threshold, save=False)
    return results
