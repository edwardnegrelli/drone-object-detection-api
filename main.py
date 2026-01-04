from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import time
from ultralytics import YOLO
import numpy as np
from typing import List, Dict

# Initialize FastAPI app
app = FastAPI(title="Drone Object Detection API")

# Load YOLOv8 model (downloads automatically on first run)
model = YOLO('yolov8n.pt')  # 'n' = nano version, fastest

# Metrics tracking
inference_times = []

@app.get("/")
def root():
    return {
        "message": "Drone Object Detection API",
        "endpoints": {
            "/predict": "POST an image to detect objects",
            "/health": "Check API health",
            "/metrics": "View performance metrics"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detect objects in an uploaded image.
    Returns detected objects with bounding boxes and confidence scores.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Track inference time
        start_time = time.time()
        
        # Run inference
        results = model(image, conf=0.5)  # 50% confidence threshold
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append({
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                })
        
        return {
            "detections": detections,
            "num_objects": len(detections),
            "inference_time_ms": round(inference_time, 2),
            "image_size": {"width": image.width, "height": image.height}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """
    Return performance metrics for monitoring.
    """
    if not inference_times:
        return {"message": "No inferences yet"}
    
    return {
        "total_inferences": len(inference_times),
        "avg_latency_ms": round(np.mean(inference_times), 2),
        "min_latency_ms": round(min(inference_times), 2),
        "max_latency_ms": round(max(inference_times), 2),
        "p95_latency_ms": round(np.percentile(inference_times, 95), 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)