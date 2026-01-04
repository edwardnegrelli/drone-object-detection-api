# Drone Object Detection API

Real-time object detection API built with YOLOv8 and FastAPI for analyzing surveillance and aerial imagery.

## Features

- **Real-time inference**: Processes images in 150-400ms on CPU
- **RESTful API**: Clean HTTP endpoints for image upload and detection
- **Performance monitoring**: Tracks latency metrics (min/max/avg/p95)
- **Automatic documentation**: Interactive API docs at `/docs`

## Tech Stack

- **YOLOv8-nano**: State-of-the-art object detection model
- **FastAPI**: High-performance async web framework
- **PyTorch**: Deep learning inference
- **Uvicorn**: ASGI server

## Performance Metrics

- **Inference latency**: 150-400ms (CPU-only, AMD Ryzen)
- **Cold start**: ~5.5s (initial model load)
- **Confidence threshold**: 50%
- **Supported objects**: 80 COCO classes (person, car, airplane, etc.)

## API Endpoints

- `GET /` - API information
- `POST /predict` - Upload image for object detection
- `GET /health` - Health check
- `GET /metrics` - Performance statistics

## Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/drone-object-detection-api.git
cd drone-object-detection-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API
python main.py
```

## Usage

Access the interactive API documentation at `http://localhost:8000/docs`

Example detection response:
```json
{
  "detections": [
    {
      "class": "airplane",
      "confidence": 0.64,
      "bbox": {"x1": 46.5, "y1": 224.0, "x2": 210.6, "y2": 316.5}
    }
  ],
  "num_objects": 1,
  "inference_time_ms": 406.99
}
```

## Use Cases

- ISR (Intelligence, Surveillance, Reconnaissance) imagery analysis
- Drone footage processing
- Automated threat detection
- Asset identification and tracking

## Project Context

Built to demonstrate model-serving infrastructure experience for defense/aerospace applications, including:
- Integration of AI/ML pipelines into REST APIs
- Performance evaluation (latency, throughput)
- System-level architecture considerations
- Production deployment readiness

## License

MIT