import os
from pathlib import Path
from ultralytics import YOLO

def export_to_int8():
    """Converts YOLOv11 PyTorch model to optimized INT8 TFLite format."""
    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "weights" / "best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Source model not found at: {model_path}")

    # Load and export with INT8 quantization for latency reduction
    model = YOLO(str(model_path))
    
    model.export(
        format='tflite',
        int8=True,
        data='coco8.yaml',
        imgsz=640
    )

if __name__ == "__main__":
    try:
        export_to_int8()
        print("Model exported to INT8 TFLite successfully.")
    except Exception as e:
        print(f" Export failed: {str(e)}")