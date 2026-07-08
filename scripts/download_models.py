import sys
import os
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("download_models")

def main():
    LOGGER.info("=== Pre-downloading YOLOv8 model for Castor Studio AI ===")
    try:
        # Load the model which will trigger the download and caching
        # By default it downloads to the current directory or the system cache (~/.config/Ultralytics)
        model = YOLO("yolov8n.pt")
        LOGGER.info("YOLOv8 model pre-downloaded successfully!")
        
        # Check where it was saved
        if os.path.exists("yolov8n.pt"):
            LOGGER.info("Saved yolov8n.pt to local directory.")
        else:
            LOGGER.info("Model cached in global Ultralytics directory.")
            
    except Exception as exc:
        LOGGER.error("Failed to pre-download YOLOv8: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()
