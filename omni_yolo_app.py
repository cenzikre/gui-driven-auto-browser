import uvicorn, gc, torch, requests, base64, httpx
import numpy as np
import supervision as sv
from fastapi import FastAPI, JSONResponse
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from util.image_utils import classify_image_string, encode_image_to_base64
from util.omni_utils import BoxAnnotator, get_xyxy_box_center
from util.api_models import IconDetectRequest, IconDetectResponse, CommonResponse
from dotenv import load_dotenv
from io import BytesIO


device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()

root_dir = Path(__file__).parent
model_dir = root_dir / 'model_weights' / 'icon_detect' / 'model.pt'
image_dir = root_dir / 'screenshots'

# print("Root dir:", root_dir)
# print("Model dir:", model_dir)
# print("Image dir:", image_dir)
# print("Path exists?", image_dir.exists())


# Verify model file exists
if not model_dir.exists():
    raise FileNotFoundError(f"Model file not found at: {model_dir}")

app = FastAPI()

class YOLOModelSingleton:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOModelSingleton, cls).__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            print("Loading YOLO model...")
            self._model = YOLO(model_dir).to(device)
        return self._model

# Create singleton instance
yolo_singleton = YOLOModelSingleton()


async def load_image(source: str) -> tuple[Image.Image, np.ndarray]:
    print("start loading image")

    image_type = classify_image_string(source)
    print("input image type:", image_type)

    try:
        if image_type == "url":
            # Handle URL
            async with httpx.AsyncClient() as client:
                response = await client.get(source)
                response.raise_for_status()  # Raise an exception for bad status codes
                image = Image.open(BytesIO(response.content))
        elif image_type == "base64":
            # Handle base64 string
            image_data = base64.b64decode(source)
            image = Image.open(BytesIO(image_data))
        elif image_type == "data_uri":
            # Handle data URI
            image_data = base64.b64decode(source.split(",")[1])
            image = Image.open(BytesIO(image_data))
        elif image_type == "file":
            # Handle local file path
            path = image_dir / source
            print("path:", path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found at {path}")
            image = Image.open(path)
        else:
            raise ValueError(f"Unsupported image type: {image_type}")
        
        # Convert to RGB if necessary (in case of RGBA or other formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_arr = np.asarray(image)
        return image, image_arr
    
    except httpx.RequestError as e:
        raise Exception(f"Network error fetching image from URL: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"Server returned error status: {str(e)}")
    except base64.binascii.Error:
        raise Exception("Invalid base64 string")
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")


def overlay_ratio(width: int) -> dict:
    box_overlay_ratio = width / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    return draw_bbox_config


def annotate_image(image_arr: np.ndarray, boxes: np.ndarray, draw_bbox_config: dict) -> np.ndarray:
    w, h, _ = image_arr.shape
    detections = sv.Detections(xyxy=boxes)
    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]
    box_annotator = BoxAnnotator(**draw_bbox_config)
    annotated_image = image_arr.copy()
    return box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels, image_size=(w,h))


async def detect_icon(request: IconDetectRequest) -> tuple[Image.Image, np.ndarray]:
    try:
        image, image_arr = await load_image(request.source)
        w, h = image.size
        print("image size:", w, h)
        draw_bbox_config = overlay_ratio(w)

        # Get model instance when needed
        yolo_model = yolo_singleton.get_model()
        
        with torch.no_grad():
            icon_result = yolo_model.predict(
                source=image,
                conf=request.conf,
                iou=request.iou
            )
            icon_bbox_tensor = icon_result[0].boxes.xyxy
            icon_conf_tensor = icon_result[0].boxes.conf

        boxes = icon_bbox_tensor.cpu().numpy()
        if len(boxes) == 0:
            return Image.fromarray(image_arr), []
        
        # Get the centers of the boxes (x, y)
        centers = get_xyxy_box_center(boxes) / np.array([w, h])
        annotated_image = annotate_image(image_arr, boxes, draw_bbox_config)
        
        # Clean up only the tensors we created in this request
        del icon_bbox_tensor
        del icon_conf_tensor
        
        return Image.fromarray(annotated_image), centers.tolist()
    except Exception as e:
        raise Exception(f"Error in icon detection: {str(e)}")


@app.post("/detect_icon", response_model=CommonResponse)
async def detect_icon_endpoint(request: IconDetectRequest):
    try:
        annotated_image, box_centers = await detect_icon(request)
        encoded_image = encode_image_to_base64(annotated_image)
        payload = IconDetectResponse(
            centers=box_centers,
            image=encoded_image
        )
        return JSONResponse(
            content={
                "status": "success",
                "data": payload,
                "error": None
            }
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "data": None,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e)
                }
            },
            status_code=500
        )


# Add periodic cleanup task
@app.on_event("startup")
async def startup_event():
    # Initial cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Only use ipc_collect if using multiple processes
        if torch.cuda.device_count() > 1:
            torch.cuda.ipc_collect()

@app.on_event("shutdown")
async def shutdown_event():
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Only use ipc_collect if using multiple processes
        if torch.cuda.device_count() > 1:
            torch.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)