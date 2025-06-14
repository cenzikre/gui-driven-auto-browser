import base64, io, os, re
from PIL import Image
from typing import Union
from urllib.parse import urlparse


def encode_image_to_base64(image: Union[str, Image.Image]) -> str:
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Invalid image type: {type(image)}")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}

def classify_image_string(s: str) -> str:
    """
    Return one of: 'data_uri', 'base64', 'url', 'file', or 'unknown'
    """
    s = s.strip()

    # 1) Data-URI (data:image/...;base64,<big blob>)
    if s.lower().startswith("data:image/"):
        return "data_uri"

    # 2) URL (http/https) with image extension at the end
    parsed = urlparse(s)
    if parsed.scheme in ("http", "https"):
        _, ext = os.path.splitext(parsed.path.lower())
        if ext in IMAGE_EXTS:
            return "url"

    # 3) Raw base-64 blob (very long, only base64 chars)
    base64_charset = re.fullmatch(r"[A-Za-z0-9+/=\s]+", s)
    if base64_charset and len(s) > 100:        # length guard
        return "base64"

    # 4) Local file name / path ending with an image extension
    _, ext = os.path.splitext(s.lower())
    if ext in IMAGE_EXTS:
        return "file"

    return "unknown"