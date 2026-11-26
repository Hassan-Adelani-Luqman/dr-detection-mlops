"""Checks shared by every endpoint that accepts an uploaded image."""
from fastapi import HTTPException, UploadFile

MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB

# Magic-byte signatures of the image types we accept.
_IMAGE_MAGIC = (
    b"\xff\xd8\xff",              # JPEG
    b"\x89PNG\r\n\x1a\n",        # PNG
    b"BM",                        # BMP
)


def looks_like_image(data: bytes) -> bool:
    """True if ``data`` starts with a JPEG, PNG or BMP signature."""
    return any(data.startswith(sig) for sig in _IMAGE_MAGIC)


def image_problem(data: bytes, max_bytes: int = MAX_IMAGE_BYTES):
    """Why ``data`` is not an acceptable image, or None if it is fine."""
    if len(data) > max_bytes:
        return f"larger than {max_bytes // (1024 * 1024)} MB"
    if not looks_like_image(data):
        return "not a JPEG, PNG or BMP image"
    return None


def read_image_upload(file: UploadFile, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Read an uploaded image into memory, or raise HTTP 400 if it isn't one.

    Reads at most ``max_bytes + 1`` bytes, so an oversized upload is rejected
    without being loaded in full.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=400,
                            detail=f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    if not looks_like_image(data):
        raise HTTPException(status_code=400,
                            detail="File is not a valid image (JPEG/PNG/BMP).")
    return data
