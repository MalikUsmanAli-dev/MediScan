"""
image_processing.py
--------------------
Digital Image Processing (DIP) module for the AI-Assisted Prescription
Recognition and Medicine Retrieval System.

Implements the classic DIP pipeline requested for prescription pre-processing:
    1. Grayscale conversion
    2. Noise reduction (Median Filter)
    3. Contrast enhancement (CLAHE)
    4. Sharpening (Unsharp Mask / Laplacian kernel)
    5. Adaptive Thresholding

All functions accept / return either NumPy arrays (OpenCV's native format)
or PIL Images, with small helpers to convert between the two so the rest
of the app (Streamlit + PDF generation) can stay format-agnostic.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Dict

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Conversion helpers
# --------------------------------------------------------------------------- #

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) into an OpenCV BGR NumPy array."""
    rgb_array = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV array (BGR or single-channel) into a PIL Image."""
    if len(cv2_image.shape) == 2:  # grayscale / binary
        return Image.fromarray(cv2_image)
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def get_image_metadata(file_bytes: bytes, pil_image: Image.Image) -> Dict[str, str]:
    """Return human-friendly metadata about an uploaded image."""
    size_kb = len(file_bytes) / 1024
    size_display = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
    width, height = pil_image.size
    return {
        "Dimensions": f"{width} x {height} px",
        "File Size": size_display,
        "Color Mode": pil_image.mode,
        "Megapixels": f"{(width * height) / 1_000_000:.2f} MP",
    }


# --------------------------------------------------------------------------- #
# Individual DIP operations
# --------------------------------------------------------------------------- #

def to_grayscale(cv2_image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)


def median_denoise(gray_image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Reduce salt-and-pepper / scanner noise using a Median Filter.
    Median filtering is preferred over Gaussian blur for document images
    because it removes speckle noise while preserving text edges.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # must be odd
    return cv2.medianBlur(gray_image, ksize)


def enhance_contrast(gray_image: np.ndarray, clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    """
    Enhance local contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization) - far better than global histogram
    equalization for unevenly lit prescription photographs.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray_image)


def sharpen_image(gray_image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """
    Sharpen handwriting / printed text using an Unsharp Mask technique:
    sharpened = original + amount * (original - blurred)
    """
    blurred = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray_image, 1 + amount, blurred, -amount, 0)
    return sharpened


def adaptive_threshold(gray_image: np.ndarray, block_size: int = 25, c: int = 10) -> np.ndarray:
    """
    Binarize the image using Adaptive Gaussian Thresholding so that
    uneven lighting across a photographed prescription doesn't break
    a single global threshold.
    """
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


# --------------------------------------------------------------------------- #
# Full pipeline orchestration
# --------------------------------------------------------------------------- #

@dataclass
class PipelineResult:
    """Container holding every intermediate stage + timing information."""
    stages: Dict[str, Image.Image] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0


def run_pipeline(pil_image: Image.Image) -> PipelineResult:
    """
    Execute the complete 5-stage Digital Image Processing pipeline and
    return every intermediate image (as PIL) plus per-stage timing,
    ready to be rendered as cards in the Streamlit UI.
    """
    result = PipelineResult()
    pipeline_start = time.perf_counter()

    # Stage 0: Original
    cv2_original = pil_to_cv2(pil_image)
    result.stages["Original"] = cv2_to_pil(cv2_original)

    # Stage 1: Grayscale
    t0 = time.perf_counter()
    gray = to_grayscale(cv2_original)
    result.timings_ms["Grayscale Conversion"] = (time.perf_counter() - t0) * 1000
    result.stages["Grayscale"] = cv2_to_pil(gray)

    # Stage 2: Median filter noise reduction
    t0 = time.perf_counter()
    denoised = median_denoise(gray)
    result.timings_ms["Median Filter (Noise Reduction)"] = (time.perf_counter() - t0) * 1000
    result.stages["Noise Reduced"] = cv2_to_pil(denoised)

    # Stage 3: Contrast enhancement (CLAHE)
    t0 = time.perf_counter()
    enhanced = enhance_contrast(denoised)
    result.timings_ms["Contrast Enhancement (CLAHE)"] = (time.perf_counter() - t0) * 1000
    result.stages["Enhanced"] = cv2_to_pil(enhanced)

    # Stage 4: Sharpening
    t0 = time.perf_counter()
    sharpened = sharpen_image(enhanced)
    result.timings_ms["Sharpening (Unsharp Mask)"] = (time.perf_counter() - t0) * 1000

    # Stage 5: Adaptive thresholding -> final processed image
    t0 = time.perf_counter()
    final = adaptive_threshold(sharpened)
    result.timings_ms["Adaptive Thresholding"] = (time.perf_counter() - t0) * 1000
    result.stages["Final Processed"] = cv2_to_pil(final)

    result.total_time_ms = (time.perf_counter() - pipeline_start) * 1000
    return result


def pil_image_to_bytes(pil_image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Serialize a PIL Image to raw bytes (used before sending to Groq / PDF)."""
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG" and pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffer, format=fmt)
    return buffer.getvalue()


def estimate_prescription_quality(gray_image: np.ndarray) -> Dict[str, float]:
    """
    Lightweight heuristic quality metrics computed locally (no AI call needed):
    - Sharpness: variance of the Laplacian (higher = crisper image)
    - Brightness: mean pixel intensity
    - Contrast: standard deviation of pixel intensities
    These feed the "Prescription Quality Assessment" KPI card.
    """
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    brightness = float(np.mean(gray_image))
    contrast = float(np.std(gray_image))
    return {
        "sharpness": round(float(laplacian_var), 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
    }
