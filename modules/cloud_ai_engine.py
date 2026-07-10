"""
cloud_ai_engine.py
---------------------
A single, provider-agnostic vision-AI client. Every supported provider
(Groq, OpenAI, OpenRouter, Together, or a custom endpoint) exposes an
OpenAI-compatible `/chat/completions` endpoint that accepts image inputs,
so this client talks to all of them the same way over plain HTTP —
no vendor-specific SDK required.

Two capabilities:
    extract(image)                    - read a prescription from scratch
                                         (used by standalone "Cloud API" mode)
    verify_and_correct(image, draft)   - given a Tesseract-derived draft
                                         list, check it against the actual
                                         image and fix/complete/add entries
                                         (used by Hybrid mode's AI pass)
"""

from __future__ import annotations

import base64
import json
import time
from typing import List, Optional

import requests
from PIL import Image

from modules.image_processing import pil_image_to_bytes
from modules.analysis_types import MedicineItem, AnalysisResult
from modules.api_providers import ApiConfig

MAX_IMAGE_DIMENSION = 2048
REQUEST_TIMEOUT = 60

EXTRACT_PROMPT = """You are a clinical pharmacy assistant AI specialized in reading
handwritten and printed medical prescriptions, including those common in Pakistan
and South Asia. Carefully analyze the prescription image provided.

Extract every medicine you can identify. For each medicine return:
- name: the medicine / drug name (brand or generic, best guess if partially legible)
- strength: dosage strength if visible (e.g. "500mg"), else null
- frequency: usage instructions / dosing frequency if visible (e.g. "1-0-1 after meals"), else null
- confidence: your confidence (integer 0-100) that this specific reading is correct

Also return:
- overall_confidence: integer 0-100, your overall confidence across the whole prescription
- prescription_quality: one of "Excellent", "Good", "Fair", "Poor" based on legibility/image quality
- doctor_notes: any other readable notes (patient name, diagnosis, follow-up date), else null

Respond with ONLY valid JSON matching this exact schema, no markdown fences, no commentary:
{
  "medicines": [
    {"name": "string", "strength": "string|null", "frequency": "string|null", "confidence": 0}
  ],
  "overall_confidence": 0,
  "prescription_quality": "string",
  "doctor_notes": "string|null"
}

If the image is not a prescription or no medicines are legible, return an empty
"medicines" array and explain why in "doctor_notes".
"""

VERIFY_PROMPT_PREFIX = """You are a clinical pharmacy assistant AI. You are given a photo of a
medical prescription, plus a DRAFT list of medicines that a local OCR engine and a
fuzzy name-matching pipeline already extracted. The draft may contain misread/garbled
names, wrong or missing dosage strength, wrong or missing frequency, entries that
aren't really medicines, or it may be missing medicines entirely.

Carefully compare the draft against what is actually visible in the image, then produce
a corrected, final list:
- Fix any medicine name that was misread, garbled, or wrongly fuzzy-matched
- Correct or fill in dosage strength (e.g. "500mg") and frequency/usage instructions
  (e.g. "1-0-1", "twice daily", "BD") by reading the image directly
- Remove any draft entries that are not actually medicines
- Add any medicines clearly visible in the image that the draft missed entirely
- For every medicine in your final list, set "source" to one of:
    "kept"      - the draft entry was already correct, no change needed
    "corrected" - you fixed the name, strength, and/or frequency
    "added"     - this medicine was missing from the draft entirely

Respond with ONLY valid JSON matching this exact schema, no markdown fences, no commentary:
{
  "medicines": [
    {"name": "string", "strength": "string|null", "frequency": "string|null", "confidence": 0, "source": "kept|corrected|added"}
  ],
  "overall_confidence": 0,
  "prescription_quality": "string",
  "doctor_notes": "string|null"
}

DRAFT LIST FROM OCR + FUZZY MATCHING:
"""


class CloudVisionAnalyzer:
    """Provider-agnostic OpenAI-compatible vision chat-completions client."""

    def __init__(self, config: ApiConfig):
        if not config or not config.api_key:
            raise ValueError("An API key is required.")
        if not config.base_url:
            raise ValueError("An API base URL is required (pick a provider or set a custom one).")
        if not config.model:
            raise ValueError("A model name is required.")
        self.provider = config.provider
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model

    # ------------------------------------------------------------------ #
    def extract(self, image: Image.Image) -> AnalysisResult:
        start = time.perf_counter()
        try:
            data_url = self._encode_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACT_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            raw_text = self._call(messages)
            elapsed = time.perf_counter() - start
            return self._parse_response(raw_text, elapsed, engine="api")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            return AnalysisResult(success=False, engine="api", error_message=self._friendly_error(exc), processing_time_s=elapsed)

    def verify_and_correct(self, image: Image.Image, draft_medicines: List[MedicineItem]) -> AnalysisResult:
        start = time.perf_counter()
        try:
            draft_json = json.dumps(
                [
                    {"name": m.name, "strength": m.strength, "frequency": m.frequency, "confidence": m.confidence}
                    for m in draft_medicines
                ]
            )
            prompt = VERIFY_PROMPT_PREFIX + draft_json
            data_url = self._encode_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            raw_text = self._call(messages)
            elapsed = time.perf_counter() - start
            return self._parse_response(raw_text, elapsed, engine="hybrid-verified")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            return AnalysisResult(success=False, engine="hybrid-verified", error_message=self._friendly_error(exc), processing_time_s=elapsed)

    # ------------------------------------------------------------------ #
    def _call(self, messages: list) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": 0.2, "response_format": {"type": "json_object"}}

        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code >= 400:
            # Some OpenAI-compatible providers reject 'response_format' — retry without it.
            payload.pop("response_format", None)
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    def _encode_image(self, image: Image.Image) -> str:
        resized = self._fit_for_upload(image)
        image_bytes = pil_image_to_bytes(resized, fmt="JPEG")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _fit_for_upload(image: Image.Image) -> Image.Image:
        w, h = image.size
        longest = max(w, h)
        if longest <= MAX_IMAGE_DIMENSION:
            return image
        scale = MAX_IMAGE_DIMENSION / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _parse_response(raw_text: str, elapsed: float, engine: str) -> AnalysisResult:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).replace("json\r\n", "", 1)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return AnalysisResult(
                success=False,
                error_message=f"Could not parse AI response as JSON ({exc}).",
                raw_response=raw_text,
                processing_time_s=elapsed,
                engine=engine,
            )

        medicines = [
            MedicineItem(
                name=item.get("name", "Unknown"),
                strength=item.get("strength"),
                frequency=item.get("frequency"),
                confidence=int(item.get("confidence", 0) or 0),
                source=item.get("source", "ai"),
            )
            for item in data.get("medicines", [])
        ]

        return AnalysisResult(
            success=True,
            medicines=medicines,
            overall_confidence=int(data.get("overall_confidence", 0) or 0),
            prescription_quality=data.get("prescription_quality", "Unknown"),
            doctor_notes=data.get("doctor_notes"),
            raw_response=raw_text,
            processing_time_s=elapsed,
            engine=engine,
        )

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "401" in lowered or "unauthorized" in lowered or "api key" in lowered:
            return "Invalid or missing API key for the selected provider. Check it in Settings."
        if "429" in lowered or "rate" in lowered or "quota" in lowered:
            return "Rate limit / quota exceeded for the selected provider. Wait a moment and try again."
        if "timeout" in lowered or "timed out" in lowered:
            return "The request timed out. Check your internet connection and try again."
        if "connectionerror" in type(exc).__name__.lower() or "connection" in lowered:
            return "Could not reach the API endpoint — check your internet connection and the base URL in Settings."
        if "404" in lowered or "not found" in lowered:
            return f"Model or endpoint not found ('{message}'). Check the model name / base URL in Settings."
        return f"API request failed: {message}"
