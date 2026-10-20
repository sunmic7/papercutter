"""Qwen visual review used only by the standalone photographed-QA importer."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"
PROMPT = """You review one already-separated photographed Chinese physics question and candidate answer images. The first image is the question; following images are labelled filenames. Do not trust filenames: an image can be a continuation, standalone answer diagram, or another question. Think silently. Return JSON only with question_text (complete Chinese transcription), answer (final answer supported by official images, otherwise empty), cot (concise auditable Chinese solution explanation, no hidden reasoning), image_domain, QA_diff, overall_confidence 0..1, and candidates [{filename, role: answer|continuation|answer_figure|unrelated|unknown, confidence:0..1}]. Keep all subparts in one compound QA. Never invent an answer or solution."""


def _image_uri(image: np.ndarray, max_side: int) -> str:
    if image is None or image.size == 0:
        raise ValueError("empty image")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(width, height))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("cannot encode image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("missing model JSON")
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("missing model JSON")
    payload = json.loads(value[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("invalid model JSON")
    return payload


def _request(data: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY")
    request = urllib.request.Request(
        os.environ.get("PAPERCUTTER_VLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/chat/completions",
        data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(1, int(os.environ.get("PAPERCUTTER_VLM_TIMEOUT", "120")))) as response:
        return json.loads(response.read().decode("utf-8"))


def review_group(question_image: np.ndarray, candidate_images: list[tuple[str, np.ndarray]]) -> dict[str, Any] | None:
    """Return a guarded QA-group review; failures remain in the review queue."""
    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        return None
    try:
        max_side = int(os.environ.get("PAPERCUTTER_VLM_MAX_SIDE", "2200"))
        content: list[dict[str, Any]] = [
            {"type": "text", "text": PROMPT + "\nQUESTION_IMAGE:"},
            {"type": "image_url", "image_url": {"url": _image_uri(question_image, max_side)}},
        ]
        for filename, image in candidate_images:
            content.extend([
                {"type": "text", "text": f"CANDIDATE_FILENAME={filename}"},
                {"type": "image_url", "image_url": {"url": _image_uri(image, max_side)}},
            ])
        response = _request({
            "model": os.environ.get("PAPERCUTTER_VLM_MODEL", DEFAULT_MODEL), "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        })
        choices = response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        return _json_object(message.get("content"))
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError, IndexError, KeyError):
        return None
