from __future__ import annotations

import io

import cv2
import numpy as np
from celery import Celery

from vectorizer.config import VectorizerConfig
from vectorizer.pipeline import run as pipeline_run

celery_app = Celery(
    "vectorizer",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"


@celery_app.task(bind=True, name="vectorizer.tasks.vectorize")
def vectorize_task(self, image_bytes: bytes, config_overrides: dict) -> str:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not decode image")

    config = VectorizerConfig(**{k: v for k, v in config_overrides.items() if v is not None})
    result = pipeline_run(image, config)
    return result.svg_string
