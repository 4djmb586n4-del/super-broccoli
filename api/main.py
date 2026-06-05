from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from .schemas import TaskStatus, VectorizeRequest
from .tasks import vectorize_task

app = FastAPI(title="Vectorizer API", version="0.1.0")


@app.post("/vectorize", response_model=TaskStatus)
async def vectorize(
    file: UploadFile = File(...),
    config: VectorizeRequest | None = None,
) -> TaskStatus:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    overrides = config.model_dump() if config else {}
    task = vectorize_task.delay(image_bytes, overrides)
    return TaskStatus(task_id=task.id, status="PENDING")


@app.get("/result/{task_id}", response_model=TaskStatus)
def get_result(task_id: str) -> TaskStatus:
    result = vectorize_task.AsyncResult(task_id)
    if result.state == "SUCCESS":
        return TaskStatus(task_id=task_id, status="SUCCESS", svg=result.result)
    if result.state == "FAILURE":
        return TaskStatus(task_id=task_id, status="FAILURE", error=str(result.result))
    return TaskStatus(task_id=task_id, status=result.state)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
