from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class VectorizeRequest(BaseModel):
    deskew_enabled: Optional[bool] = None
    binarize_method: Optional[str] = None
    angle_snap_threshold_deg: Optional[float] = None
    fidelity_level: Optional[float] = None
    output_dpi: Optional[int] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str          # PENDING | STARTED | SUCCESS | FAILURE
    svg: Optional[str] = None
    error: Optional[str] = None
