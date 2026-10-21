"""Standalone scanner API: ``uvicorn ecgscan_service.app:app`` (deployment without interpretation)."""

from fastapi import FastAPI

from cardiolens.api.guard import RequestGuard
from ecgscan import PIPELINE_VERSION
from ecgscan.config import load_yaml
from ecgscan_service.router import router

app = FastAPI(title="ECG Scanner", version=PIPELINE_VERSION)
# Key check and body cap before multipart parsing (D136): the upload limit plus form fields.
app.add_middleware(RequestGuard, max_body_bytes=int(load_yaml("service")["upload"]["max_mb"] * 1024 * 1024) + 1024 * 1024)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}
