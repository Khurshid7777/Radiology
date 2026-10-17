"""Standalone scanner API: ``uvicorn ecgscan_service.app:app`` (deployment without interpretation)."""

from fastapi import FastAPI

from ecgscan import PIPELINE_VERSION
from ecgscan_service.router import router

app = FastAPI(title="ECG Scanner", version=PIPELINE_VERSION)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}
