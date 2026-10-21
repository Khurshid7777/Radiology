"""ECG scanner service: ``/scans`` API over the ``ecgscan`` pipeline (TZ §4, M4).

* ``router`` — FastAPI ``APIRouter`` (mounted into ``cardiolens.api.app``; standalone
  as ``uvicorn ecgscan_service.app:app``);
* ``service`` — submission, background processing, confirmation, analysis;
* ``store`` — SQLite job store and audit log (survives a restart, SD-005);
* ``vault`` — encrypted originals (SD-008).
"""
