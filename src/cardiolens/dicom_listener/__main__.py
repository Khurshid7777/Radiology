"""``python -m cardiolens.dicom_listener [--config path]`` — run the ECG DICOM receiver (D96)."""

from __future__ import annotations

import argparse
import logging

from cardiolens.dicom_listener import EcgListener, load_listener_config


def main() -> None:
    ap = argparse.ArgumentParser(description="DICOM C-STORE receiver for ECG devices -> CardioLens API")
    ap.add_argument("--config", help="alternative dicom_listener.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_listener_config(a.config)
    try:
        listener = EcgListener(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    listener.serve_forever()


if __name__ == "__main__":
    main()
