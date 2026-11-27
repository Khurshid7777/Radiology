"""D332: an ST / infarction bench from PTB-XL 1.0.3 — every ST-elevation / injury record, sampled established
infarcts by wall, the usual STEMI mimics and normal controls.

    python tools/mi_bench/fetch.py --out data/ptbxl_mi [--per-class 150] [--controls 400]

FULL: subendocardial / epicardial injury (INJAS, INJAL, INJIN, INJLA, INJIL) and STE_ (non-specific ST elevation).
SAMPLED: AMI, ASMI, IMI, ALMI, ILMI, LMI (infarct by wall), LVH, CLBBB (mimics). Controls: NORM + SR.
Licensed data: the folder is git-ignored; only scripts and summaries go to git.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("st_bench_fetch", Path(__file__).resolve().parents[1] / "st_bench" / "fetch.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)  # tools/st_bench/fetch.py (both files are named fetch.py)

st.FULL = ("INJAS", "INJAL", "INJIN", "INJLA", "INJIL", "STE_")
st.SAMPLED = ("AMI", "ASMI", "IMI", "ALMI", "ILMI", "LMI", "LVH", "CLBBB")

if __name__ == "__main__":
    st.main()
