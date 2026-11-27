"""D338: ischaemia descriptors bench from PTB-XL 1.0.3 — ischaemia by wall, ST depression, non-specific ST-T, plus NORM.

    python tools/isch_bench/fetch.py --out data/ptbxl_isch [--per-class 150] [--controls 400]

Licensed data: the folder is git-ignored; only scripts and summaries go to git.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("st_bench_fetch", Path(__file__).resolve().parents[1] / "st_bench" / "fetch.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)  # tools/st_bench/fetch.py (both files are named fetch.py)

st.FULL = ()
st.SAMPLED = ("ISC_", "ISCAL", "ISCIN", "ISCAS", "ISCLA", "ISCIL", "STD_", "NST_", "INVT", "LOWT")

if __name__ == "__main__":
    st.main()
