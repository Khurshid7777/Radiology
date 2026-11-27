"""D358 acceptance: AF withdrawn on the same conducted P (pipeline.AF_SAME_P_WITHDRAW) — off vs on.

Each record runs through the site pipeline with the flag off; only where AF / possible AF is named does it run again
with the flag on (the rule does nothing elsewhere). One JSON line per record (labels, codes off/on, alerts off/on) —
numbers and codes only, no signal; the output stays in the given folder (git-ignored data, never committed).

    python tools/atrial_map/af_same_p_bench.py SET K N OUT_DIR
SET: rhythm_odd | rhythm_even | fresh | fresh2 | afl | afl_even (Chapman svt_bench AFL) | af_bench (D359)."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cardiolens.pipeline as P
from cardiolens.layer1_ingestion import load_wfdb

AF = {"atrial_fibrillation", "possible_atrial_fibrillation"}


def items(name: str):
    if name in ("rhythm_odd", "rhythm_even", "fresh", "fresh2"):
        d = {"fresh": "ptbxl_fresh", "fresh2": "ptbxl_fresh2"}.get(name, "ptbxl_rhythm")
        for r in json.loads((ROOT / "data" / d / "manifest.json").read_text())["records"]:
            if name.startswith("rhythm") and int(r["ecg_id"]) % 2 != (name == "rhythm_odd"):
                continue
            yield str(r["ecg_id"]), r["labels"], ROOT / "data" / d / r["filename_hr"], r.get("age")
    elif name in ("af_bench", "af_bench2"):
        for r in json.loads((ROOT / "data" / name / "manifest.json").read_text())["records"]:
            yield r["name"], r["labels"], ROOT / "data" / name / r["folder"] / r["name"], r.get("age")
    elif name in ("afl", "afl_even"):
        for r in json.loads((ROOT / "data/svt_bench/manifest.json").read_text())["records"]:
            if "AFL" in r["labels"] and (name == "afl" or int(r["name"][2:]) % 2 == 0):
                yield r["name"], r["labels"], ROOT / "data/svt_bench" / r["folder"] / r["name"], r.get("age")
    else:
        raise SystemExit(f"unknown set {name}")


def run(path: Path, age, flag: bool):
    P.AF_SAME_P_WITHDRAW = flag
    sig = load_wfdb(str(path))
    try:
        sig.patient.age_years = float(age) if age not in (None, "") else None
    except ValueError:
        pass
    res = P.run_pipeline(sig, mode="A")
    return sorted(res.findings.codes()), sorted({a.code for a in res.safety.alerts})


def main() -> None:
    name, k, n, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_{k}of{n}.jsonl"
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    with out.open("a") as fh:
        for i, (rid, labels, path, age) in enumerate(items(name)):
            if i % n != k or rid in done:
                continue
            row = {"id": rid, "labels": labels}
            try:
                off = run(path, age, False)
                on = run(path, age, True) if AF & set(off[0]) else off
                row.update(codes_off=off[0], alerts_off=off[1], codes_on=on[0], alerts_on=on[1])
            except Exception as ex:  # a record that fails is counted, not fatal
                row["error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
            fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
