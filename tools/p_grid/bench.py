"""D316: complete AV block with and without the atrial-grid route, per record (codes, alerts only).

    python tools/p_grid/bench.py {avb,ptb} --out rows.jsonl [--shard k/n]
"""
import argparse, json, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))


def main():
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    import importlib
    M = importlib.import_module("cardiolens.layer3_measure.measure")
    from cardiolens.pipeline import run_pipeline
    from p_grid.dump import items
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["avb", "ptb"]); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    with a.out.open("w") as fh:
        for path, ident, labels, age, odd in items(a.which)[k::n]:
            row = {"id": str(ident), "labels": labels, "odd": odd}
            try:
                sig = load_wfdb(path)
                try:
                    sig.patient.age_years = float(age) if age not in (None, "") else None
                except (TypeError, ValueError):
                    pass
                for flag in (False, True):
                    M.P_GRID_BLOCK = flag
                    res = run_pipeline(sig, mode="A")
                    row["on" if flag else "off"] = {"codes": sorted(res.findings.codes()),
                                                    "alerts": sorted({x.code for x in res.safety.alerts})}
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
