"""Command line: ``python -m ecgscan bench ...`` (TZ §4: CLI for batch evaluation)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ecgscan", description="ECG scanner tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="generate a synthetic suite, run the digitisers, write the report")
    b.add_argument("--suite", default="standard", help="suite name in configs/scanner/bench.yaml")
    b.add_argument("--out", required=True, help="report directory (summary.json, report.md, cases.jsonl)")
    b.add_argument("--cases", default=None, help="case directory to reuse or create (default: <out>/cases)")
    b.add_argument("--n", type=int, default=None, help="override cases per slice")
    b.add_argument("--slices", nargs="*", default=None, help="only these slices")
    b.add_argument("--digitizers", nargs="*", default=None, help="only these digitisers")
    b.add_argument("--resume", action="store_true", help="keep rows already in <out>/cases.jsonl, score the rest")
    b.add_argument("--images", default=None,
                   help="folder of images + WFDB records (ECG-Image-Database / ECG-Image-Kit) instead of a suite")
    b.add_argument("--assume-layout", default="3x4+rhythm", help="layout of the --images pages")
    b.add_argument("--assume-speed", type=float, default=25.0, help="paper speed of the --images pages")
    b.add_argument("--assume-gain", type=float, default=10.0, help="gain of the --images pages")
    g = sub.add_parser("generate", help="only render the cases of a suite")
    g.add_argument("--suite", default="standard")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--slices", nargs="*", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        from ecgscan.bench.cases import generate_suite

        cases, skipped = generate_suite(args.suite, args.out, cases_per_slice=args.n, slices=args.slices)
        print(f"{len(cases)} cases written to {args.out}; skipped: {skipped or '—'}")
        return 0
    from ecgscan.bench.cases import bench_config
    from ecgscan.bench.digitizers import default_digitizers
    from ecgscan.bench.run import run_bench

    digs = default_digitizers(bench_config())
    if args.digitizers:
        digs = [d for d in digs if d.name in args.digitizers]
    cases = None
    if args.images:
        from ecgscan.bench.cases import image_record_cases

        cases = image_record_cases(args.images, layout=args.assume_layout, speed_mm_s=args.assume_speed,
                                   gain_mm_mv=args.assume_gain)
        if not cases:
            print(f"no image + WFDB pairs found in {args.images}", file=sys.stderr)
            return 2
    summary = run_bench(args.suite, args.out, cases_dir=args.cases, digitizers=digs, cases_per_slice=args.n,
                        slices=args.slices, cases=cases, resume=args.resume)
    print(f"report: {args.out}/report.md ({summary['n_cases']} cases, {summary['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
