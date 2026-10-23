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
    b.add_argument("--real", default=None,
                   help="data/real-style folder: <device>/<id>.jpg + digital original + device.yaml (M5)")
    b.add_argument("--assume-layout", default="3x4+rhythm", help="layout of the --images pages")
    b.add_argument("--assume-speed", type=float, default=25.0, help="paper speed of the --images pages")
    b.add_argument("--assume-gain", type=float, default=10.0, help="gain of the --images pages")
    g = sub.add_parser("generate", help="only render the cases of a suite")
    g.add_argument("--suite", default="standard")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--slices", nargs="*", default=None)
    bd = sub.add_parser("board", help="render a printable capture board (SD-065)")
    bd.add_argument("--name", default="a3", help="board in configs/scanner/boards/")
    bd.add_argument("--out", required=True, help=".pdf (prints at actual size) or .png")
    bd.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    if args.cmd == "board":
        from ecgscan.board import load_board, render_board

        img = render_board(load_board(args.name), px_per_mm=args.dpi / 25.4)
        if args.out.lower().endswith(".pdf"):
            from PIL import Image

            Image.fromarray(img).save(args.out, resolution=float(args.dpi))   # page size = board size
        else:
            import cv2

            cv2.imwrite(args.out, img)
        print(f"{args.name}: {img.shape[1]}x{img.shape[0]} px at {args.dpi} dpi -> {args.out}; print at 100 %")
        return 0

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
    if args.real:
        from ecgscan.bench.real import real_cases

        cases = real_cases(args.real)
        if not cases:
            print(f"no photo + digital original pairs found in {args.real}", file=sys.stderr)
            return 2
    summary = run_bench(args.suite, args.out, cases_dir=args.cases, digitizers=digs, cases_per_slice=args.n,
                        slices=args.slices, cases=cases, resume=args.resume)
    print(f"report: {args.out}/report.md ({summary['n_cases']} cases, {summary['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
