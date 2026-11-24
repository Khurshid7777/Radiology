"""SD-150: how well the lead labels are read on synthetic strip parts (and A4 sheets of the stand). Aggregates only.

    python tools/scanner/label_check.py parts <dir made by strip_parts.py>
    python tools/scanner/label_check.py sheets <stand cases dir> <slice> [<slice> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

GROUP_OF = {"I": ("I", "II", "III"), "aVR": ("aVR", "aVL", "aVF"), "V1": ("V1", "V2", "V3"), "V4": ("V4", "V5", "V6")}


def score(visible: set[str], read: list[str]) -> dict:
    """Visible labels read right, missed, and labels read that were not there (wrong)."""
    got = set(read)
    return {"visible": len(visible), "right": len(visible & got), "wrong": len(got - visible)}


def parts(d: Path) -> dict:
    import cv2

    from ecgscan.layout.labels import groups_seen, read_lead_labels

    truth = json.loads((d / "truth.json").read_text())
    tot = {"visible": 0, "right": 0, "wrong": 0, "groups_visible": 0, "groups_named": 0, "groups_wrong": 0, "photos": 0}
    times = []
    for t in truth:
        img = cv2.imread(str(d / t["file"]))
        t0 = time.perf_counter()
        labels = read_lead_labels(img)
        times.append(time.perf_counter() - t0)
        visible = {lead for first, g in t["groups"].items() if g["label_in"] for lead in GROUP_OF[first]}
        s = score(visible, [lb.lead for lb in labels])
        for k in ("visible", "right", "wrong"):
            tot[k] += s[k]
        vis_groups = {GROUP_OF[first] for first, g in t["groups"].items() if g["label_in"]}
        named = {g for g, _ in groups_seen(labels, column_tol_px=0.05 * img.shape[1])}
        tot["groups_visible"] += len(vis_groups)
        tot["groups_named"] += len(named & vis_groups)
        tot["groups_wrong"] += len(named - set(GROUP_OF.values()) - vis_groups) + len(named - vis_groups)
        tot["photos"] += 1
    tot["seconds_median"] = round(sorted(times)[len(times) // 2], 2)
    tot["seconds_max"] = round(max(times), 2)
    return tot


def sheets(root: Path, slices: list[str]) -> dict:
    import cv2

    from ecgscan.layout import get_layout
    from ecgscan.layout.labels import read_lead_labels

    tot = {"visible": 0, "right": 0, "wrong": 0, "photos": 0}
    for sl in slices:
        for d in sorted((root / sl).glob("*/")):
            meta = json.loads((d / "case.json").read_text())
            if not meta["render"].get("labels", True):
                continue
            lay = get_layout(meta["layout"])
            visible = {lead for row in lay.rows for lead in row} | set(lay.rhythm)
            img = cv2.imread(str(d / "image.jpg"))
            s = score(visible, [lb.lead for lb in read_lead_labels(img)])
            for k in ("visible", "right", "wrong"):
                tot[k] += s[k]
            tot["photos"] += 1
    return tot


if __name__ == "__main__":
    if sys.argv[1] == "parts":
        print(json.dumps(parts(Path(sys.argv[2]))))
    else:
        print(json.dumps(sheets(Path(sys.argv[2]), sys.argv[3:])))
