"""Markdown report of a bench run, in Russian (TZ §0.6: reports in ru/uz).

TODO(M4): the Uzbek version, once the scanner's i18n files exist (SD-016).
"""

from __future__ import annotations

from pathlib import Path


def _f(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.0f} %"


COLUMNS = [
    ("n", "n", lambda a: _f(a["n"])),
    ("отказ/ошибка", "", lambda a: f"{a['rejected']}/{a['errors']}"),
    ("r", "", lambda a: _f(a["r_median"])),
    ("SNR дБ", "", lambda a: _f(a["snr_db_median"], 1)),
    ("RMSE мВ", "", lambda a: _f(a["rmse_mv_median"], 3)),
    ("ΔR мВ", "", lambda a: _f(a["r_amp_err_mv_median"], 3)),
    ("RR мс", "", lambda a: _f(a["rr_ms_mae"], 1)),
    ("QRS мс", "", lambda a: _f(a["qrs_ms_mae"], 1)),
    ("QT мс", "", lambda a: _f(a["qt_ms_mae"], 1)),
    ("PR мс", "", lambda a: _f(a["pr_ms_mae"], 1)),
    ("скорость ✓", "", lambda a: _pct(a["speed_ok"])),
    ("усиление ✓", "", lambda a: _pct(a["gain_ok"])),
    ("плохие", "", lambda a: _pct(a["bad"])),
    ("плохие без пометки", "", lambda a: _pct(a["unflagged_bad"])),
    ("время с", "", lambda a: _f(a["time_s_median"], 1)),
]


def _table(rows: list[tuple[str, dict]]) -> list[str]:
    head = "| срез | " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|---|" + "|".join(":-:" for _ in COLUMNS) + "|"
    return [head, sep] + [f"| {name} | " + " | ".join(c[2](a) for c in COLUMNS) + " |" for name, a in rows]


def write_report(summary: dict, path: str | Path) -> Path:
    lines = [
        f"# Бенчмарк сканера ЭКГ — набор `{summary['suite']}`",
        "",
        (f"Случаев: {summary['n_cases']} · версия пайплайна {summary['pipeline_version']} · git {summary['git']} · "
        f"Python {summary['python']} · {summary['generated_at']} · {summary['wall_s']} с"),
        "",
        ("Метрики (ТЗ §11, SD-010): r, SNR, RMSE и ΔR — медианы по случаям (в случае — медиана по трассам "
        "после поиска сдвига времени ±0.25 с). RR/QRS/QT/PR — средняя абсолютная ошибка, **вносимая оцифровкой**: "
        "один и тот же делинеатор CardioLens на истине в тех же окнах отведений и на результате. "
        "«Плохой» случай — ошибка RR, QRS или QT > 20 мс или интервал не измерен; «без пометки» — такой "
        "случай вернулся со статусом `ok` (нечестный исход). Ошибка оцифровщика и явный отказ — честные."),
        "",
    ]
    if summary.get("unavailable"):
        lines += ["**Не запускались:**", ""]
        lines += [f"- `{k}` — {v}" for k, v in summary["unavailable"].items()]
        lines.append("")
    if summary.get("skipped_slices"):
        lines += ["**Пропущенные срезы:**", ""]
        lines += [f"- `{k}` — {v}" for k, v in summary["skipped_slices"].items()]
        lines.append("")
    for dname in summary["digitizers"]:
        lines += [f"## `{dname}`", ""]
        rows = [(s, a) for s, a in summary["by_slice"][dname].items()]
        rows.append(("**все**", summary["overall"][dname]))
        lines += _table(rows)
        lines.append("")
        acc = summary["acceptance"].get(dname)
        if acc:
            lines += ["Критерии приёмки ТЗ §11 (срез «фото чистой бумаги»):", "",
                      "| критерий | значение | порог | |", "|---|:-:|:-:|:-:|"]
            for k, v in acc.items():
                mark = "—" if v["passed"] is None else ("✅" if v["passed"] else "❌")
                lines.append(f"| {k} | {_f(v['value'])} | {v['threshold']} | {mark} |")
            lines.append("")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
