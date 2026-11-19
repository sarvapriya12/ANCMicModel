"""Aggregation, the non-regression gate and Markdown/CSV reporting.

The gate enforces "don't decrease the output result" for every optimisation:
a candidate pipeline is accepted against a baseline only if, on the headline
(real-speech) scenes and by paired bootstrap over scenes, the 95 % lower
confidence bound of every metric's improvement stays above its tolerance.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# metric -> (higher_is_better, tolerance in metric units)
GATE_TOLERANCES: dict[str, tuple[bool, float]] = {
    "estoi": (True, 0.005),
    "si_sdr": (True, 0.2),
    "pesq": (True, 0.05),
    "f0_recall": (True, 0.01),
    "f0_gpe": (False, 0.01),
    "ltass_rmse_db": (False, 0.5),
    "fric_abs_db": (False, 1.0),
    "mod_corr": (True, 0.01),
}

HEADLINE_TIERS = ("S2", "S3")

# SI-SDR is unbounded below: in the clipping-dominated top SPL band every
# pipeline scores -25..-30 dB and those scenes would dominate any mean. As is
# standard practice the per-scene value is clipped before averaging/differencing.
METRIC_CLIPS: dict[str, tuple[float, float]] = {"si_sdr": (-15.0, 30.0)}

# SPL bands at the primary mic: 0 = 90-105, 1 = 105-120, 2 = 120-130, 3 = 130-138 dB.
# Band 3 is beyond the microphone's 120 dB overload point (>70 % of samples clip).
OPERATING_BANDS = (0, 1, 2)
SUMMARY_METRICS = (
    "si_sdr",
    "estoi",
    "pesq",
    "f0_recall",
    "f0_gpe",
    "vuv_acc",
    "ltass_rmse_db",
    "fric_abs_db",
    "mod_corr",
    "pause_db",
)

Row = dict


def read_rows(path: str | Path) -> list[Row]:
    rows: list[Row] = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            out: Row = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def write_rows(rows: list[Row], path: str | Path) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _nanmean(vals) -> float:
    a = np.asarray([v for v in vals if isinstance(v, (int, float))], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def select(rows: list[Row], **eq) -> list[Row]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def aggregate(
    rows: list[Row], by: tuple[str, ...], metrics=SUMMARY_METRICS
) -> list[Row]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row: Row = dict(zip(by, key))
        row["n"] = len(rs)
        for m in metrics:
            row[m] = _nanmean([r.get(m) for r in rs])
        out.append(row)
    return out


@dataclass
class GateResult:
    metric: str
    delta: float  # candidate - baseline, signed so that + is better
    lo: float
    hi: float
    tolerance: float
    passed: bool


def paired_bootstrap(
    deltas: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95 % CI of per-scene paired deltas."""
    d = deltas[np.isfinite(deltas)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return (
        float(d.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def gate(
    rows: list[Row],
    candidate: str,
    baseline: str,
    tiers: tuple[str, ...] = HEADLINE_TIERS,
    tolerances: dict[str, tuple[bool, float]] | None = None,
    bands: tuple[int, ...] | None = None,
) -> list[GateResult]:
    """Non-regression gate of ``candidate`` against ``baseline``.

    ``bands`` restricts the scenes to those SPL bands (default: all)."""
    tolerances = tolerances or GATE_TOLERANCES

    def keep(r: Row, name: str) -> bool:
        return (
            r["pipeline"] == name
            and r["tier"] in tiers
            and (bands is None or int(r["band"]) in bands)
        )

    def val(r: Row, metric: str) -> float:
        v = r.get(metric, np.nan)
        if metric in METRIC_CLIPS and isinstance(v, (int, float)):
            lo_, hi_ = METRIC_CLIPS[metric]
            v = min(max(v, lo_), hi_)
        return v

    cand = {r["scene_id"]: r for r in rows if keep(r, candidate)}
    base = {r["scene_id"]: r for r in rows if keep(r, baseline)}
    common = sorted(set(cand) & set(base))
    out: list[GateResult] = []
    for metric, (higher, tol) in tolerances.items():
        d = np.array(
            [
                (val(cand[s], metric) - val(base[s], metric))
                * (1.0 if higher else -1.0)
                for s in common
            ],
            dtype=float,
        )
        mean, lo, hi = paired_bootstrap(d)
        out.append(
            GateResult(metric, mean, lo, hi, tol, bool(np.isfinite(lo) and lo >= -tol))
        )
    return out


def gate_passed(results: list[GateResult]) -> bool:
    return all(r.passed for r in results)


# ---------------------------------------------------------------- Markdown


def _fmt(v, nd=3) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, float) and float(v).is_integer() and abs(v) > 1000:
        return f"{v:.0f}"
    return f"{v:.{nd}f}"


def md_table(rows: list[Row], cols: list[str], nd: int = 3) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join(
        "| " + " | ".join(_fmt(r.get(c), nd) for c in cols) + " |\n" for r in rows
    )
    return head + body


def gate_table(results: list[GateResult]) -> str:
    rows = [
        {
            "metric": r.metric,
            "delta (+=better)": r.delta,
            "95% CI low": r.lo,
            "95% CI high": r.hi,
            "tolerance": -r.tolerance,
            "result": "PASS" if r.passed else "FAIL",
        }
        for r in results
    ]
    cols = [
        "metric",
        "delta (+=better)",
        "95% CI low",
        "95% CI high",
        "tolerance",
        "result",
    ]
    return md_table(rows, cols, nd=4)


def compute_summary(rows: list[Row], pipelines: list[str]) -> list[Row]:
    out = []
    for p in pipelines:
        rs = [r for r in rows if r["pipeline"] == p]
        if not rs:
            continue
        out.append(
            {
                "pipeline": p,
                "n": len(rs),
                "latency_ms": _nanmean([r.get("latency_ms") for r in rs]),
                "block_mean_ms": _nanmean([r.get("block_mean_ms") for r in rs]),
                "block_p99_ms": _nanmean([r.get("block_p99_ms") for r in rs]),
                "block_p999_ms": _nanmean([r.get("block_p999_ms") for r in rs]),
                "rtf": _nanmean([r.get("rtf") for r in rs]),
                "unhealthy": _nanmean([r.get("unhealthy_blocks") for r in rs]),
            }
        )
    return out


def write_report(
    rows: list[Row], out_dir: str | Path, baseline: str | None = None
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(rows, out / "results.csv")
    pipelines = []
    for r in rows:
        if r["pipeline"] not in pipelines:
            pipelines.append(r["pipeline"])

    lines = ["# Evaluation report\n"]
    lines.append(
        f"Scenes: {len({r['scene_id'] for r in rows})}; pipelines: {', '.join(pipelines)}\n"
    )

    lines.append("\n## Compute and latency\n")
    lines.append(
        md_table(
            compute_summary(rows, pipelines),
            [
                "pipeline",
                "n",
                "latency_ms",
                "block_mean_ms",
                "block_p99_ms",
                "block_p999_ms",
                "rtf",
                "unhealthy",
            ],
            nd=2,
        )
    )
    lines.append("\nBlock deadline is 10.67 ms (512 samples @ 48 kHz).\n")

    for tier in ("S2", "S3", "S1"):
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        title = {
            "S1": "S1 parametric voice (stress / property tier)",
            "S2": "S2 real speech",
            "S3": "S3 real speech + Lombard",
        }[tier]
        lines.append(f"\n## {title}\n")
        lines.append(
            md_table(aggregate(sub, ("pipeline",)), ["pipeline", "n", *SUMMARY_METRICS])
        )

    sub = [r for r in rows if r["tier"] in HEADLINE_TIERS]
    if sub:
        lines.append("\n## Headline (S2+S3) by SPL band at the microphone\n")
        for pipe in pipelines:
            agg = aggregate(select(sub, pipeline=pipe), ("band",))
            lines.append(
                f"\n**{pipe}**\n\n" + md_table(agg, ["band", "n", *SUMMARY_METRICS])
            )
        lines.append("\n## Headline (S2+S3) by noise group\n")
        for pipe in pipelines:
            agg = aggregate(select(sub, pipeline=pipe), ("group",))
            lines.append(
                f"\n**{pipe}**\n\n" + md_table(agg, ["group", "n", *SUMMARY_METRICS])
            )

    if baseline:
        lines.append(
            f"\n## Non-regression gate vs `{baseline}` (headline tiers S2+S3)\n\n"
            "Paired bootstrap over scenes, 95 % lower bound must stay above -tolerance. "
            "Per-scene SI-SDR is clipped to [-15, 30] dB. The verdict uses the operating "
            "range (SPL bands 0-2); band 3 (130-138 dB) is beyond the INMP441's overload "
            "point and is reported for information.\n"
        )
        for pipe in pipelines:
            if pipe in (baseline, "raw"):
                continue
            if not any(
                r["pipeline"] == pipe and r["tier"] in HEADLINE_TIERS for r in rows
            ):
                lines.append(
                    f"\n### {pipe}: no headline (S2/S3) scenes in this run; gate not evaluated\n"
                )
                continue
            res_op = gate(rows, pipe, baseline, bands=OPERATING_BANDS)
            res_all = gate(rows, pipe, baseline)
            verdict = "ACCEPTED" if gate_passed(res_op) else "REJECTED"
            lines.append(
                f"\n### {pipe}: {verdict} (operating range, < 130 dB at the mic)\n\n"
                + gate_table(res_op)
            )
            lines.append(
                f"\nAll bands incl. 130-138 dB (information only): "
                f"{'would pass' if gate_passed(res_all) else 'would fail'}\n\n"
                + gate_table(res_all)
            )

    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path
