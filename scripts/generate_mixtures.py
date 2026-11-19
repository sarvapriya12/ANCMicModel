"""Generate the synthetic dual-microphone scene sets.

    # frozen evaluation set (needs LibriSpeech test-clean for tiers S2/S3)
    uv run python scripts/generate_mixtures.py --set eval_v1 --workers 8

    # tiny CI/smoke subset (parametric speech only, no downloads)
    uv run python scripts/generate_mixtures.py --set tiny

Scenes are fully determined by (base seed, index), so the set can be rebuilt
bit-for-bit; ``data/manifests/<set>.json`` records every spec plus the sha256
of every audio file so drift in the generator or its libraries is detected.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETS = {
    # name: (base_seed, n_scenes, duration_s, tiers, speech_root)
    "eval_v1": (20260101, 240, 10.0, ("S1", "S2", "S3"), "test-clean"),
    "tiny": (7, 6, 3.0, ("S1",), None),
}


def _render_one(args: tuple) -> tuple[str, dict[str, str], dict]:
    from highspl.data.mixture import (
        SpeechPool,
        render_scene,
        sample_scene_spec,
        write_scene,
    )

    index, base_seed, duration, tier, speech_dir, out_dir = args
    spec = sample_scene_spec(base_seed, index, duration_s=duration, tier=tier)
    pool = SpeechPool(speech_dir) if speech_dir else None
    scene = render_scene(spec, pool)
    hashes = write_scene(scene, out_dir)
    return spec.scene_id, hashes, scene.meta


def build(
    name: str, out_root: Path, workers: int, limit: int | None, verify: bool
) -> int:
    from highspl.data.mixture import GROUPS, SPL_BANDS, TIERS

    base_seed, n, duration, tiers, speech = SETS[name]
    n = min(n, limit) if limit else n
    speech_dir = None
    if speech:
        speech_dir = ROOT / "data" / "raw" / "librispeech" / "LibriSpeech" / speech
        if not speech_dir.is_dir():
            print(
                f"missing {speech_dir}; run scripts/download_data.py", file=sys.stderr
            )
            return 2

    out_dir = out_root / name
    cell = len(GROUPS) * len(SPL_BANDS)
    jobs = []
    for i in range(n):
        tier = tiers[(i // cell) % len(tiers)] if len(tiers) < len(TIERS) else None
        jobs.append(
            (
                i,
                base_seed,
                duration,
                tier,
                str(speech_dir) if speech_dir else None,
                str(out_dir),
            )
        )

    entries: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (sid, hashes, meta) in enumerate(ex.map(_render_one, jobs), 1):
            entries[sid] = {"sha256": hashes, "meta": meta}
            print(
                f"[{k:3d}/{n}] {sid}  snr={meta['snr_db']:+6.1f} dB  clip={meta['clip_fraction_front'] * 100:4.1f}%",
                flush=True,
            )

    manifest_path = ROOT / "data" / "manifests" / f"{name}.json"
    if verify and manifest_path.is_file():
        old = json.loads(manifest_path.read_text())["scenes"]
        bad = [
            s
            for s, e in entries.items()
            if s in old and old[s]["sha256"] != e["sha256"]
        ]
        if bad:
            print(
                f"HASH MISMATCH in {len(bad)} scenes, e.g. {bad[:3]}", file=sys.stderr
            )
            return 1
        print("all hashes match the committed manifest")
        return 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "set": name,
                "base_seed": base_seed,
                "duration_s": duration,
                "scenes": entries,
            },
            indent=1,
        )
    )
    print(f"wrote {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="tiny", choices=list(SETS))
    ap.add_argument("--out", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, default=None, help="render only the first N scenes"
    )
    ap.add_argument(
        "--verify", action="store_true", help="compare hashes to the committed manifest"
    )
    args = ap.parse_args(argv)
    return build(args.set, Path(args.out), args.workers, args.limit, args.verify)


if __name__ == "__main__":
    sys.exit(main())
