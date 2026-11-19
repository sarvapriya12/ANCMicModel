"""Download and unpack LibriSpeech clean-speech subsets (CC BY 4.0).

    test-clean  -> speech for the frozen evaluation set (40 held-out speakers)
    dev-clean   -> speech for training / validation (a disjoint 40 speakers)

Idempotent: archives are MD5-verified and re-used; extraction is skipped when
the target directory already exists.

    uv run python scripts/download_data.py            # both subsets
    uv run python scripts/download_data.py test-clean
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "librispeech"
BASE_URL = "https://www.openslr.org/resources/12"

SUBSETS = {
    "test-clean": ("32fa31d27d2e1cad72775fee3f4849a9", 346_663_984),
    "dev-clean": ("42e2234ba48799c1f50f24a7926300a1", 337_926_286),
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> None:
    md5, _size = SUBSETS[name]
    archive = DEST / f"{name}.tar.gz"
    target = DEST / "LibriSpeech" / name
    if target.is_dir() and any(target.rglob("*.flac")):
        print(f"[skip] {name}: already extracted at {target}")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or md5sum(archive) != md5:
        print(f"[get ] {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}.tar.gz", archive)
        got = md5sum(archive)
        if got != md5:
            archive.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: checksum mismatch ({got} != {md5})")
    print(f"[unpk] {name} ...")
    with tarfile.open(archive) as tar:
        tar.extractall(DEST, filter="data")
    print(f"[ok  ] {name}: {len(list(target.rglob('*.flac')))} utterances")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("subsets", nargs="*", default=list(SUBSETS), choices=list(SUBSETS))
    args = ap.parse_args(argv)
    for name in args.subsets:
        fetch(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
