"""Run the synthetic-neck generator (the submodule at tools/synthetic_neck) in
this project's environment, so it renders with this project's CUDA torch. The
submodule's own pyproject.toml and uv.lock are not used.

    uv run python tools/synthetic_neck_cli.py generate --n 8 --size 128 --out data/synthetic_neck
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "synthetic_neck" / "src"
if not (SRC / "synthetic_neck").is_dir():
    sys.exit(f"error: {SRC} is missing; run `git submodule update --init tools/synthetic_neck`")
# At import, not under __main__: with --jobs > 1 the spawned workers re-import
# this file and need the path too.
sys.path.insert(0, str(SRC))

from synthetic_neck.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
