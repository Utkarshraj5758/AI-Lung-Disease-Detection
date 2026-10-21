"""Download a model bundle from a Hugging Face model repo and check that it loads.

    python scripts/download_weights.py --repo-id <user>/<model-repo> [--out artifacts/best.pt]

``--repo-id`` defaults to ``$LUNGXRAY_HF_REPO``, then ``app.hf_repo_id`` in the config.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from lungxray.config import add_config_args, config_from_args
from lungxray.inference import Predictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--repo-id", help="Hugging Face model repo, e.g. user/lungxray-model")
    parser.add_argument("--filename", default="best.pt", help="file in the repo (default best.pt)")
    parser.add_argument("--revision", help="branch, tag or commit (default: main)")
    parser.add_argument("--out", type=Path, help="destination (default <artifacts_dir>/best.pt)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    add_config_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)

    repo_id = args.repo_id or os.environ.get("LUNGXRAY_HF_REPO") or config.app.hf_repo_id
    if not repo_id:
        parser.error("no repo: pass --repo-id, set LUNGXRAY_HF_REPO, or set app.hf_repo_id")
    out = args.out or Path(config.paths.artifacts_dir) / "best.pt"
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to replace it", file=sys.stderr)
        return 1

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo_id, filename=args.filename, revision=args.revision)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, out)
    try:
        predictor = Predictor(out, device="cpu")
    except (ValueError, FileNotFoundError) as e:
        out.unlink()
        print(f"Downloaded file is not a usable bundle: {e}", file=sys.stderr)
        return 1
    meta = predictor.meta
    print(
        f"Saved {predictor.model_version} to {out} (created {meta['created_at']}, threshold "
        f"{predictor.threshold:.3f}, temperature {predictor.temperature:.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
