"""Assemble a Hugging Face Space (Streamlit SDK) for the app in build/space/.

    python scripts/build_space.py --weights-repo <user>/<model-repo> [--out build/space]
    hf upload <user>/<space> build/space . --type space

The Space gets the app, the package and the config, a requirements.txt that installs
CPU-only PyTorch plus this package, and a README.md whose front matter configures the
Space. The bundle is not copied: the app downloads it from ``--weights-repo`` on start.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ("app", "src", "configs", "pyproject.toml", "LICENSE")
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
REPO_URL = "https://github.com/Utkarshraj5758/AI-Lung-Disease-Detection"

SPACE_README = """---
title: Chest X-ray Pneumonia Classifier (research)
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: {sdk_version}
app_file: app/streamlit_app.py
pinned: false
license: mit
short_description: Pediatric chest X-ray pneumonia classifier, research only
---

# Chest X-ray pneumonia classifier (research)

Research and education only; not a medical device. Trained on pediatric chest X-rays
(children aged one to five, one hospital); not validated for adults or clinical use.

Source, model card and limitations: {repo_url}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--weights-repo", required=True, help="model repo with best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "space")
    parser.add_argument("--sdk-version", help="Streamlit version (default: the installed one)")
    args = parser.parse_args(argv)
    build_space(args.out, args.weights_repo, args.sdk_version or _streamlit_version())
    print(f"Space files written to {args.out}")
    return 0


def build_space(out: Path, weights_repo: str, sdk_version: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for name in COPY:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, out / name, ignore=ignore)
        else:
            shutil.copy2(source, out / name)

    config = out / "configs" / "default.yaml"
    text = config.read_text(encoding="utf-8")
    if 'hf_repo_id: ""' not in text:
        raise SystemExit('configs/default.yaml has no empty app.hf_repo_id ("") to fill in')
    config.write_text(text.replace('hf_repo_id: ""', f'hf_repo_id: "{weights_repo}"', 1), "utf-8")

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    deps = [line for line in requirements if line.strip() and not line.startswith("#")]
    (out / "requirements.txt").write_text(
        "\n".join([f"--extra-index-url {CPU_TORCH_INDEX}", *deps, "."]) + "\n", "utf-8"
    )
    readme = SPACE_README.format(sdk_version=sdk_version, repo_url=REPO_URL)
    (out / "README.md").write_text(readme, encoding="utf-8")


def _streamlit_version() -> str:
    try:
        return version("streamlit")
    except PackageNotFoundError:
        raise SystemExit("streamlit is not installed; pass --sdk-version") from None


if __name__ == "__main__":
    sys.exit(main())
