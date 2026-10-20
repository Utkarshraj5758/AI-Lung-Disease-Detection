"""Write a random-init model bundle for smoke tests and demos.

Its predictions are meaningless. Usage: python scripts/make_dummy_bundle.py --out artifacts/dummy.pt
"""

from __future__ import annotations

import argparse

from lungxray.config import add_config_args, config_from_args
from lungxray.testing import make_random_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/dummy.pt", help="output bundle path")
    parser.add_argument("--seed", type=int, default=0)
    add_config_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    path = make_random_bundle(
        args.out,
        model_name=config.model.name,
        image_size=config.data.image_size,
        resize_size=config.data.resize_size,
        seed=args.seed,
    )
    print(f"Wrote {path} (random weights, testing only)")


if __name__ == "__main__":
    main()
