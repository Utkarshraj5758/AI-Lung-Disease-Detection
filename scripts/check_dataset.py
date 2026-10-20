"""Dataset audit: class counts, image sizes, patient-id coverage and patient overlap per split.

    python scripts/check_dataset.py [--set paths.data_dir=dataset] [--no-sizes]

Overlap is reported, not fixed: the split folders are used as they are on disk.
Exits with status 1 if fewer than 95% of file names yield a patient id.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter

from PIL import Image

from lungxray.config import add_config_args, config_from_args
from lungxray.data import CLASS_NAMES, SPLITS, patient_overlap, scan_dataset

MIN_PATIENT_ID_COVERAGE = 0.95


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--no-sizes", action="store_true", help="skip reading image sizes")
    add_config_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    records = scan_dataset(config.paths.data_dir)
    print(f"Dataset: {config.paths.data_dir} ({len(records)} images)\n")

    counts = Counter((r.split, r.label) for r in records)
    print(f"{'split':6s} " + " ".join(f"{c:>10s}" for c in CLASS_NAMES) + f" {'total':>7s}")
    for split in SPLITS:
        row = [counts[(split, c)] for c in CLASS_NAMES]
        print(f"{split:6s} " + " ".join(f"{n:10d}" for n in row) + f" {sum(row):7d}")

    if not args.no_sizes:
        print("\nImage size (width x height): min / median / max")
        for split in SPLITS:
            sizes = []
            for record in records:
                if record.split == split:
                    with Image.open(record.path) as img:  # reads the header only
                        sizes.append(img.size)
            widths, heights = [w for w, _ in sizes], [h for _, h in sizes]
            print(
                f"  {split:6s} {min(widths)}x{min(heights)} / "
                f"{statistics.median(widths):.0f}x{statistics.median(heights):.0f} / "
                f"{max(widths)}x{max(heights)}"
            )

    print("\nPatient-id coverage (file names parsed)")
    for split in SPLITS:
        for label in CLASS_NAMES:
            group = [r for r in records if r.split == split and r.label == label]
            parsed = sum(r.patient is not None for r in group)
            print(f"  {split:6s} {label:10s} {parsed}/{len(group)} ({parsed / len(group):.1%})")
    coverage = sum(r.patient is not None for r in records) / len(records)
    print(f"  overall {coverage:.2%}")

    print("\nPatient overlap between splits (reported, not fixed)")
    for pair, stats in patient_overlap(records).items():
        a, b = pair.split("/")
        print(
            f"  {pair:10s} {stats['shared_patients']} shared patients "
            f"(of {stats[f'{a}_patients']} {a} / {stats[f'{b}_patients']} {b}); "
            f"{stats[f'{b}_images_from_shared_patients']} of {stats[f'{b}_images']} "
            f"{b} images belong to them"
        )

    unparsed = [r.path.name for r in records if r.patient is None]
    if unparsed:
        print(f"\nUnparsed file names (first 10): {unparsed[:10]}")
    if coverage < MIN_PATIENT_ID_COVERAGE:
        print(f"\nPatient-id coverage {coverage:.1%} is below {MIN_PATIENT_ID_COVERAGE:.0%}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
