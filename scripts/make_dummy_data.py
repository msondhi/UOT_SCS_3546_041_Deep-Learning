"""Create a tiny fake dataset for checking the code runs end to end.

    python scripts/make_dummy_data.py --out data/dummy
"""

import argparse

from deepfake_kyc.dummy_data import make_dummy_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/dummy")
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    print(f"Created {make_dummy_dataset(args.out, args.per_class, args.size)}")
