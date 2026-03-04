from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CURRENT_YEAR = 2022


def run(args: argparse.Namespace) -> None:
    inp: Path = args.input
    out_dir: Path = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)

    # --- 1) Top brands ---
    if "BRAND" in df.columns:
        top = df["BRAND"].value_counts().head(10)

        plt.figure()
        top.sort_values().plot(kind="barh")
        plt.title("Top 10 brands by registrations (2022)")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "top_brands_2022.png", dpi=200)
        plt.close()

    # --- 2) Vehicle age distribution ---
    if "MAKE_YEAR" in df.columns:
        years = pd.to_numeric(df["MAKE_YEAR"], errors="coerce")
        years = years[(years >= 1950) & (years <= CURRENT_YEAR)]
        ages = CURRENT_YEAR - years

        plt.figure()
        plt.hist(ages.dropna(), bins=40)
        plt.title("Vehicle age distribution (2022)")
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "vehicle_age_distribution_2022.png", dpi=200)
        plt.close()

    # --- 3) Person type share ---
    if "PERSON" in df.columns:
        share = (df["PERSON"].value_counts(normalize=True) * 100).sort_index()

        plt.figure()
        share.plot(kind="bar")
        plt.title("Share by owner type (PERSON)")
        plt.ylabel("Percent (%)")
        plt.tight_layout()
        plt.savefig(out_dir / "person_type_share_2022.png", dpi=200)
        plt.close()

    print(f"Saved figures to: {out_dir}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data-visualize", help="Build and save figures")
    p.add_argument("--input", required=True, type=Path, help="Path to processed dataset (.parquet)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to save figures",
    )
    p.set_defaults(func=run)