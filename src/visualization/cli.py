from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.db import get_engine


CURRENT_YEAR = 2022


def run(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    df = pd.read_sql(f'SELECT * FROM "{args.table}"', engine)

    if "BRAND" in df.columns:
        top = df["BRAND"].value_counts().head(10)

        plt.figure()
        top.sort_values().plot(kind="barh")
        plt.title("Top 10 brands by registrations (2022)")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "top_brands_2022.png", dpi=200)
        plt.close()

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
    p.add_argument("--table", default="vehicles", help="Source table")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/figures"),
        help="Directory to save figures",
    )
    p.set_defaults(func=run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-visualize", description="Visualization module")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subparser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
