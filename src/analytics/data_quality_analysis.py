from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run(args: argparse.Namespace) -> None:
    inp: Path = args.input

    if inp.suffix.lower() == ".parquet":
        df = pd.read_parquet(inp)
    else:
        df = pd.read_csv(inp, sep=";", low_memory=False)

    print(f"Shape: {df.shape}")
    print("\nMissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nDuplicate rows:", int(df.duplicated().sum()))

    for col in ["MAKE_YEAR", "CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"\n{col}: non-null={int(s.notna().sum())}, min={s.min()}, max={s.max()}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data-quality", help="Run basic data quality checks")
    p.add_argument("--input", required=True, type=Path, help="Path to .parquet or source .csv")
    p.set_defaults(func=run)