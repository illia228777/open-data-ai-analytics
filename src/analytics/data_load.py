from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(
    csv_path: Path,
    *,
    sep: str = ";",
    quotechar: str = '"',
    low_memory: bool = False,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    return pd.read_csv(
        csv_path,
        sep=sep,
        quotechar=quotechar,
        low_memory=low_memory,
        nrows=nrows,
    )


def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def run(args: argparse.Namespace) -> None:
    df = load_csv(
        args.csv,
        low_memory=args.low_memory,
        nrows=args.nrows,
    )

    print(f"Loaded shape: {df.shape}")
    print("Columns:", list(df.columns))
    print(df.head(args.head))

    if args.out is not None:
        save_parquet(df, args.out)
        print(f"Saved Parquet: {args.out}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "data-load",
        help="Load dataset from CSV and optionally export to Parquet",
    )
    p.add_argument("--csv", required=True, type=Path, help="Path to CSV (e.g. data/raw/file.csv)")
    p.add_argument("--out", type=Path, default=None, help="Optional output Parquet path")
    p.add_argument("--nrows", type=int, default=None, help="Read only first N rows")
    p.add_argument("--head", type=int, default=5, help="Print first N rows")
    p.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable pandas low_memory (may cause mixed dtype warnings)",
    )
    p.set_defaults(func=run)