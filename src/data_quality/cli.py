from __future__ import annotations

import argparse

import pandas as pd

from utils.db import get_engine


def run(args: argparse.Namespace) -> None:
    engine = get_engine()
    cols = ["BRAND", "MAKE_YEAR", "PERSON", "CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "BODY", "FUEL", "N_REG_NEW"]
    col_list = ", ".join(f'"{c}"' for c in cols)
    df = pd.read_sql(f'SELECT {col_list} FROM "{args.table}"', engine)

    print(f"Shape: {df.shape}")

    print("\nBasic statistics:")
    print(df.describe(include="all"))

    print("\nMissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nDuplicate rows:", int(df.duplicated().sum()))

    for col in ["MAKE_YEAR", "CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"\n{col}: non-null={int(s.notna().sum())}, min={s.min()}, max={s.max()}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data-quality", help="Run basic data quality checks")
    p.add_argument("--table", default="vehicles", help="Source table")
    p.set_defaults(func=run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-quality", description="Data quality module")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subparser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
