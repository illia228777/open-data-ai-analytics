from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import psycopg

from utils.db import build_psycopg_conninfo


def infer_columns(csv_path: Path, separator: str) -> list[str]:
    head = pl.read_csv(csv_path, separator=separator, n_rows=1)
    return head.columns


def create_table(conn: psycopg.Connection, table: str, columns: list[str]) -> None:
    cols_sql = ", ".join(f'"{c}" TEXT' for c in columns)
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        cur.execute(f'CREATE TABLE "{table}" ({cols_sql})')


def copy_csv(conn: psycopg.Connection, table: str, csv_path: Path, separator: str) -> int:
    copy_sql = (
        f'COPY "{table}" FROM STDIN '
        f"WITH (FORMAT csv, DELIMITER '{separator}', HEADER true, QUOTE '\"')"
    )
    rows = 0
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy, open(csv_path, "rb") as f:
            while chunk := f.read(1 << 20):
                copy.write(chunk)
        rows = cur.rowcount
    return rows


def run(args: argparse.Namespace) -> None:
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    columns = infer_columns(args.csv, args.separator)
    print(f"Detected {len(columns)} columns: {columns[:5]}...")

    with psycopg.connect(build_psycopg_conninfo()) as conn:
        create_table(conn, args.table, columns)
        print(f"Created table '{args.table}'.")

        rows = copy_csv(conn, args.table, args.csv, args.separator)
        conn.commit()
        print(f"Loaded {rows} rows into '{args.table}'.")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "data-load",
        help="Load CSV into Postgres via COPY",
    )
    p.add_argument("--csv", required=True, type=Path, help="Path to source CSV")
    p.add_argument("--table", default="vehicles", help="Target table name")
    p.add_argument("--separator", default=";", help="CSV delimiter")
    p.set_defaults(func=run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-load", description="Data load module")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subparser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
