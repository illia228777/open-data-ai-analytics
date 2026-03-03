from __future__ import annotations

import argparse

from .data_load import add_subparser as add_data_load_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analytics",
        description="Open Data AI Analytics CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_data_load_subparser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.func(args)