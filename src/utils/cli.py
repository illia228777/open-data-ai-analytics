from __future__ import annotations

import argparse

from .download import add_download_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="utils",
        description="Utilities CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_download_subparser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.func(args)