from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(args.dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.dir} on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("web-serve", help="Serve a static site directory")
    p.add_argument("--dir", type=Path, default=Path("./site"), help="Directory to serve")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.set_defaults(func=run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="web", description="Web module")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subparser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
