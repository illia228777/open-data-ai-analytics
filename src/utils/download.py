import argparse
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError



def build_download_url(file_id: str) -> str:
    """Build a direct download URL for a public Google Drive file."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download(file_id: str, output_path: Path) -> None:
    """Download a file from Google Drive, handling the large-file confirmation page."""
    url = build_download_url(file_id)

    print(f"Downloading from Google Drive (file ID: {file_id}) ...")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urlopen(req, timeout=120)

        # Google Drive shows a virus-scan warning page for large files.
        # If we get HTML instead of binary data, we need to confirm the download.
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            print("Large file detected, confirming download ...")
            confirm_url = f"{url}&confirm=t"
            req = Request(confirm_url, headers={"User-Agent": "Mozilla/5.0"})
            response = urlopen(req, timeout=300)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = 0
        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                print(f"\r  Downloaded {total / 1024 / 1024:.1f} MB", end="", flush=True)

        print(f"\n  Saved to {output_path}")

    except HTTPError as e:
        print(f"\nHTTP Error {e.code}: {e.reason}")
        print("Make sure the file is shared as 'Anyone with the link'.")
        sys.exit(1)
    except URLError as e:
        print(f"\nNetwork error: {e.reason}")
        sys.exit(1)


def run(args: argparse.Namespace):
    download(args.file_id, args.output)

def add_download_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("download", help="Download dataset from Google Drive")
    p.add_argument("--file-id", type=str, default='1w15cItMy_FuM_J-7qHr2j2zirANyczQr', help="Google Drive file ID")
    p.add_argument("--output", "-o", type=Path, default=Path("./data/dataset.parquet"), help="Output file path")
    p.set_defaults(func=run)

