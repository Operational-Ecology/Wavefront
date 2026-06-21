"""Command-line interface: ``wavefront fetch|manifest <id>``."""
from __future__ import annotations

import argparse
import logging
import sys

from . import __version__, _art
from .client import Client
from .exceptions import WavefrontError


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}B"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wavefront", description="Fetch finwave datasets.")
    p.add_argument("--version", action="version", version=f"wavefront {__version__}")
    p.add_argument("--api-key", default=None, help="overrides $FW_API_TOKEN")
    p.add_argument("--base-url", default=None, help="finwave base URL")
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    p.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    p.add_argument("--no-art", action="store_true", help="disable the wave animation")
    sub = p.add_subparsers(dest="cmd", required=False)

    m = sub.add_parser("manifest", help="print a version's metadata + formats")
    m.add_argument("dataset_version_id")

    f = sub.add_parser("fetch", help="download + extract a dataset version")
    f.add_argument("dataset_version_id")
    f.add_argument("--format", default="yolo")
    f.add_argument("--dest", default=None, help="extract dir (default: cache)")
    f.add_argument("--force", action="store_true", help="ignore cache")

    args = p.parse_args(argv)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    show_art = not (args.no_art or args.quiet)
    if args.cmd is None:                     # bare `wavefront` → wave + wordmark
        if show_art:
            _art.banner()
        return 0
    kw = {}
    if args.base_url:
        kw["base_url"] = args.base_url
    try:
        client = Client(args.api_key, **kw)
        if args.cmd == "manifest":
            mf = client.manifest(args.dataset_version_id)
            print(f"{mf.name} (v{mf.version_number})")
            print(f"  samples: {mf.sample_count}  annotations: {mf.annotation_count}")
            print(f"  formats: {mf.available_formats or '(none generated yet)'}")
            print(f"  fingerprint: {mf.fingerprint}")
            return 0
        if args.cmd == "fetch":
            if show_art:
                _art.wave()
            last = [0.0]

            def prog(got, total):
                pct = f" {100*got/total:.0f}%" if total else ""
                if got - last[0] >= (1 << 23) or got == total:  # ~8MB steps
                    print(f"\r  downloading {_fmt_bytes(got)}{pct}", end="", file=sys.stderr)
                    last[0] = got

            ds = client.fetch(args.dataset_version_id, format=args.format,
                              dest=args.dest, force=args.force, progress=prog)
            print("", file=sys.stderr)
            print(ds.path)
            print(f"  {ds.num_images} images, {ds.num_labels} labels, classes={ds.classes}",
                  file=sys.stderr)
            return 0
    except WavefrontError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
