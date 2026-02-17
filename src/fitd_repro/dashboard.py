from __future__ import annotations

import argparse
from pathlib import Path

from .dashboard_server import serve_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fitd_dashboard",
        description="Launch the FITD matrix dashboard UI.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port for the dashboard server.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root containing data/, results/, and src/.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    serve_dashboard(
        project_root=Path(args.project_root).resolve(),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
