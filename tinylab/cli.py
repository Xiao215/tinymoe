"""``tinylab`` command line: info, stage, pipeline, bench, viewer, chat, inspect, report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from tinylab.config import HardwareConfig, load_stage_config
from tinylab.logging import setup_logging


def _cmd_info(args: argparse.Namespace) -> int:
    from tinylab import hardware

    cfg = load_stage_config(args.overrides).hardware if args.overrides else HardwareConfig()
    report = {"backends": hardware.describe(), "runtime": hardware.resolve_runtime(cfg).summary()}
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinylab", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="print detected capabilities and the resolved runtime")
    p.add_argument("overrides", nargs="*", help="e.g. hardware=mac_m1_16gb hardware.precision=fp32")
    p.set_defaults(fn=_cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    code: int = args.fn(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
