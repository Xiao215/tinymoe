"""``tinymoe`` command line.

tinymoe info [overrides]                      detected capabilities + resolved runtime
tinymoe stage <name> [overrides]              run one stage (lineage from the manifest)
tinymoe pipeline <pipeline.yaml> [overrides]  run every stage of a pipeline
tinymoe bench <pipeline.yaml> [overrides]     time/memory estimates per stage
tinymoe report <run_dir>                      cross-stage summary table
tinymoe viewer --runs runs/                   live viewer (FastAPI + React)
tinymoe chat <checkpoint> --tokenizer <path>  chat REPL (shows <think>)
tinymoe inspect <checkpoint>                  config, parameter counts, router state
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tinymoe.config import HardwareConfig, load_stage_config
from tinymoe.logging import setup_logging


def _cmd_info(args: argparse.Namespace) -> int:
    from tinymoe import hardware

    cfg = load_stage_config(args.overrides).hardware if args.overrides else HardwareConfig()
    report = {"backends": hardware.describe(), "runtime": hardware.resolve_runtime(cfg).summary()}
    print(json.dumps(report, indent=2))
    return 0


def _cmd_stage(args: argparse.Namespace) -> int:
    from tinymoe.stages.common import run_stage
    from tinymoe.train import distributed

    run_stage(load_stage_config([f"stage={args.name}", *args.overrides]))
    distributed.shutdown()
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from tinymoe.pipeline import run_pipeline

    run_dir = run_pipeline(Path(args.path), args.overrides)
    print(f"run directory: {run_dir}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from tinymoe.eval import report

    print(report.markdown(report.write(Path(args.run_dir))))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from tinymoe.tools import bench

    return bench.main(Path(args.path), args.overrides, args.json)


def _cmd_viewer(args: argparse.Namespace) -> int:
    from viewer.server.app import serve

    serve(Path(args.runs), args.host, args.port)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from tinymoe.tools import chat

    return chat.main(
        Path(args.checkpoint), Path(args.tokenizer) if args.tokenizer else None, args.overrides
    )


def _cmd_inspect(args: argparse.Namespace) -> int:
    from tinymoe.tools import inspect_ckpt

    print(json.dumps(inspect_ckpt.describe(Path(args.checkpoint)), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinymoe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="print detected capabilities and the resolved runtime")
    p.add_argument("overrides", nargs="*", help="e.g. hardware=mac_m1_16gb hardware.precision=fp32")
    p.set_defaults(fn=_cmd_info)

    p = sub.add_parser("stage", help="run a single stage")
    p.add_argument("name", choices=["pretrain", "midtrain", "sft", "dpo", "grpo", "distill"])
    p.add_argument("overrides", nargs="*")
    p.set_defaults(fn=_cmd_stage)

    p = sub.add_parser("pipeline", help="run a pipeline YAML")
    p.add_argument("path")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(fn=_cmd_pipeline)

    p = sub.add_parser("bench", help="benchmark every stage of a pipeline")
    p.add_argument("path")
    p.add_argument("overrides", nargs="*")
    p.add_argument("--json", default=None, help="also write results to this JSON file")
    p.set_defaults(fn=_cmd_bench)

    p = sub.add_parser("report", help="write and print the cross-stage report of a run")
    p.add_argument("run_dir")
    p.set_defaults(fn=_cmd_report)

    p = sub.add_parser("viewer", help="serve the run viewer")
    p.add_argument("--runs", default="runs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=_cmd_viewer)

    p = sub.add_parser("chat", help="chat with a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--tokenizer", default=None, help="defaults to <run_dir>/tokenizer.json")
    p.add_argument("overrides", nargs="*", help="e.g. hardware=cpu_smoke eval.temperature=0.7")
    p.set_defaults(fn=_cmd_chat)

    p = sub.add_parser("inspect", help="describe a checkpoint")
    p.add_argument("checkpoint")
    p.set_defaults(fn=_cmd_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    code: int = args.fn(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
