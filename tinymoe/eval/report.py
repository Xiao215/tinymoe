"""Cross-stage summary table (markdown + JSON) from the stage summaries in the manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COLUMNS = (
    ("val/ppl/stories", "story ppl"),
    ("val/loss/math", "math val loss"),
    ("eval/math/pass1", "math pass@1"),
    ("eval/math/pass4", "math pass@4"),
    ("eval/math/format", "format ok"),
    ("eval/math/think_len", "think len"),
    ("eval/story/mention", "story on-topic"),
)


def collect(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = []
    buckets: set[str] = set()
    for entry in manifest.get("stages", []):
        summary = entry.get("summary") or {}
        buckets |= {k.rsplit("/", 1)[1] for k in summary if k.startswith("eval/math/pass1/")}
        rows.append({"stage": entry["name"], "status": entry.get("status"), "metrics": summary})
    distill = run_dir / "artifacts" / "distill_comparison.json"
    return {
        "run_id": manifest.get("run_id"),
        "columns": [{"key": k, "label": label} for k, label in COLUMNS],
        "buckets": sorted(buckets),
        "rows": rows,
        "distill": json.loads(distill.read_text()) if distill.is_file() else None,
    }


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "-"


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# TinyMoE report - {report['run_id']}", ""]
    header = ["stage", *(c["label"] for c in report["columns"])]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in report["rows"]:
        cells = [row["stage"], *(_fmt(row["metrics"].get(c["key"])) for c in report["columns"])]
        lines.append("| " + " | ".join(cells) + " |")
    if report["buckets"]:
        lines += ["", "## pass@1 per difficulty bucket", ""]
        header = ["stage", *report["buckets"]]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for row in report["rows"]:
            cells = [
                row["stage"],
                *(_fmt(row["metrics"].get(f"eval/math/pass1/{b}")) for b in report["buckets"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    if report["distill"]:
        lines += ["", "## Distillation (student)", "", report["distill"]["markdown"]]
    return "\n".join(lines) + "\n"


def write(run_dir: Path) -> dict[str, Any]:
    report = collect(run_dir)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "artifacts" / "report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "artifacts" / "report.md").write_text(markdown(report))
    return report
