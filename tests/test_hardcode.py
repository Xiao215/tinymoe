"""No hard-coded devices or machine paths outside ``hardware.py`` and the config schema.

Every device decision goes through ``tinymoe/hardware.py`` (capabilities -> runtime) and
every path comes from config, so the same code runs on CPU, MPS, CUDA and multi-GPU. This
scans Python *tokens* (string literals and attribute chains, not comments or prose), so a
docstring may say "MPS" while code may not branch on it.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {
    ROOT / "tinymoe" / "hardware.py",  # the one place devices are named
    ROOT / "tinymoe" / "config.py",  # the schema lists the allowed option values
}
DEVICE_LITERAL = re.compile(r"^(cpu|mps|cuda|nccl|gloo)(:\d+)?$")
ABS_PATH = re.compile(r"^(/(Users|home|tmp|private|var|opt|mnt|Volumes)\b|[A-Za-z]:\\)")
DEVICE_API = re.compile(r"\btorch\s*\.\s*(cuda|mps|backends\s*\.\s*(mps|cuda))\b|\.cuda\(")


def sources() -> list[Path]:
    files = [*(ROOT / "tinymoe").rglob("*.py"), *(ROOT / "viewer" / "server").rglob("*.py")]
    return sorted(f for f in files if f not in ALLOWED)


def offences(path: Path, root: Path = ROOT) -> list[str]:
    text = path.read_text()
    rel = path.relative_to(root)
    out = []
    code_only = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.STRING:
            try:
                value = ast.literal_eval(tok.string)
            except (ValueError, SyntaxError):  # f-strings: check their literal text
                value = tok.string
            if isinstance(value, str) and (DEVICE_LITERAL.match(value) or ABS_PATH.match(value)):
                out.append(f"{rel}:{tok.start[0]}: literal {value!r}")
        elif tok.type != tokenize.COMMENT:
            code_only.append(tok.string)
    for m in DEVICE_API.finditer(" ".join(code_only)):
        out.append(f"{rel}: device API {m.group(0)!r}")
    return out


def test_scanner_catches_offences(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        'import torch\nx = torch.device("cuda")\ny = "/Users/me/data"\n'
        "torch.cuda.synchronize()\n# mps in a comment is fine\n"
    )
    found = [o.split(": ", 1)[1] for o in offences(bad, tmp_path)]
    assert "literal 'cuda'" in found and "literal '/Users/me/data'" in found
    assert any(f.startswith("device API") for f in found)
    assert len(found) == 3


def test_no_hard_coded_devices_or_paths() -> None:
    found = [o for f in sources() for o in offences(f)]
    assert not found, "hard-coded device/path outside hardware.py / config:\n" + "\n".join(found)
