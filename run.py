#!/usr/bin/env python3
"""Single public entry point for the reproducibility repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

HELP = """\
Safe actor-critic AER/UE reproducibility package

Usage:
  python run.py verify
      Verify the SHA-256 manifest and run all automated tests.

  python run.py reproduce [options]
      Run the complete paper-consistent benchmark. Results are written to
      outputs/reproduction unless --out is supplied.

  python run.py postprocess [options]
      Rebuild tables, figures, audits, and optionally the animation from a
      completed reproduction directory.

  python run.py animate [options]
      Rebuild only the 10-second visualization from a completed reproduction.

  python run.py sensitivity [options]
      Run the separate descriptive AER-weight sensitivity diagnostic.

Examples:
  python run.py verify
  python run.py reproduce --animate
  python run.py postprocess --out outputs/reproduction --animate
"""


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(HELP)
        return

    action, passthrough = sys.argv[1], sys.argv[2:]
    if action == "verify":
        if passthrough:
            raise SystemExit("verify does not accept additional arguments")
        run([sys.executable, str(SCRIPTS / "verify_manifest.py")])
        run([sys.executable, "-m", "unittest", "discover", "-v"])
        return

    scripts = {
        "reproduce": "run_reproducible_demo.py",
        "postprocess": "postprocess_results.py",
        "animate": "make_animation.py",
        "sensitivity": "run_posthoc_aer_weight_sensitivity.py",
    }
    script = scripts.get(action)
    if script is None:
        print(f"Unknown command: {action}\n")
        print(HELP)
        raise SystemExit(2)
    run([sys.executable, str(SCRIPTS / script), *passthrough])


if __name__ == "__main__":
    main()
