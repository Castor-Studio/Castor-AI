"""Writes the final ai_status.json contract file consumed by the front-end
installer/app to decide whether to offer "run AI locally".

Called at the end of the Inno Setup install with the resource report
produced earlier by detect_resources.py and the user's override choice
(only relevant when tier == "low").

Usage:
    python write_status.py --report resource_report.json --confirmed true|false

Writes to: %ProgramData%\\CastoStudio\\ai_status.json
(falls back to --out PATH for local testing on non-Windows).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys


def default_status_path() -> str:
    if platform.system() == "Windows":
        program_data = os.environ.get("ProgramData", r"C:\ProgramData")
        return os.path.join(program_data, "CastoStudio", "ai_status.json")
    # Non-Windows dev fallback.
    return os.path.join(os.path.expanduser("~"), ".castostudio", "ai_status.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to the resource_check JSON report.")
    parser.add_argument(
        "--confirmed",
        default="false",
        help="'true' if the user explicitly confirmed installing despite low resources.",
    )
    parser.add_argument(
        "--install-dir",
        default=None,
        help="Absolute path where the app was installed, so the front-end doesn't need to hardcode it.",
    )
    parser.add_argument("--out", help="Override the output path (for local testing).")
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    user_confirmed_override = args.confirmed.strip().lower() in ("1", "true", "yes", "oui")

    status = {
        "version": 1,
        "checked_at": report.get("checked_at"),
        "install_dir": args.install_dir,
        "cpu_cores": report.get("cpu_cores"),
        "ram_gb": report.get("ram_gb"),
        "gpu_name": report.get("gpu_name"),
        "gpu_vram_gb": report.get("gpu_vram_gb"),
        "score_percent": report.get("score_percent"),
        "tier": report.get("tier"),
        "recommended": report.get("recommended"),
        "user_confirmed_override": user_confirmed_override,
        # Single flag the front-end should actually branch on.
        "ai_ready": bool(report.get("recommended")) or user_confirmed_override,
    }

    out_path = args.out or default_status_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
