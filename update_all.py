#!/usr/bin/env python3
"""Prepare the four panel scripts for a new model cycle.

The plotting scripts historically embedded the cycle in filenames and labels.
This small deterministic updater keeps those values synchronized before a run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL_SPECS = {
    "hadramout_v17_3comp_panel_24h.py": "scenario_3comp_24h_{tag}.npz",
    "hadramout_v17_3comp_panel.py": "scenario_3comp_{tag}.npz",
    "hadramout_ai_3comp_panel_24h.py": "scenario_3comp_24h_ai_{tag}.npz",
    "hadramout_ai_3comp_panel.py": "scenario_3comp_ai_{tag}.npz",
}


def update_file(path: Path, date: dt.date, cycle: str, check: bool) -> bool:
    old = path.read_text(encoding="utf-8")
    tag = f"{date:%Y%m%d}{cycle}"
    filename_tag = f"{date:%Y%m%d}_{cycle}Z"
    new = old

    # Update output filenames and cached NPZ names.
    new = re.sub(r'OUTPUT = "([^\"]+)_\d{8}_\d{2}Z_panel\.png"',
                 rf'OUTPUT = "\g<1>_{filename_tag}_panel.png"', new, count=1)
    npz_name = PANEL_SPECS[path.name].format(tag=tag)
    new = re.sub(r'np\.load\("gfs_cache/v17replay/[^\"]+\.npz"\)',
                 f'np.load("gfs_cache/v17replay/{npz_name}")', new, count=1)

    # Update machine-readable cycle labels everywhere in the panel.
    new = re.sub(r"\d{8}/\d{2}Z", f"{date:%Y%m%d}/{cycle}Z", new)
    # Only replace standalone YYYYMMDD values; do not touch the YYYYMMDDHH
    # token inside an NPZ filename.
    new = re.sub(r"20\d{6}(?!\d)", f"{date:%Y%m%d}", new)

    # Update the two human-readable window endpoints, preserving 24h/10-day span.
    readable = list(re.finditer(r"\d{1,2} [A-Z][a-z]{2} \d{4} \d{2}Z", new))
    if readable:
        start = dt.datetime.combine(date, dt.time(int(cycle), tzinfo=dt.timezone.utc))
        span = dt.timedelta(days=1 if "24h" in path.name else 10)
        replacements = [start.strftime("%-d %b %Y %HZ"),
                        (start + span).strftime("%-d %b %Y %HZ")]
        for match, replacement in reversed(list(zip(readable[:2], replacements))):
            new = new[:match.start()] + replacement + new[match.end():]

    if new == old:
        return False
    if not check:
        path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="cycle date as YYYYMMDD")
    parser.add_argument("cycle", help="cycle hour, e.g. 00, 06, 12, 18")
    parser.add_argument("--check", action="store_true", help="report changes without writing")
    args = parser.parse_args()
    date = dt.datetime.strptime(args.date, "%Y%m%d").date()
    cycle = args.cycle.zfill(2)
    if cycle not in {"00", "06", "12", "18"}:
        raise SystemExit("cycle must be one of 00, 06, 12, 18")
    changed = []
    for name in PANEL_SPECS:
        if update_file(ROOT / name, date, cycle, args.check):
            changed.append(name)
    print(f"cycle={date:%Y%m%d}/{cycle}Z changed={len(changed)} check={args.check}")
    for name in changed:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
