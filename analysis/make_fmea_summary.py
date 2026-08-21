#!/usr/bin/env python3
"""Parse FMEA.md, check it and emit it as data.

    python analysis/make_fmea_summary.py --out results/fmea.json

An FMEA in a markdown table rots: an entry cites a test that gets renamed, the
RPN column stops matching S x O x D after someone edits a score, two rows end
up with the same identifier.  This script is the check that stops that and CI
runs it.

It verifies that

* every RPN equals severity x occurrence x detection;
* every score is in 1..10 and every identifier is unique and sequential;
* every ``tests/...::test_name`` cited in a mitigation actually exists;
* every ``analysis/x.py`` cited actually exists.

and then writes the table out as JSON sorted by RPN, which is what the README
table and the plots read.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

ROW = re.compile(r"^\|\s*(FM-\d+)\s*\|")
TEST_REFERENCE = re.compile(r"(tests/[\w/]+\.py)(?:::(\w+))?")
SCRIPT_REFERENCE = re.compile(r"(analysis/[\w/]+\.py|src/szp/[\w/]+\.py|config/[\w.]+)")


def parse(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not ROW.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 10:
            raise SystemExit(f"{path}: row has {len(cells)} cells, expected at least 10:\n{line}")
        identifier, item, mode, effect, sev, occ, det, rpn = cells[:8]
        mitigation = cells[8]
        residual = cells[9]
        rows.append(
            {
                "id": identifier,
                "item": item,
                "failure_mode": mode,
                "effect": effect,
                "severity": int(sev),
                "occurrence": int(occ),
                "detection": int(det),
                "rpn": int(rpn),
                "mitigation": mitigation,
                "residual_risk": residual,
                "tests": [
                    f"{file}::{name}" if name else file
                    for file, name in TEST_REFERENCE.findall(mitigation + " " + residual)
                ],
                "references": SCRIPT_REFERENCE.findall(mitigation + " " + residual),
            }
        )
    return rows


def check(rows: List[Dict[str, object]]) -> List[str]:
    problems: List[str] = []
    if not rows:
        problems.append("no FM- rows found at all")
    seen = set()
    for index, row in enumerate(rows, start=1):
        identifier = row["id"]
        if identifier in seen:
            problems.append(f"{identifier}: duplicate identifier")
        seen.add(identifier)
        expected = f"FM-{index:02d}"
        if identifier != expected:
            problems.append(f"{identifier}: out of order, expected {expected}")
        for column in ("severity", "occurrence", "detection"):
            value = row[column]
            if not 1 <= int(value) <= 10:
                problems.append(f"{identifier}: {column} is {value}, must be 1..10")
        product = row["severity"] * row["occurrence"] * row["detection"]
        if product != row["rpn"]:
            problems.append(
                f"{identifier}: RPN is {row['rpn']} but S x O x D is {product}"
            )
    return problems


def check_references(rows: List[Dict[str, object]]) -> List[str]:
    problems: List[str] = []
    collected = set()
    for row in rows:
        for reference in row["references"]:
            path = REPO_ROOT / reference
            if not path.exists():
                problems.append(f"{row['id']}: cites {reference}, which does not exist")
        for test in row["tests"]:
            collected.add((row["id"], test))
    for identifier, test in sorted(collected):
        file_part, _, name = test.partition("::")
        path = REPO_ROOT / file_part
        if not path.exists():
            problems.append(f"{identifier}: cites {file_part}, which does not exist")
        elif name and name not in _test_names(path):
            problems.append(f"{identifier}: cites {test}, which is not defined in {file_part}")
    return problems


def _test_names(path: Path) -> set:
    """Test functions defined in a file.

    Read out of the source rather than by asking pytest to collect.  Collection
    imports every test module, which needs the optional dependencies installed,
    and ``--collect-only -q`` prints per-file counts rather than node ids on
    current pytest, so parsing it is fragile in both directions.
    """
    return set(re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.MULTILINE))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmea", default=str(REPO_ROOT / "analysis" / "FMEA.md"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-test-check", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rows = parse(Path(args.fmea))
    problems = check(rows)
    if not args.skip_test_check:
        problems += check_references(rows)

    if problems:
        print(f"{args.fmea}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem}")
        return 1

    ordered = sorted(rows, key=lambda row: -row["rpn"])
    payload = {
        "entries": ordered,
        "count": len(ordered),
        "highest_rpn": ordered[0]["rpn"] if ordered else 0,
        "top_five": [row["id"] for row in ordered[:5]],
        "severity_10_count": sum(1 for row in rows if row["severity"] == 10),
    }
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if not args.quiet:
        print(f"{len(rows)} failure modes, all consistent")
        print(f"{'id':<8}{'RPN':>5}  {'S':>2}{'O':>3}{'D':>3}  item / mode")
        for row in ordered:
            print(
                f"{row['id']:<8}{row['rpn']:>5}  {row['severity']:>2}"
                f"{row['occurrence']:>3}{row['detection']:>3}  "
                f"{row['item']}: {row['failure_mode']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
