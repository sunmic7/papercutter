#!/usr/bin/env python3
"""Measure question--answer pairing quality against human review decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(manifest: str | Path) -> dict:
    path = Path(manifest).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training manifest was not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reviewed = [record for record in records if record.get("label_provenance") == "reviewed"]
    accepted = [record for record in reviewed if record.get("pair_valid")]
    automatic = [record for record in reviewed if (record.get("pairing") or {}).get("decision_band") == "auto"]
    automatic_accepted = [record for record in automatic if record.get("pair_valid")]
    return {
        "schema_version": 1,
        "manifest": str(path),
        "reviewed_pairs": len(reviewed),
        "accepted_pairs": len(accepted),
        "automatic_pairs": len(automatic),
        "automatic_pairs_accepted": len(automatic_accepted),
        "automatic_precision": len(automatic_accepted) / len(automatic) if automatic else None,
        "automatic_accepted_coverage": len(automatic_accepted) / len(accepted) if accepted else None,
        "manual_review_required": sum((record.get("pairing") or {}).get("decision_band") != "auto" for record in reviewed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pairing decisions using reviewed training data")
    parser.add_argument("manifest")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    result = evaluate(args.manifest)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["report"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
