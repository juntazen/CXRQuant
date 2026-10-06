"""CPU extraction, paired comparison and strict manifest verification."""

import argparse
import json
import sys
from pathlib import Path
from cxrquant.clinical_safety.fact_extractor import extract_labels
from cxrquant.clinical_safety.safety_metrics import fact_preservation
from cxrquant.validation import validate_benchmark


def _load_reports(path):
    path = Path(path)
    value = path.read_text().splitlines() if path.suffix == ".txt" else json.loads(path.read_text())
    if not isinstance(value, list) or not value or any(not isinstance(x, str) for x in value):
        raise ValueError("Expected a nonempty list of report strings")
    return value


def cmd_verify(args):
    folder = Path(args.results)
    manifest_path = Path(args.manifest) if args.manifest else folder / "manifest.json"
    matrix = validate_benchmark(
        json.loads((folder / "multimodel_benchmark.json").read_text()),
        json.loads(manifest_path.read_text()),
    )
    print(json.dumps(matrix))
    print("Complete manifest matrix; all stored clinical and NLG metrics match recomputation.")
    print("This verifies numerical consistency, not clinical safety.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cxrquant-audit")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--text")
    e.add_argument("--json", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--quantized", required=True)
    c.add_argument("--uncertain", choices=["positive", "negative"], default="positive")
    c.add_argument("--json", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--results", required=True)
    v.add_argument("--manifest")
    args = ap.parse_args(argv)
    try:
        if args.command == "verify":
            return cmd_verify(args)
        if args.command == "extract":
            result = extract_labels(args.text if args.text is not None else sys.stdin.read())
        else:
            result = fact_preservation(
                [extract_labels(x) for x in _load_reports(args.quantized)],
                [extract_labels(x) for x in _load_reports(args.baseline)],
                args.uncertain,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as e:
        print("Validation failed: " + str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
