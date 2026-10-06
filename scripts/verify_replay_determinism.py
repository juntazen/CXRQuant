#!/usr/bin/env python3
"""Replay a completed run twice and require byte-identical scoring artifacts."""

import argparse
import json
from pathlib import Path

from cxrquant.replay import replay
from cxrquant.runio import atomic_json, sha256_file


ARTIFACTS = (
    "bootstrap_fnf_ci.json",
    "mcnemar_pooled.json",
    "mcnemar_results.json",
    "tables/metrics.csv",
    "tables/bootstrap.csv",
    "tables/mcnemar_pooled.csv",
    "figures/fnf_matrix.png",
    "replay_status.json",
)


def hashes(run):
    return {name: sha256_file(run / name) for name in ARTIFACTS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    before = hashes(run)
    replay(run)
    middle = hashes(run)
    replay(run)
    after = hashes(run)
    if before != middle or middle != after:
        raise RuntimeError("CPU replay is not byte deterministic")
    result = {
        "status": "passed",
        "run_id": run.name,
        "replays_in_this_check": 2,
        "total_equivalent_snapshots": 3,
        "hashes": after,
        "note": "CPU scoring replay of fixed GPU predictions; not repeat training",
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
