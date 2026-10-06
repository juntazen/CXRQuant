#!/usr/bin/env python3
"""Bridge to the original CheXpert labeller (Irvin et al., AAAI 2019).

The reference labeller (stanfordmlgroup/chexpert-labeler with NegBio, BLLIP and
Stanford CoreNLP) requires its own Python 3.6 environment, so it is driven as an
external tool in three explicit steps:

``export``  writes every unique text of a run (references, all predictions and the
            extractor-validation texts) to headerless single-column CSV shards;
``label``   runs ``label.py`` on each shard in parallel inside that environment;
``import``  reads the labelled shards back in input order and stores
            ``chexpert_labels.json`` keyed by the SHA-256 of the text, using the
            same {1, 0, -1, None} encoding as ``fact_extractor.extract_labels``.

Nothing is imputed: a shard whose row count differs from its input fails import.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from cxrquant.clinical_safety.fact_extractor import CHEXPERT_CATEGORIES
from cxrquant.paths import REPO_ROOT
from cxrquant.runio import atomic_json, sha256_file


def text_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def run_texts(run, data):
    manifest = json.loads((run / "manifest.json").read_text())
    texts = set(manifest["references"])
    for model in manifest["models"]:
        for cfg in manifest["configs"]:
            texts.update(json.loads((run / "models" / model / (cfg + ".json")).read_text())["predictions"])
    spec = importlib.util.spec_from_file_location("extractor_comparison", REPO_ROOT / "scripts" / "extractor_comparison.py")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    comparator.DATA = data
    texts.update(t["text"] for t in comparator.load_test())
    return sorted(texts)


def export(args):
    texts = run_texts(args.run_dir.resolve(), args.data)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    shards = [texts[i :: args.shards] for i in range(args.shards)]
    index = []
    for i, shard in enumerate(shards):
        path = work / f"shard_{i:03d}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            for t in shard:
                # The labeller treats an empty row as missing; keep a stable placeholder.
                writer.writerow([" ".join(t.split()) or "."])
        index.append({"shard": path.name, "keys": [text_key(t) for t in shard]})
    atomic_json(work / "index.json", {"n_texts": len(texts), "shards": index})
    print(f"exported {len(texts)} texts into {len(shards)} shards")


def label(args):
    work = args.work_dir.resolve()
    index = json.loads((work / "index.json").read_text())
    env = dict(os.environ, PATH=str(args.env / "bin") + os.pathsep + os.environ["PATH"], JAVA_HOME=str(args.env),
               PYTHONPATH=str(args.negbio), NLTK_DATA=str(args.nltk_data))

    def one(item):
        src = work / item["shard"]
        out = work / ("labeled_" + item["shard"])
        if out.exists():
            return item["shard"], 0
        with (work / (item["shard"] + ".log")).open("w") as log:
            code = subprocess.run([str(args.env / "bin" / "python"), "label.py", "--reports_path", str(src),
                                   "--output_path", str(out)], cwd=args.labeler, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        return item["shard"], code

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        codes = dict(ex.map(one, index["shards"]))
    failed = {k: v for k, v in codes.items() if v}
    print(json.dumps({"failed": failed, "n_shards": len(codes)}))
    return 1 if failed else 0


def parse(value):
    if value == "":
        return None
    v = float(value)
    if v not in (1.0, 0.0, -1.0):
        raise ValueError(f"unexpected label {value}")
    return int(v)


def import_labels(args):
    work = args.work_dir.resolve()
    index = json.loads((work / "index.json").read_text())
    labels = {}
    for item in index["shards"]:
        with (work / ("labeled_" + item["shard"])).open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(item["keys"]):
            raise ValueError(f"{item['shard']}: {len(rows)} labelled rows for {len(item['keys'])} inputs")
        for key, row in zip(item["keys"], rows, strict=True):
            labels[key] = {c: parse(row[c]) for c in CHEXPERT_CATEGORIES}
    if len(labels) != index["n_texts"]:
        raise ValueError("label count differs from exported text count")
    run = args.run_dir.resolve()
    atomic_json(run / "chexpert_labels.json", {
        "provenance": {
            "tool": "stanfordmlgroup/chexpert-labeler (master) with NegBio (master), BLLIP GENIA+PubMed, Stanford CoreNLP",
            "labeler_sample_check": "official sample_reports.csv reproduced labeled_reports.csv byte-for-byte",
            "environment": "conda env from chexpert-labeler environment.yml (Python 3.6.7) + openjdk 8",
            "index_hash": sha256_file(work / "index.json"),
            "script_hash": sha256_file(__file__),
            "n_texts": len(labels),
        },
        "labels": labels,
    })
    print(f"imported {len(labels)} label vectors")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "label", "import"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--work-dir", type=Path, required=True)
        p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "iuxray_paired.json")
        if name == "export":
            p.add_argument("--shards", type=int, default=48)
        if name == "label":
            p.add_argument("--env", type=Path, required=True)
            p.add_argument("--labeler", type=Path, required=True)
            p.add_argument("--negbio", type=Path, required=True)
            p.add_argument("--nltk-data", type=Path, required=True)
            p.add_argument("--jobs", type=int, default=48)
    args = ap.parse_args(argv)
    return {"export": export, "label": label, "import": import_labels}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
