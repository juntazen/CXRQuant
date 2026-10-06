#!/usr/bin/env python3
"""Rebuild data/iuxray_paired.json byte for byte from the official OpenI IU-XRAY report release.

The OpenI reports are distributed under CC BY-NC-ND 4.0, so the public repository ships identifiers only:
data/corpus_image_ids.txt lists the 878 radiograph identifiers that were present locally when the corpus was built
(the first images of an interrupted download of the official image archive). The original builder kept every report
with a FINDINGS or IMPRESSION section and at least one of those images; this script applies the same rule, in the same
order and JSON layout, and checks the SHA-256 recorded in every run identity.

  curl -O https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
  mkdir -p openi && tar -xzf NLMCXR_reports.tgz -C openi
  python scripts/rebuild_corpus_from_openi.py --openi openi/ecgen-radiology --images data/corpus_image_ids.txt \
      --out data/iuxray_paired.json
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED_SHA256 = "0597c49a73f1bffcc199cd3b3f27a7c984ef5337a3efa7ccdc053a7c2d02cf23"


def parse(xml_path, present):
    root = ET.parse(xml_path).getroot()
    uid_elem = root.find(".//uId")
    uid = uid_elem.get("id") if uid_elem is not None else ""
    abstract = {}
    for ab in root.findall(".//AbstractText"):
        abstract[ab.get("Label", "").upper()] = re.sub(r"\s+", " ", (ab.text or "").strip())
    mesh = [m.text.strip() for m in root.findall(".//MeSH/major") if m.text]
    images = []
    for img in root.findall(".//parentImage"):
        iid = img.get("id", "")
        if iid in present:
            cap = img.find("caption")
            images.append({"id": iid, "path": f"data/{iid}.png",
                           "caption": cap.text.strip() if cap is not None and cap.text else ""})
    findings, impression = abstract.get("FINDINGS", ""), abstract.get("IMPRESSION", "")
    if not findings and not impression:
        return None
    report = ""
    if findings:
        report += f"FINDINGS: {findings}"
    if impression:
        report += (" " if report else "") + f"IMPRESSION: {impression}"
    return {"uid": uid, "findings": findings, "impression": impression, "report": report,
            "indication": abstract.get("INDICATION", ""), "comparison": abstract.get("COMPARISON", ""),
            "mesh_labels": mesh, "images": images, "is_normal": "normal" in [x.lower() for x in mesh]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openi", type=Path, required=True, help="directory with the OpenI report XML files")
    ap.add_argument("--images", type=Path, required=True, help="data/corpus_image_ids.txt")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    present = {x.strip() for x in a.images.read_text().split() if x.strip()}
    records = [r for r in (parse(p, present) for p in sorted(a.openi.glob("*.xml"))) if r and r["images"]]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    ok = digest == EXPECTED_SHA256
    print(f"wrote {len(records)} records to {a.out}; sha256 {digest} "
          f"{'matches' if ok else 'DIFFERS FROM'} the run identity data_hash")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
