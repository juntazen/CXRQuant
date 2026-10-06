#!/usr/bin/env python3
"""Parse the official OpenI IU-XRAY report release into the JSON used by scripts/revision_analyses.py.

  curl -O https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
  mkdir -p openi && tar -xzf NLMCXR_reports.tgz -C openi
  python scripts/parse_openi_release.py --xml-dir openi/ecgen-radiology --out openi_parsed.json

Output: {uid: {findings, impression, n_img, mesh, imgs}} for all 3,955 reports.
Expected SHA-256 of NLMCXR_reports.tgz: 8fb6de7eec73d8c3665067ad4bb003ccd57f971ae316d2642e1627ac7268667a
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    recs = {}
    for f in sorted(a.xml_dir.glob("*.xml")):
        t = ET.parse(f).getroot()
        uid = t.find("uId").get("id")
        sec = {x.get("Label"): (x.text or "").strip() for x in t.iter("AbstractText")}
        imgs = [p.get("id") for p in t.iter("parentImage")]
        recs[uid] = {"findings": sec.get("FINDINGS", ""), "impression": sec.get("IMPRESSION", ""),
                     "n_img": len(imgs), "mesh": [m.text for m in t.iter("major")], "imgs": imgs}
    a.out.write_text(json.dumps(recs))
    print(f"parsed {len(recs)} reports -> {a.out}")


if __name__ == "__main__":
    main()
