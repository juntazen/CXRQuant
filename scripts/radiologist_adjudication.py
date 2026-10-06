#!/usr/bin/env python3
"""Quality control of returned reader workbooks and creation of the adjudication workbook.

  # 1) check that a returned workbook is complete (run for each reader)
  python scripts/radiologist_adjudication.py check --workbook reader_R1.xlsx

  # 2) build the adjudicator workbook with only the items on which Reader 1 and Reader 2 disagree
  python scripts/radiologist_adjudication.py build --r1 reader_R1.xlsx --r2 reader_R2.xlsx --out reader_ADJ.xlsx

The adjudication workbook keeps the reader-workbook column layout (so scripts/radiologist_analysis.py reads it
unchanged) and adds, to the right, the two readers' answers for each disputed field. Part A disputes: Q1 or Q3.
Part B disputes: any category whose positive/uncertain versus negative/not-mentioned state differs.
"""
import argparse
import sys

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

CATS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
VALID_A = {4: {"yes", "no"}, 6: {"none", "minor", "significant"}, 7: {"neither", "A", "B", "both"}}
VALID_B = {"positive", "negative", "uncertain", "not mentioned"}
POS = {"positive": 1, "uncertain": 1, "negative": 0, "not mentioned": 0}


def rows(ws):
    return [list(r) for r in ws.iter_rows(min_row=2, values_only=True) if r and r[0]]


def check(path):
    wb = load_workbook(path, data_only=True)
    problems = []
    for r in rows(wb["Part A"]):
        for col, ok in VALID_A.items():
            v = r[col].strip() if isinstance(r[col], str) else r[col]
            if v not in ok:
                problems.append(f"Part A {r[0]}: column {'EFGH'[col - 4]} = {r[col]!r}")
        if r[4] == "yes" and not (r[5] or "").strip():
            problems.append(f"Part A {r[0]}: Q1 = yes but Q2 (which findings) is empty")
        if r[4] == "no" and r[6] not in (None, "none"):
            problems.append(f"Part A {r[0]}: Q1 = no but Q3 = {r[6]!r} (expected 'none')")
    for r in rows(wb["Part B"]):
        for c, v in zip(CATS, r[2:15]):
            if v not in VALID_B:
                problems.append(f"Part B {r[0]}: {c} = {v!r}")
    n_a, n_b = len(rows(wb["Part A"])), len(rows(wb["Part B"]))
    print(f"{path}: Part A {n_a} items, Part B {n_b} items, {len(problems)} problem(s)")
    for p in problems[:200]:
        print("  -", p)
    return len(problems)


def build(a):
    w1, w2 = load_workbook(a.r1, data_only=True), load_workbook(a.r2, data_only=True)
    out = load_workbook(a.r1)  # same layout and validation lists
    A1 = {r[0]: r for r in rows(w1["Part A"])}
    A2 = {r[0]: r for r in rows(w2["Part A"])}
    B1 = {r[0]: r for r in rows(w1["Part B"])}
    B2 = {r[0]: r for r in rows(w2["Part B"])}
    wa, wb_ = out["Part A"], out["Part B"]
    red = PatternFill("solid", fgColor="FDE9E7")
    hdr = Font(bold=True)
    # Part A: keep disputed rows, clear answers, append readers' answers
    keep_a = [i for i in A1 if A1[i][4] != A2[i][4] or A1[i][6] != A2[i][6]]
    for r in range(wa.max_row, 1, -1):
        if wa.cell(r, 1).value not in keep_a:
            wa.delete_rows(r)
    for j, name in enumerate(["R1 Q1", "R2 Q1", "R1 Q3", "R2 Q3", "R1 Q2", "R2 Q2"], start=10):
        wa.cell(1, j, name).font = hdr
    for r in range(2, wa.max_row + 1):
        i = wa.cell(r, 1).value
        for col in (5, 6, 7, 8, 9):
            wa.cell(r, col).value = None
        vals = [A1[i][4], A2[i][4], A1[i][6], A2[i][6], A1[i][5], A2[i][5]]
        for j, v in enumerate(vals, start=10):
            c = wa.cell(r, j, v)
            c.fill = red
            c.alignment = Alignment(wrap_text=True, vertical="top")
    # Part B: keep rows with any binary disagreement, clear all, show both readers for disputed categories
    keep_b = [i for i in B1 if any(POS.get(x) != POS.get(y) for x, y in zip(B1[i][2:15], B2[i][2:15]))]
    for r in range(wb_.max_row, 1, -1):
        if wb_.cell(r, 1).value not in keep_b:
            wb_.delete_rows(r)
    wb_.cell(1, 17, "Disputed categories: Reader 1 / Reader 2").font = hdr
    wb_.column_dimensions["Q"].width = 70
    for r in range(2, wb_.max_row + 1):
        i = wb_.cell(r, 1).value
        notes = []
        for k, c in enumerate(CATS):
            x, y = B1[i][2 + k], B2[i][2 + k]
            cell = wb_.cell(r, 3 + k)
            if POS.get(x) != POS.get(y):
                cell.value = None
                cell.fill = red
                notes.append(f"{c}: {x} / {y}")
            else:
                cell.value = x  # agreed value kept, adjudicator need not change it
        wb_.cell(r, 17, "; ".join(notes)).alignment = Alignment(wrap_text=True, vertical="top")
    ins = out["Instructions"]
    ins["A1"] = "CXRQuant adjudication - you receive only the items on which the two readers disagreed."
    ins["A2"] = "Adjudicator ID: ____________    Date: ____________"
    ins["A4"] = ("Part A: answer Q1-Q4 for each row as in the reader protocol. Columns J-O show the two readers' "
                 "answers for reference; your answer is final.")
    ins["A5"] = ("Part B: only the red cells need an answer; the other cells are the two readers' agreed value. "
                 "Column Q lists the readers' answers for the red cells.")
    for r in range(6, 20):
        ins.cell(r, 1).value = None
    out.save(a.out)
    print(f"adjudication workbook: {len(keep_a)} Part A items, {len(keep_b)} Part B items -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--workbook", required=True)
    b = sub.add_parser("build")
    b.add_argument("--r1", required=True)
    b.add_argument("--r2", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(1 if check(a.workbook) else 0)
    build(a)


if __name__ == "__main__":
    main()
