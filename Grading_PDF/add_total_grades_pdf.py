"""
Adds total (human/TA) grades for each student into the pdf_grading_results files.

Edits in-place:
  pdf_grading_results/grading_results.csv  – adds column 'total_grade'
  pdf_grading_results/grading_results.json – adds key 'total_grade' per student

Usage:
  python3 add_total_grades_pdf.py
"""

import csv
import json

INPUT_CSV  = "pdf_grading_results/grading_results.csv"
INPUT_JSON = "pdf_grading_results/grading_results.json"

OUTPUT_CSV  = "pdf_grading_results/grading_results_with_ta_score.csv"   # in-place edit
OUTPUT_JSON = "pdf_grading_results/grading_results_with_ta_score.json"  # in-place edit

# Map student pdf_id (str) -> total grade
TOTAL_GRADES = {
    "1": 55,
    "2": 54,
    "3": 57,
}


def update_csv():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "total_grade" not in fieldnames:
        fieldnames = fieldnames + ["total_grade"]

    for row in rows:
        row["total_grade"] = TOTAL_GRADES.get(str(row["pdf_id"]), "")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV updated: {OUTPUT_CSV}")
    print(f"  Rows processed: {len(rows)}")


def update_json():
    with open(INPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    # Each value may already be a dict (if re-run) or a raw list of runs.
    new_data = {}
    for student_id, value in data.items():
        if isinstance(value, list):
            runs = value
        else:
            # Already wrapped — preserve existing keys and overwrite total_grade
            runs = value.get("runs", [])
        new_data[student_id] = {
            "total_grade": TOTAL_GRADES.get(str(student_id)),
            "runs": runs,
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4)

    print(f"JSON updated: {OUTPUT_JSON}")
    grades_written = {sid: new_data[sid]["total_grade"] for sid in new_data}
    print(f"  Grades written: {grades_written}")


if __name__ == "__main__":
    update_csv()
    update_json()
