"""
Adds human (TA) grades from dataset_os grading.json files into a copy of
consolidated_results/grading_results.csv.

New columns added:
  ta1_total_score   – sum of score_1 across all questions for the student
  ta2_total_score   – sum of score_2 across all questions
  ta3_total_score   – sum of score_3 across all questions
  ta_grades_by_question – JSON with per-question scores for each TA

Usage:
  python3 add_ta_grades.py
Output:
  consolidated_results/grading_results_with_ta.csv
"""

import csv
import json
import os

DATASET_DIR = "test_files/dataset_os"
INPUT_CSV   = "consolidated_results_rerun_4/grading_results.csv"
OUTPUT_CSV  = "consolidated_results_rerun_4/grading_results_with_ta.csv"
QUESTIONS   = [1, 2, 3, 4, 5, 6]
NUM_TAS     = 3


def load_ta_grades():
    """
    Returns a dict keyed by student_id (str).
    Each value is another dict keyed by question number (str):
      { "score_1": ..., "score_2": ..., "score_3": ..., "full_points": ... }
    """
    ta_grades = {}   # { student_id: { question: { score_1, score_2, score_3, full_points } } }

    for q in QUESTIONS:
        path = os.path.join(DATASET_DIR, f"q{q}", "grading.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for student_id, qdata in data.items():
            entry = qdata[str(q)]
            if student_id not in ta_grades:
                ta_grades[student_id] = {}

            ta_grades[student_id][str(q)] = {
                "full_points": entry.get("full_points"),
                "score_1":     entry.get("score_1"),
                "score_2":     entry.get("score_2"),
                "score_3":     entry.get("score_3"),
            }

    return ta_grades


def compute_ta_totals(student_grades):
    """
    Given the per-question dict for one student, sum up each TA's scores.
    Returns (ta1_total, ta2_total, ta3_total).
    Missing scores (None) are treated as 0.
    """
    totals = [0.0, 0.0, 0.0]
    for qdata in student_grades.values():
        for i, key in enumerate(["score_1", "score_2", "score_3"]):
            val = qdata.get(key)
            if val is not None:
                totals[i] += float(val)
    return totals[0], totals[1], totals[2]


def main():
    ta_grades = load_ta_grades()

    with open(INPUT_CSV, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        original_fieldnames = reader.fieldnames or []
        rows = list(reader)

    new_fieldnames = original_fieldnames + [
        "ta1_total_score",
        "ta2_total_score",
        "ta3_total_score",
        "ta_grades_by_question",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
        writer.writeheader()

        for row in rows:
            student_id = str(row["student_id"])

            if student_id in ta_grades:
                student_grades = ta_grades[student_id]
                ta1, ta2, ta3 = compute_ta_totals(student_grades)
                ta_json = json.dumps(student_grades)
            else:
                ta1 = ta2 = ta3 = None
                ta_json = json.dumps({})

            row["ta1_total_score"]      = ta1
            row["ta2_total_score"]      = ta2
            row["ta3_total_score"]      = ta3
            row["ta_grades_by_question"] = ta_json

            writer.writerow(row)

    print(f"Written: {OUTPUT_CSV}")
    print(f"Rows processed: {len(rows)}")


if __name__ == "__main__":
    main()
