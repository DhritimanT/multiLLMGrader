"""
Consolidate grading results from two runs:
- grading_results/: all 7 models, but gemini runs may have truncated responses
- grading_results_only_gemini/: only gemini-2.5-flash and gemini-2.5-pro, with truncation fixed

Output: consolidated_results/ with gemini entries replaced by the cleaner re-run.
"""

import json
import os
import pandas as pd

GEMINI_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro"}

ALL_RESULTS_DIR = "grading_results"
GEMINI_RESULTS_DIR = "grading_results_only_gemini"
OUTPUT_DIR = "consolidated_results"


def consolidate():
    with open(os.path.join(ALL_RESULTS_DIR, "grading_results.json")) as f:
        all_results: dict = json.load(f)

    with open(os.path.join(GEMINI_RESULTS_DIR, "grading_results.json")) as f:
        gemini_results: dict = json.load(f)

    consolidated: dict = {}

    for student_id, entries in all_results.items():
        # Keep only non-gemini entries from the original run
        non_gemini = [e for e in entries if e.get("model") not in GEMINI_MODELS]

        # Take gemini entries from the clean re-run (fall back to original if missing)
        gemini_entries = gemini_results.get(student_id, [
            e for e in entries if e.get("model") in GEMINI_MODELS
        ])

        consolidated[student_id] = non_gemini + gemini_entries

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "grading_results.json"), "w") as f:
        json.dump(consolidated, f, indent=4)

    # Build flat list for CSV
    all_rows = []
    for student_id, entries in consolidated.items():
        for entry in entries:
            all_rows.append({"student_id": student_id, **entry})

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "grading_results.csv"), index=False)

    # Summary
    models = sorted(set(r.get("model") for r in all_rows))
    students = sorted(consolidated.keys())
    print(f"Consolidated {len(students)} students, {len(models)} models into '{OUTPUT_DIR}/'")
    print(f"Models: {models}")
    print(f"Total rows in CSV: {len(df)}")


if __name__ == "__main__":
    consolidate()
