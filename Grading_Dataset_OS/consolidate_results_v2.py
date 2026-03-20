"""
Consolidate grading results from main run and re-run:
- ALL_RESULTS_DIR: all 7 models, but some runs (say gemini-2.5-flash and global.anthropic.claude-haiku-4-5-20251001-v1:0 for certain students) may have truncated responses
- FLASH_RESULTS_DIR: only incomplete runs (say gemini-2.5-flash and global.anthropic.claude-haiku-4-5-20251001-v1:0 for certain students), with truncation fixed

Output: OUTPUT_DIR with entries replaced by the cleaner re-run.
"""

import json
import os
import pandas as pd

FLASH_MODELS = {"gemini-2.5-flash"}

ALL_RESULTS_DIR = "consolidated_results_rerun_3"
FLASH_RESULTS_DIR = "grading_results_rerun_4"
OUTPUT_DIR = "consolidated_results_rerun_4"


def consolidate():
    with open(os.path.join(ALL_RESULTS_DIR, "grading_results.json")) as f:
        all_results: dict = json.load(f)

    with open(os.path.join(FLASH_RESULTS_DIR, "grading_results.json")) as f:
        flash_results: dict = json.load(f)

    consolidated: dict = {}

    for student_id, entries in all_results.items():
        non_flash_entries = [e for e in entries if e.get("model") not in FLASH_MODELS]

        rerun_entries = flash_results.get(student_id, [])
        rerun_models = {e.get("model") for e in rerun_entries}
        # For each flash model: use rerun entries if that model was re-run, else keep originals
        flash_entries = rerun_entries + [
            e for e in entries
            if e.get("model") in FLASH_MODELS and e.get("model") not in rerun_models
        ]

        consolidated[student_id] = non_flash_entries + flash_entries

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
