"""
Find incomplete rows in grading_results CSV.

A row is considered incomplete if:
  - overall_feedback is missing (NaN, None, or empty string)
  - Any of questions 1-6 is missing from feedback_by_question
    (nonexistent key or value is an empty string)
"""

import ast
import sys
import pandas as pd

EXPECTED_QUESTIONS = {"1", "2", "3", "4", "5", "6"}
CSV_PATH = "consolidated_results_rerun_4/grading_results.csv"
OUT_PATH = "consolidated_results_rerun_4/incomplete_rows.csv"

def check_row(row) -> list[str]:
    """Return a list of issue descriptions for the row, or empty list if complete."""
    issues = []

    # Check overall_feedback
    overall = row.get("overall_feedback")
    if pd.isna(overall) or str(overall).strip() == "":
        issues.append("missing overall_feedback")

    # Check feedback_by_question
    fbq_raw = row.get("feedback_by_question")
    if pd.isna(fbq_raw) or str(fbq_raw).strip() == "":
        issues.append("feedback_by_question is entirely missing")
        return issues

    try:
        fbq = ast.literal_eval(str(fbq_raw))
    except (ValueError, SyntaxError):
        issues.append("feedback_by_question could not be parsed")
        return issues

    if not isinstance(fbq, dict):
        issues.append("feedback_by_question is not a dict")
        return issues

    missing_qs = []
    for q in EXPECTED_QUESTIONS:
        val = fbq.get(q)
        if val is None:
            missing_qs.append(q)
        elif isinstance(val, str) and val.strip() == "":
            missing_qs.append(q)

    if missing_qs:
        issues.append(f"missing question feedback for: {sorted(missing_qs)}")

    return issues


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from '{csv_path}'\n")

    incomplete_rows = []
    for idx, row in df.iterrows():
        issues = check_row(row)
        if issues:
            incomplete_rows.append({
                "row_index": idx,
                "student_id": row.get("student_id"),
                "model": row.get("model"),
                "run": row.get("run"),
                "issues": issues,
            })

    if not incomplete_rows:
        print("Yeiiii! No incomplete rows found. 🎉")
        return

    print(f"Found {len(incomplete_rows)} incomplete row(s):\n")
    for entry in incomplete_rows:
        print(
            f"  row {entry['row_index']:>4}  student_id={entry['student_id']}  "
            f"model={entry['model']}  run={entry['run']}"
        )
        for issue in entry["issues"]:
            print(f"           -> {issue}")
    print()

    # Save to CSV
    out_df = pd.DataFrame([
        {
            "row_index": e["row_index"],
            "student_id": e["student_id"],
            "model": e["model"],
            "run": e["run"],
            "issues": "; ".join(e["issues"]),
        }
        for e in incomplete_rows
    ])
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved summary to '{OUT_PATH}'")


if __name__ == "__main__":
    main()
