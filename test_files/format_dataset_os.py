"""
Generates:
  1. assignment.json  – one assignment with 6 short-answer questions built from
                        tutorialCriteria/ + grading JSONs.
  2. student_answers/<student_id>.json – one answer file per student (40 total),
                        format:  { "<question_id>": "<answer_text>", ... }

Both are written inside the dataset_os/ directory next to the source data.
"""

import json
import os
import uuid
from datetime import datetime, timezone

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, "dataset_os")
CRITERIA_DIR = os.path.join(DATASET_DIR, "tutorialCriteria")
OUT_ASSIGNMENT = os.path.join(SCRIPT_DIR, "assignment.json")
OUT_ANSWERS_DIR = os.path.join(SCRIPT_DIR, "student_answers")

NUM_QUESTIONS = 6   # q1 … q6
# ─────────────────────────────────────────────────────────────────────────────


def load_criteria(q_num: int) -> dict:
    path = os.path.join(CRITERIA_DIR, f"q{q_num}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # key inside criteria file equals the sub-question number (str)
    return data[str(q_num)]


def load_grading(q_num: int) -> dict:
    path = os.path.join(DATASET_DIR, f"q{q_num}", "grading.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_question_text(grading: dict, q_num: int) -> str:
    """Return question text (same for all students – grab from first student)."""
    first_student = next(iter(grading))
    item = grading[first_student][str(q_num)]
    q = item["question"]
    # question may be a list or a plain string
    if isinstance(q, list):
        return " ".join(q)
    return q


# ── 1. Build assignment JSON ──────────────────────────────────────────────────

def build_question(q_num: int) -> dict:
    criteria = load_criteria(q_num)
    grading  = load_grading(q_num)

    return {
        "id":          q_num,
        "order":       q_num,
        "type":        "short-answer",
        "points":      criteria["full_points"],
        "question":    get_question_text(grading, q_num),
        "correctAnswer": criteria.get("answer", ""),
        "rubric":      criteria.get("criteria", ""),
        "rubricType":  "overall",
        # static / boilerplate keys
        "code":        "",
        "hasCode":     False,
        "options":     [],
        "equations":   [],
        "hasDiagram":  False,
        "outputType":  "",
        "codeLanguage": "",
        "optionalParts":       False,
        "requiredPartsCount":  0,
        "allowMultipleCorrect": False,
        "multipleCorrectAnswers": [],
    }


def build_assignment() -> dict:
    questions   = [build_question(q) for q in range(1, NUM_QUESTIONS + 1)]
    total_pts   = sum(q["points"] for q in questions)
    now         = datetime.now(timezone.utc).isoformat()

    return {
        "id":                   str(uuid.uuid4()),
        "title":                "Operating Systems – Tutorial Assignment",
        "description":          "Tutorial questions covering scheduling, concurrency, and memory management.",
        "total_points":         str(total_pts),
        "total_questions":      str(NUM_QUESTIONS),
        "status":               "published",
        "question_types":       ["short-answer"],
        "engineering_level":    "undergraduate",
        "engineering_discipline": "computer science",
        "due_date":             None,
        "google_form_url":      None,
        "google_form_response_url": None,
        "shared_count":         "0",
        "created_at":           now,
        "updated_at":           now,
        "questions":            questions,
    }


# ── 2. Build per-student answer JSONs ─────────────────────────────────────────

def collect_all_grading() -> dict[int, dict]:
    """Return {q_num: grading_dict} for all questions."""
    return {q: load_grading(q) for q in range(1, NUM_QUESTIONS + 1)}


def build_student_answers(all_grading: dict[int, dict]) -> dict[str, dict[str, str]]:
    """Return {student_id: {question_id: answer_text}}."""
    # Collect all student IDs (should be 1-40)
    student_ids = sorted(next(iter(all_grading.values())).keys(), key=int)

    result: dict[str, dict[str, str]] = {}
    for sid in student_ids:
        answers: dict[str, str] = {}
        for q_num, grading in all_grading.items():
            item = grading.get(sid, {}).get(str(q_num))
            if item is not None:
                answers[str(q_num)] = item.get("answer", "")
        result[sid] = answers
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_ANSWERS_DIR, exist_ok=True)

    # --- Assignment ----------------------------------------------------------
    assignment = build_assignment()
    with open(OUT_ASSIGNMENT, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=4, ensure_ascii=False)
    print(f"Assignment written → {OUT_ASSIGNMENT}")

    # --- Student answers ------------------------------------------------------
    all_grading = collect_all_grading()
    student_answers = build_student_answers(all_grading)

    for sid, answers in student_answers.items():
        out_path = os.path.join(OUT_ANSWERS_DIR, f"{sid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)

    print(f"Student answer files written → {OUT_ANSWERS_DIR}/")
    print(f"  {len(student_answers)} files created for students: "
          f"{sorted(student_answers.keys(), key=int)[:5]} … "
          f"{sorted(student_answers.keys(), key=int)[-1]}")


if __name__ == "__main__":
    main()
