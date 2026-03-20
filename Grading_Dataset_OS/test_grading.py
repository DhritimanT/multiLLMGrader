"""
Test student submission grading with different models of various providers.
Providers: OpenAI, Anthropic, Google Gemini
Models tested:
- OpenAI: gpt-4o, gpt-5
- Anthropic: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5
- Google Gemini: gemini-2.5-flash, gemini-2.5-pro
Output: Save results for all models in a structured format (CSV and JSON) for easy comparison and analysis.
"""
import dotenv

dotenv.load_dotenv()

import json
import os
import time
from typing import Dict, List
from config import logging

from grading_service import LLMGrader

logger = logging.getLogger(__name__)


def test_grading():
    max_runs = 3
    test_data_dir = "test_files"
    student_answers_dir = os.path.join(test_data_dir, "student_answers")

    # Load all student answer files
    student_submissions: Dict[str, Dict] = {}
    for filename in sorted(os.listdir(student_answers_dir)):
        if filename.endswith(".json"):
            student_id = filename.replace(".json", "")
            with open(os.path.join(student_answers_dir, filename), "r") as f:
                student_submissions[student_id] = json.load(f)

    with open(os.path.join(test_data_dir, "assignment.json"), "r") as f:
        assignment_dict = json.load(f)

    models_to_test = [
        "gpt-5",
        "gpt-4o",
        "global.anthropic.claude-opus-4-6-v1",
        "global.anthropic.claude-sonnet-4-6",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    # Store results keyed by student_id
    results: Dict[str, List[Dict]] = {sid: [] for sid in student_submissions.keys()}

    for model in models_to_test:
        try:
            for student_id, submission_answers in student_submissions.items():
                for run in range(1, max_runs + 1):
                    grader = LLMGrader(model=model)
                    start_time = time.time()
                    (
                        total_score,
                        total_points,
                        feedback_by_question,
                        overall_feedback,
                        time_taken,
                    ) = grader.grade_submission(
                        assignment=assignment_dict,
                        submission_answers=submission_answers,
                        options={},
                    )
                    end_time = time.time()
                    total_time_taken = end_time - start_time
                    
                    logger.info(f"Graded student {student_id} with model {model} in run {run}: score {total_score}/{total_points}, time taken {time_taken:.2f}s, total time {total_time_taken:.2f}s")
                    logger.info(f"Feedback by question: {json.dumps(feedback_by_question, indent=2)}")

                    results[student_id].append({
                        "model": model,
                        "run": run,
                        "total_score": total_score,
                        "total_points": total_points,
                        "feedback_by_question": feedback_by_question,
                        "overall_feedback": overall_feedback,
                        "llm_call_time_taken": time_taken,
                        "total_time_taken": total_time_taken,
                    })
        except Exception as e:
            logger.info(f"Error grading with model {model}: {str(e)}")
            for student_id in student_submissions.keys():
                results[student_id].append({
                    "model": model,
                    "error": str(e),
                })

    out_dir = "grading_results"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "grading_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    import pandas as pd
    all_results = []
    for student_id, student_results in results.items():
        for res in student_results:
            all_results.append({
                "student_id": student_id,
                **res,
            })
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "grading_results.csv"), index=False)


if __name__ == "__main__":
    test_grading()
