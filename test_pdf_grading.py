"""
Test hand-written pdf submission grading with different models of various providers.
Providers: OpenAI, Anthropic, Google Gemini
Models tested:
- OpenAI: gpt-4o, gpt-5
- Anthropic: claude-3-7-sonnet-20250219, claude-3-5-haiku-20241022
- Google Gemini: gemini-2.0-flash, gemini-1.5-pro
Output: Save results for all models in a structured format (e.g. CSV and JSON) for easy comparison and analysis.
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

def test_pdf_grading():
    # max_runs = 3 # Run each model 3 times for consistency
    max_runs = 3
    # Load test PDF submissions and assignment details
    test_data_dir = "test_files"
    pdf_submissions = {
        "1": "assignments/97182b65-8eb9-4e93-9332-308104003723/uploads/2217f0ba-0dde-4fcb-bfcd-b4f0fdba9582.pdf",
        "2": "assignments/97182b65-8eb9-4e93-9332-308104003723/uploads/f0e00cb8-2068-4639-9ab0-5ab27e9af7ed.pdf",
        "3": "assignments/97182b65-8eb9-4e93-9332-308104003723/uploads/95723a95-ce20-46ee-803b-414191322765.pdf",
    }
    with open(os.path.join(test_data_dir, "AssignmentRahul.json"), "r") as f:
        assignment_dict = json.load(f)

    # Define models to test
    models_to_test = [
        "gpt-5",
        "gpt-4o",
        "global.anthropic.claude-opus-4-6-v1",
        "global.anthropic.claude-sonnet-4-6",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    # Store results
    results: Dict[str, List[Dict]] = {pdf_id: [] for pdf_id in pdf_submissions.keys()}

    for model in models_to_test:
        try:
            grader = LLMGrader(model=model)
            for pdf_id, pdf_s3_key in pdf_submissions.items():
                for run in range(1,max_runs+1):
                    start_time = time.time()
                    (
                        total_score,
                        total_points,
                        feedback_by_question,
                        overall_feedback,
                        llm_call_time_taken,
                    ) = grader.grade_pdf_direct(
                        assignment=assignment_dict,
                        pdf_s3_key=pdf_s3_key,
                        options={},
                    )
                    end_time = time.time()
                    
                    total_time_taken = end_time - start_time

                    results[pdf_id].append({
                        "model": model,
                        "run": run,
                        "total_score": total_score,
                        "total_points": total_points,
                        "feedback_by_question": feedback_by_question,
                        "overall_feedback": overall_feedback,
                        "llm_call_time_taken": llm_call_time_taken,
                        "total_time_taken": total_time_taken,
                    })
        except Exception as e:
            logger.info(f"Error grading with model {model}: {str(e)}")
            results[pdf_id].append({
                "model": model,
                "error": str(e),
            })

    out_dir = "grading_results"
    os.makedirs(out_dir, exist_ok=True)

    # Save results to JSON and CSV for analysis
    with open(os.path.join(out_dir, "grading_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    import pandas as pd
    all_results = []
    for pdf_id, pdf_results in results.items():
        for res in pdf_results:
            all_results.append({
                "pdf_id": pdf_id,
                **res,
            })
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "grading_results.csv"), index=False)

if __name__ == "__main__":
    test_pdf_grading()
