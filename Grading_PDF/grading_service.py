from __future__ import annotations

import ast
import base64
import io
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from openai import OpenAI
from pylatexenc.latex2text import LatexNodes2Text

from config import logging
from storage import s3_presign_url

# Safely import Pydantic to enable Strict Structured Outputs for Gemini
try:
    from pydantic import BaseModel
    class GeminiQuestionGrade(BaseModel):
        question_id: str
        reasoning: str
        score: float
        strengths: str
        areas_for_improvement: str
        breakdown: str

    class GeminiGradingResponse(BaseModel):
        grades: List[GeminiQuestionGrade]
        overall_feedback: str
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

logger = logging.getLogger(__name__)

class LLMGrader:
    """Grades assignment submissions using LLMs with optional diagram (vision) support.

    This service expects assignment questions (including rubrics and points) and a student's
    answers. Answers can be strings or structured objects containing `text`, optional
    `diagram` (with `s3_key`), and nested `subAnswers` for multi-part questions.
    """

    @staticmethod
    def _detect_provider(model: str) -> str:
        """Detect the LLM provider from the model name."""
        if model.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        elif model.startswith(("claude-", "anthropic.", "us.", "eu.", "ap.", "global.")):
            return "anthropic"
        elif model.startswith("gemini-"):
            return "gemini"
        # Default to OpenAI for unknown models
        return "openai"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        self.model = model
        self.provider = self._detect_provider(model)

        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import boto3
            from botocore.config import Config
            region = os.getenv("AWS_S3_REGION", "us-west-1")
            config = Config(
                read_timeout=3600,  # 60 minutes (3600 seconds)
                connect_timeout=60,  # Connection timeout
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'  # Handles throttling with exponential backoff
                }
            )
            self.client = boto3.client("bedrock-runtime", region_name=region, config=config)
        elif self.provider == "gemini":
            import google.genai as _genai
            self.client = _genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider for model: {model}")

    def _call_llm(
        self,
        system_content: str,
        user_content: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_schema: Optional[Any] = None,
    ) -> str:
        """Make a single LLM call routing to the configured provider.

        Args:
            system_content: System instruction text.
            user_content: List of content parts in OpenAI message format
                          (``{"type": "text", "text": ...}`` or
                           ``{"type": "image_url", "image_url": {"url": ...}}``).
            temperature: Sampling temperature (ignored for reasoning models).
            max_tokens: Maximum tokens in the response.
            response_schema: Optional schema definition (Pydantic model) for Structured Outputs.

        Returns:
            The model's response text.
        """
        import requests as _requests

        if self.provider == "openai":
            system_msg = {"role": "system", "content": system_content}
            # Reasoning models (gpt-5, o-series) use reasoning_effort instead of temperature
            if self.model.startswith("gpt-5") or self.model.startswith(("o1", "o3", "o4")):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_msg, {"role": "user", "content": user_content}],
                    reasoning_effort="high",
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_msg, {"role": "user", "content": user_content}],
                    temperature=temperature,
                    max_tokens=16384,  # Use max tokens for OpenAI to allow for long responses
                )
            return (response.choices[0].message.content or "").strip()

        elif self.provider == "anthropic":
            # Amazon Bedrock Converse API
            bedrock_content: List[Dict[str, Any]] = []
            for part in user_content:
                if part.get("type") == "text":
                    bedrock_content.append({"text": part["text"]})
                elif part.get("type") == "pdf_document":
                    # Native PDF support via Bedrock document block
                    pdf_bytes = base64.b64decode(part["base64"])
                    bedrock_content.append({
                        "document": {
                            "format": "pdf",
                            "name": "submission",
                            "source": {"bytes": pdf_bytes},
                            "citations": {
                                "enabled": True
                            },
                        }
                    })
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        img_format = media_type.split("/")[1]  # e.g. "jpeg", "png"
                        img_bytes = base64.b64decode(data)
                    else:
                        import requests as _img_req
                        _r = _img_req.get(url, timeout=30)
                        _r.raise_for_status()
                        content_type = _r.headers.get("content-type", "image/jpeg").split(";")[0]
                        img_format = content_type.split("/")[1]
                        img_bytes = _r.content
                    bedrock_content.append({
                        "image": {
                            "format": img_format,
                            "source": {"bytes": img_bytes},
                        }
                    })
            response = self.client.converse(
                modelId=self.model,
                system=[{"text": system_content}],
                messages=[{"role": "user", "content": bedrock_content}],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            )
            # Collect all text blocks (citations mode may return multiple content blocks)
            text_parts = [
                block["text"]
                for block in response["output"]["message"]["content"]
                if "text" in block
            ]
            return " ".join(text_parts).strip()

        elif self.provider == "gemini":
            from google.genai import types as _genai_types

            parts: List[Any] = []
            for part in user_content:
                if part.get("type") == "text":
                    parts.append(_genai_types.Part(text=part["text"]))
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        image_bytes = base64.b64decode(data)
                        parts.append(_genai_types.Part.from_bytes(data=image_bytes, mime_type=media_type))
                    else:
                        resp = _requests.get(url, timeout=30)
                        resp.raise_for_status()
                        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]
                        parts.append(_genai_types.Part.from_bytes(data=resp.content, mime_type=content_type))

            # Dynamically build config to support optional schema enforcement
            config_kwargs = {
                "system_instruction": system_content,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if response_schema is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = response_schema

            response = self.client.models.generate_content(
                model=self.model,
                config=_genai_types.GenerateContentConfig(**config_kwargs),
                contents=[_genai_types.Content(role="user", parts=parts)],
            )
            return (response.text or "").strip()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def grade_submission(
        self,
        assignment: Dict[str, Any],
        submission_answers: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        telemetry_data: Optional[Dict[str, Any]] = None,
        submission_method: Optional[str] = None,
    ) -> Tuple[float, float, Dict[str, Any], str]:
        """Grade a single submission using bulk LLM call with AI plagiarism detection.

        Args:
            assignment: Assignment data with questions
            submission_answers: Student's answers
            options: Grading options
            telemetry_data: Telemetry data - supports both formats:
                - Legacy: {"pasted": bool, "pasteCount": int, ...} (submission-level)
                - New: {"per_question": {"1": {...}, "2": {...}}, "submission_level": {...}}

        Returns: total_score, total_points, feedback_by_question, overall_feedback
        """
        
        # Extract AI penalty percentage from assignment (default to 50% for backward compatibility)
        ai_penalty_percentage = assignment.get("ai_penalty_percentage")
        if ai_penalty_percentage is None:
            ai_penalty_percentage = 50.0
        penalty_multiplier = 1.0 - (ai_penalty_percentage / 100.0)

        # Extract answered subquestion IDs for optional parts filtering
        answered_subquestion_ids = self._extract_answered_subquestion_ids(
            assignment.get("questions", []), submission_answers
        )

        flattened_questions = self._flatten_questions(
            assignment.get("questions", []), "", answered_subquestion_ids
        )
        logger.info("flattened_questions" + json.dumps(flattened_questions, indent=2))

        flattened_answers = self._flatten_answers(submission_answers)
        logger.info("flattened_answers" + json.dumps(flattened_answers, indent=2))

        # Partition questions: deterministic (MCQ/TF) vs LLM-required
        deterministic_questions: List[Dict[str, Any]] = []
        llm_questions: List[Dict[str, Any]] = []
        for q in flattened_questions:
            if self._is_deterministic_question(q):
                deterministic_questions.append(q)
            else:
                llm_questions.append(q)

        feedback_by_question: Dict[str, Any] = {}
        overall_feedback = ""

        # Grade deterministic questions locally
        for question in deterministic_questions:
            q_id = str(question.get("id"))
            max_points = float(question.get("points", 0) or 0)
            answer_obj = flattened_answers.get(q_id)
            q_type = (question.get("type") or "").lower()

            # Grade the question
            if q_type == "multiple-choice":
                score, fb = self._grade_multiple_choice(
                    question, answer_obj, max_points
                )
            elif q_type == "true-false":
                score, fb = self._grade_true_false(question, answer_obj, max_points)
            else:
                score, fb = 0.0, {"breakdown": "Unsupported deterministic type"}

            feedback_by_question[q_id] = {
                "score": score,
                "max_points": max_points,
                "strengths": fb.get("strengths", ""),
                "areas_for_improvement": fb.get("areas_for_improvement", ""),
                "breakdown": fb.get("breakdown", ""),
                "ai_flag": None,  # AI detection removed
            }

        logger.info("feedback_by_question before LLM" + json.dumps(feedback_by_question, indent=2))

        # If there are LLM-required questions, build prompt only for them
        if llm_questions:
            prompt_text, diagram_s3_keys = self._build_bulk_prompt(
                llm_questions, flattened_answers
            )

            logger.info("prompt_text" + json.dumps(prompt_text, indent=2))
            # logger.info("diagram_s3_keys", diagram_s3_keys)

            # Build multimodal messages
            system_msg = {
                "role": "system",
                "content": (
                    "You are an expert academic grader. Grade strictly per rubric and points. "
                    "Always return concise, fair judgments in the exact JSON format requested."
                ),
            }

            user_content: List[Dict[str, Any]] = []
            user_content.append({"type": "text", "text": prompt_text})

            # Add all diagram images
            for s3_key in diagram_s3_keys:
                try:
                    presigned = s3_presign_url(s3_key, expires_in=3600)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": presigned},
                        }
                    )
                except Exception:
                    # If presign fails, proceed without image
                    logger.info(f"Failed to presign S3 key: {s3_key}")
                    pass

            logger.info("user_content" + json.dumps(user_content, indent=2))

            # Use Structured Outputs schema for Gemini if available
            # schema = GeminiGradingResponse if HAS_PYDANTIC and self.provider == "gemini" else None
            # Increase tokens for Gemini to accommodate Chain-of-Thought
            # max_tokens_to_use = 8000 if self.provider == "gemini" else 2000
            # For grading, we want to allow as much response as possible to get detailed feedback, so we'll use max tokens for all providers. The main constraint is the model's overall context window, which should be sufficient given our prompt and expected response size.
            max_tokens_to_use = 8192

            # Make LLM call, retrying for Gemini if response appears truncated
            _MAX_GEMINI_RETRIES = 5
            start_time = time.time()
            for attempt in range(_MAX_GEMINI_RETRIES if self.provider == "gemini" else 1):
                result_text = self._call_llm(
                    system_content=system_msg["content"],
                    user_content=user_content,
                    temperature=0.1,
                    max_tokens=max_tokens_to_use,
                    response_schema=None,
                )
                if self.provider == "gemini" and "[/OVERALL]" not in result_text and attempt < _MAX_GEMINI_RETRIES - 1:
                    logger.warning(
                        f"Gemini response appears truncated (missing [/OVERALL]) on attempt {attempt + 1}. Retrying..."
                    )
                    continue
                break
            end_time = time.time()
            time_taken = end_time - start_time

            logger.info(f"Raw LLM response: {result_text}")
            logger.info(f"LLM call duration: {time_taken:.2f} seconds")

            # Parse bulk response
            (
                llm_feedback_by_question,
                overall_feedback_llm,
            ) = self._parse_bulk_grading_response(result_text, llm_questions)

            # Run AI detection and apply penalties for LLM-graded questions
            for q_id, feedback in llm_feedback_by_question.items():
                answer_obj = flattened_answers.get(q_id)
                answer_text = self._extract_answer_text(answer_obj)
                ai_flag = None  # AI detection removed

                # Apply penalty if hard flag
                if ai_flag and ai_flag.get("flag_level") == "hard":
                    original_score = feedback.get("score", 0.0)
                    penalized_score = (
                        original_score * penalty_multiplier
                    )  # Apply configurable penalty
                    ai_flag["original_score"] = original_score
                    ai_flag["penalized_score"] = penalized_score
                    feedback["score"] = penalized_score

                # Add AI flag to feedback
                feedback["ai_flag"] = ai_flag

            # Merge results
            feedback_by_question.update(llm_feedback_by_question)
            # Keep overall feedback from LLM if present
            overall_feedback = overall_feedback_llm or overall_feedback

        logger.info("feedback_by_question after LLM" + json.dumps(feedback_by_question, indent=2))

        # Calculate totals
        total_points = sum(float(q.get("points", 0) or 0) for q in flattened_questions)
        total_score = sum(fb.get("score", 0.0) for fb in feedback_by_question.values())

        return total_score, total_points, feedback_by_question, overall_feedback, time_taken

    def grade_pdf_direct(
        self,
        assignment: Dict[str, Any],
        pdf_s3_key: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, Dict[str, Any], str]:
        """Grade a PDF submission by sending the raw PDF pages to the vision LLM.

        This POC method bypasses the pre-extracted answer JSON entirely. The LLM
        is shown every page of the student's answer sheet as an image alongside
        the full question rubric and is asked to return grading feedback in the
        same JSON format as grade_submission.

        Args:
            assignment: Assignment data with questions (same schema as grade_submission).
            pdf_s3_key: S3 object key of the original PDF answer sheet.
            options: Unused for now, reserved for future tuning options.

        Returns:
            (total_score, total_points, feedback_by_question, overall_feedback)
            — identical return type to grade_submission so callers are compatible.
        """
        import requests

        if self.provider != "anthropic":
            try:
                from pdf2image import convert_from_path
            except ImportError:
                raise RuntimeError(
                    "pdf2image is required for grade_pdf_direct. "
                    "Install it with: pip install pdf2image"
                )

        # -------------------------------------------------------------------
        # 1. Flatten questions (no optional-part filtering — LLM sees whole sheet)
        # -------------------------------------------------------------------
        flattened_questions = self._flatten_questions(
            assignment.get("questions", []), "", {}
        )
        logger.info(
            "[grade_pdf_direct] flattened_questions" + json.dumps(flattened_questions, indent=2),
        )

        total_points = sum(float(q.get("points", 0) or 0) for q in flattened_questions)

        # -------------------------------------------------------------------
        # 2. Build the rubric prompt (no student answers — visible in PDF images)
        # -------------------------------------------------------------------
        # if self.provider == "gemini":
        #     prompt_parts = [
        #         "You are an expert academic grader.\n"
        #         "The student's answer sheet is attached below as one image per page.\n"
        #         "Grade EVERY question listed below by locating the student's written answer "
        #         "in the images. FIRST write out your step-by-step reasoning, THEN score.\n\n"
        #         "Return ONLY a JSON object with this exact structure:\n"
        #         "{\n"
        #         '  "grades": [\n'
        #         '    {\n'
        #         '      "question_id": "<id>",\n'
        #         '      "reasoning": "<step-by-step logic against rubric>",\n'
        #         '      "score": <float in [0, max_points]>,\n'
        #         '      "strengths": "<brief strengths>",\n'
        #         '      "areas_for_improvement": "<areas to improve>",\n'
        #         '      "breakdown": "<detailed analysis>"\n'
        #         '    }\n'
        #         '  ],\n'
        #         '  "overall_feedback": "<overall assessment>"\n'
        #         "}\n\n"
        #         "Grade strictly according to the rubric and max points for each question.\n"
        #         "If a question is not answered in the PDF, assign 0 points.\n\n"
        #         "--- QUESTIONS, REFERENCE ANSWERS & RUBRICS ---\n",
        #     ]
        # else:
        #     prompt_parts = [
        #         "You are an expert academic grader.\n"
        #         "The student's answer sheet is attached below as one image per page.\n"
        #         "Grade EVERY question listed below by locating the student's written answer "
        #         "in the images.\n\n"
        #         "Return ONLY a JSON object with this exact structure:\n"
        #         "{\n"
        #         '  "question_<id>": {\n'
        #         '    "score": <float in [0, max_points]>,\n'
        #         '    "strengths": "<brief strengths>",\n'
        #         '    "areas_for_improvement": "<areas to improve>",\n'
        #         '    "breakdown": "<detailed analysis>"\n'
        #         "  },\n"
        #         '  "overall_feedback": "<overall assessment>"\n'
        #         "}\n\n"
        #         "Grade strictly according to the rubric and max points for each question.\n"
        #         "If a question is not answered in the PDF, assign 0 points.\n\n"
        #         "--- QUESTIONS, REFERENCE ANSWERS & RUBRICS ---\n",
        #     ]

        prompt_parts = [
            "You are an expert academic grader.\n"
            "The student's answer sheet is attached below as one image per page.\n"
            "Grade EVERY question listed below by locating the student's written answer "
            "in the images. FIRST write out your step-by-step reasoning, THEN score.\n\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "grades": [\n'
            '    {\n'
            '      "question_id": "<id (the question id e.g., 1, 2.1, 2.1.3)>",\n'
            '      "reasoning": "<step-by-step logic against rubric>",\n'
            '      "score": <float in [0, max_points]>,\n'
            '      "strengths": "<brief strengths>",\n'
            '      "areas_for_improvement": "<areas to improve>",\n'
            '      "breakdown": "<detailed analysis>"\n'
            '    }\n'
            '  ],\n'
            '  "overall_feedback": "<overall assessment>"\n'
            "}\n\n"
            "Grade strictly according to the rubric and max points for each question.\n"
            "If a question is not answered in the PDF, assign 0 points.\n\n"
            "--- QUESTIONS, REFERENCE ANSWERS & RUBRICS ---\n",
        ]


        for q in flattened_questions:
            q_id = str(q.get("id"))
            q_type = q.get("type", "text")
            equations = q.get("equations", [])
            question_text = self._sanitize_text_for_prompt(
                q.get("question", ""), equations
            )
            rubric = self._sanitize_text_for_prompt(q.get("rubric", ""), equations)
            correct_answer = self._sanitize_text_for_prompt(
                q.get("correctAnswer", q.get("correct_answer", "")), equations
            )
            max_pts = float(q.get("points", 0) or 0)

            prompt_parts.append(f"QUESTION {q_id} ({q_type}):")
            prompt_parts.append(question_text)
            if correct_answer:
                prompt_parts.append(f"REFERENCE ANSWER:\n{correct_answer}")
            if rubric:
                prompt_parts.append(f"RUBRIC:\n{rubric}")
            prompt_parts.append(f"MAX POINTS: {max_pts}")
            prompt_parts.append("")  # blank line between questions

        prompt_text = "\n".join(prompt_parts)
        logger.info("[grade_pdf_direct] prompt_text" + prompt_text)

        # -------------------------------------------------------------------
        # 3. Download PDF from S3 — native doc for Anthropic, page images for others
        # -------------------------------------------------------------------
        presigned_url = s3_presign_url(pdf_s3_key, expires_in=3600)
        resp = requests.get(presigned_url, timeout=60)
        resp.raise_for_status()
        pdf_bytes = resp.content

        if self.provider == "anthropic":
            # Send the raw PDF directly — Claude handles text + visual extraction natively
            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            logger.info(f"[grade_pdf_direct] using Anthropic native PDF support ({len(pdf_bytes)} bytes)")
            # Anthropic recommends placing the document BEFORE the text prompt
            page_parts: List[Dict[str, Any]] = [
                {"type": "pdf_document", "base64": b64_pdf}
            ]
        else:
            # Convert each page to a JPEG and send as image_url parts
            tmp_pdf_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_bytes)
                    tmp_pdf_path = tmp_pdf.name

                pages = convert_from_path(tmp_pdf_path, dpi=200)
                logger.info(f"[grade_pdf_direct] converted {len(pages)} PDF pages to images")

                page_parts = []
                for page_idx, page_img in enumerate(pages):
                    buf = io.BytesIO()
                    page_img.save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{b64}"
                    page_parts.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )
                    logger.info(f"[grade_pdf_direct] encoded page {page_idx + 1}")
            finally:
                if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                    os.remove(tmp_pdf_path)

        # -------------------------------------------------------------------
        # 4. Build and make the single multimodal LLM call
        # -------------------------------------------------------------------
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert academic grader. "
                "Grade strictly per rubric and points. "
                "Always return concise, fair judgments in the exact JSON format requested."
            ),
        }

        # For Anthropic the document block must come before the text prompt
        if self.provider == "anthropic":
            user_content: List[Dict[str, Any]] = page_parts + [{"type": "text", "text": prompt_text}]
        else:
            user_content = [{"type": "text", "text": prompt_text}] + page_parts

        logger.info(
            f"[grade_pdf_direct] making LLM call ({self.provider}, {len(page_parts)} content part(s))"
        )

        schema = GeminiGradingResponse if HAS_PYDANTIC and self.provider == "gemini" else None

        start_time = time.time()
        result_text = self._call_llm(
            system_content=system_msg["content"],
            user_content=user_content,
            temperature=0.1,
            max_tokens=20000,
            response_schema=schema,
        )
        end_time = time.time()
        time_taken = end_time - start_time
        logger.info(f"[grade_pdf_direct] LLM call duration: {time_taken:.2f} seconds")
        logger.info("[grade_pdf_direct] raw LLM response: " + json.dumps(result_text, indent=2))

        # -------------------------------------------------------------------
        # 5. Parse response and compute totals
        # -------------------------------------------------------------------
        feedback_by_question, overall_feedback = self._parse_bulk_grading_response(
            result_text, flattened_questions
        )

        total_score = sum(fb.get("score", 0.0) for fb in feedback_by_question.values())

        logger.info(
            f"[grade_pdf_direct] result: {total_score}/{total_points}" + json.dumps(feedback_by_question, indent=2)
        )

        return total_score, total_points, feedback_by_question, overall_feedback, time_taken

    def _sanitize_text_for_prompt(
        self, text: str, equations: List[dict[str, Any]] = []
    ) -> str:
        if not text:
            return ""

        # Replace LaTeX equations with text equations
        for eq in equations:
            eq_id = eq.get("id")
            latex = eq.get("latex")
            eq_text = LatexNodes2Text().latex_to_text(latex)
            eq_type = eq.get("type", "inline")
            if eq_id and latex:
                # Replace equation placeholders with LaTeX representation
                placeholder = f"<eq {eq_id}>"
                replacement = f"{eq_text}"
                logger.info(
                    f"[_sanitize_text_for_prompt] Replacing equation placeholder {placeholder} with {replacement}"
                )
                text = text.replace(placeholder, replacement)

        # Replace excessive whitespace
        sanitized = text.replace("   ", " ").replace("  ", " ").strip()
        return sanitized

    def _is_deterministic_question(self, question: Dict[str, Any]) -> bool:
        q_type = (question.get("type") or "").lower()
        return q_type in {"multiple-choice", "true-false"}

    def _parse_mcq_answer_to_index(
        self, answer: str, options: List[str]
    ) -> Optional[str]:
        """Parse MCQ answer to index string. Handles:
        - Index strings: "0", "1", "2"
        - Single letters: "A", "B", "C", "D" (most common format)
        - Letter prefixes: "A)", "B.", "a)", "b."
        - Roman numerals: "i)", "ii)", "I)", "II."
        - Numbered: "1.", "2.", "3)"
        - Full option text match
        """
        if not answer or not options:
            return None

        answer = answer.strip()

        # Direct index match
        if answer.isdigit():
            idx = int(answer)
            if 0 <= idx < len(options):
                return str(idx)

        # Single letter match (A, B, C, D, etc.) - Handle before letter prefix
        # This is the most common format in student answers
        if len(answer) == 1 and answer.isalpha():
            letter = answer.upper()
            letter_idx = ord(letter) - ord("A")
            if 0 <= letter_idx < len(options):
                return str(letter_idx)

        # Letter prefix match (A), B., a), b.)
        if len(answer) >= 2 and answer[0].isalpha() and answer[1] in ".)":
            letter = answer[0].upper()
            letter_idx = ord(letter) - ord("A")
            if 0 <= letter_idx < len(options):
                return str(letter_idx)

        # Roman numeral match (i, ii, iii, iv, etc.)
        roman_map = {
            "i": 0,
            "ii": 1,
            "iii": 2,
            "iv": 3,
            "v": 4,
            "vi": 5,
            "vii": 6,
            "viii": 7,
            "ix": 8,
            "x": 9,
            "I": 0,
            "II": 1,
            "III": 2,
            "IV": 3,
            "V": 4,
            "VI": 5,
            "VII": 6,
            "VIII": 7,
            "IX": 8,
            "X": 9,
        }
        if answer.lower() in roman_map:
            idx = roman_map[answer.lower()]
            if idx < len(options):
                return str(idx)

        # Numbered prefix match (1., 2., 3), etc.)
        if len(answer) > 1 and answer[0].isdigit() and answer[1] in ".)":
            idx = int(answer[0]) - 1  # Convert 1-based to 0-based
            if 0 <= idx < len(options):
                return str(idx)

        # Full text match
        for i, option in enumerate(options):
            if answer.lower() == option.lower():
                return str(i)

        return None

    def _normalize_mcq_correct_set(self, question: Dict[str, Any]) -> List[str]:
        """Return list of correct answers as indices (strings) when possible.
        Accepts either `correctAnswer` (single index string) or `multipleCorrectAnswers` which may
        contain option texts or indices.
        """
        options: List[str] = question.get("options") or []
        correct_set: List[str] = []
        correct_answer = question.get("correctAnswer")
        allow_multi = bool(question.get("allowMultipleCorrect"))
        multi_list: List[str] = question.get("multipleCorrectAnswers") or []

        # If allowMulti and multi_list provided, map texts to indices when needed
        if allow_multi and multi_list:
            for item in multi_list:
                item_str = str(item)
                parsed_idx = self._parse_mcq_answer_to_index(item_str, options)
                if parsed_idx is not None:
                    correct_set.append(parsed_idx)
            # De-duplicate
            correct_set = sorted(set(correct_set), key=lambda x: int(x))
        else:
            # Single-select: prefer `correctAnswer` if present, otherwise fall back to first of multi_list
            if correct_answer is not None:
                parsed_idx = self._parse_mcq_answer_to_index(
                    str(correct_answer), options
                )
                if parsed_idx is not None:
                    correct_set = [parsed_idx]
            elif multi_list:
                # Try to map first entry
                first = str(multi_list[0])
                parsed_idx = self._parse_mcq_answer_to_index(first, options)
                if parsed_idx is not None:
                    correct_set = [parsed_idx]
        return correct_set

    def _normalize_mcq_student_selection(
        self, answer_obj: Any, options: List[str]
    ) -> List[str]:
        """Return list of selected indices (strings). Accepts:
        - single string index (e.g., "1")
        - list/array of strings/ints (e.g., ["0", "2"]) for multi-select
        - Python literal list string (e.g., "['0', '1']" or "[0, 1]")
        - comma-separated string (e.g., "0,2,3")
        - text-based answers (e.g., "A)", "B.", "i)", "1.", "b)", full option text)
        """
        if answer_obj is None:
            return []
        if isinstance(answer_obj, dict):
            # Accept object answers that may carry text/diagram; in MCQ we expect selection in `text`
            if "text" in answer_obj:
                answer_obj = answer_obj.get("text")
            else:
                # No text field; nothing to grade
                return []

        # Handle list/array inputs
        if isinstance(answer_obj, list):
            indices = []
            for item in answer_obj:
                parsed_idx = self._parse_mcq_answer_to_index(str(item), options)
                if parsed_idx is not None:
                    indices.append(parsed_idx)
            return indices

        # Handle string inputs
        if isinstance(answer_obj, str):
            s = answer_obj.strip()

            # Try to parse as Python literal first (handles "['0', '1']" or "[0, 1]" format)
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed_list = ast.literal_eval(s)
                    if isinstance(parsed_list, list):
                        indices = []
                        for item in parsed_list:
                            parsed_idx = self._parse_mcq_answer_to_index(
                                str(item), options
                            )
                            if parsed_idx is not None:
                                indices.append(parsed_idx)
                        return indices
                except (ValueError, SyntaxError):
                    # Not valid Python literal, fall through to other parsing methods
                    pass

            if "," in s:
                # Comma-separated values
                indices = []
                for part in s.split(","):
                    part = part.strip()
                    if part:
                        parsed_idx = self._parse_mcq_answer_to_index(part, options)
                        if parsed_idx is not None:
                            indices.append(parsed_idx)
                return indices
            else:
                # Single value
                parsed_idx = self._parse_mcq_answer_to_index(s, options)
                return [parsed_idx] if parsed_idx is not None else []

        # Handle numeric inputs
        try:
            parsed_idx = self._parse_mcq_answer_to_index(str(int(answer_obj)), options)
            return [parsed_idx] if parsed_idx is not None else []
        except Exception:
            return []

    def _grade_multiple_choice(
        self, question: Dict[str, Any], answer_obj: Any, max_points: float
    ) -> Tuple[float, Dict[str, Any]]:
        allow_multi = bool(question.get("allowMultipleCorrect"))
        options = question.get("options") or []
        correct_set = set(self._normalize_mcq_correct_set(question))
        selected_set = set(self._normalize_mcq_student_selection(answer_obj, options))

        if not correct_set:
            return 0.0, {"breakdown": "No correct answer configured"}

        if not allow_multi:
            score = (
                max_points
                if selected_set and list(selected_set)[0] in correct_set
                else 0.0
            )
        else:
            if not selected_set:
                score = 0.0
            else:
                intersection = len(correct_set & selected_set)
                union = len(correct_set | selected_set)
                score = (intersection / union) * max_points if union > 0 else 0.0

        return score, {
            "breakdown": f"Selected={sorted(selected_set)} Correct={sorted(correct_set)}",
            "strengths": "Correct selection" if selected_set == correct_set else "",
            "areas_for_improvement": "Review correct options"
            if score < max_points
            else "",
        }

    def _grade_true_false(
        self, question: Dict[str, Any], answer_obj: Any, max_points: float
    ) -> Tuple[float, Dict[str, Any]]:
        correct_raw = question.get("correctAnswer")
        correct_val: Optional[bool] = None
        if isinstance(correct_raw, bool):
            correct_val = correct_raw
        elif isinstance(correct_raw, str):
            cr = correct_raw.strip().lower()
            if cr in {"true", "t", "1", "yes", "y"}:
                correct_val = True
            elif cr in {"false", "f", "0", "no", "n"}:
                correct_val = False

        # Normalize student answer
        student_val: Optional[bool] = None
        if isinstance(answer_obj, dict) and "text" in answer_obj:
            answer_obj = answer_obj.get("text")
        if isinstance(answer_obj, bool):
            student_val = answer_obj
        elif isinstance(answer_obj, str):
            s = answer_obj.strip().lower()
            if s in {"true", "t", "1", "yes", "y"}:
                student_val = True
            elif s in {"false", "f", "0", "no", "n"}:
                student_val = False

        if correct_val is None or student_val is None:
            return 0.0, {"breakdown": "Invalid or missing true/false answer"}

        score = max_points if correct_val == student_val else 0.0
        return score, {
            "breakdown": f"Student={student_val} Correct={correct_val}",
            "strengths": "Correct truth value" if score == max_points else "",
            "areas_for_improvement": "Review statement truth value"
            if score == 0
            else "",
        }

    def _flatten_questions(
        self,
        questions: List[Dict[str, Any]],
        parent_id: str = "",
        answered_subquestion_ids: Dict[str, List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Recursively flatten questions to handle nested multi-part questions at any depth.

        Generates composite IDs that match the PDF extraction format:
        - Top-level questions: keep original ID (e.g., "1", "17", "33")
        - Subquestions: create composite IDs like "17.1", "17.2", "33.1.1" using sequential numbering
        This matches the format from _normalize_question_number() in pdf_answer_processor.py

        For optional parts (optionalParts: true), only includes subquestions that were answered.
        """
        if answered_subquestion_ids is None:
            answered_subquestion_ids = {}

        flattened: List[Dict[str, Any]] = []
        subquestion_counter = 0  # Counter for subquestions at current level

        for q in questions:
            original_id = str(q.get("id"))

            if q.get("type") == "multi-part" and q.get("subquestions"):
                # For multi-part questions, flatten subquestions
                if parent_id:
                    # Nested multi-part: increment counter and use as part of parent ID
                    subquestion_counter += 1
                    sub_parent_id = f"{parent_id}.{subquestion_counter}"
                else:
                    # Top-level multi-part: use original ID as prefix
                    sub_parent_id = original_id

                # For optional parts, filter to only include answered subquestions
                subquestions_to_process = q.get("subquestions", [])
                if q.get("optionalParts"):
                    # Get list of answered subquestion IDs for this question
                    answered_subq_ids = answered_subquestion_ids.get(original_id, [])
                    # Filter to only include answered subquestions
                    subquestions_to_process = [
                        subq
                        for subq in subquestions_to_process
                        if str(subq.get("id")) in answered_subq_ids
                    ]

                # Recursively flatten subquestions with composite parent ID
                sub_flattened = self._flatten_questions(
                    subquestions_to_process, sub_parent_id, answered_subquestion_ids
                )
                flattened.extend(sub_flattened)
            else:
                # Regular question - create composite ID if we have a parent
                if parent_id:
                    # Use SEQUENTIAL numbering (1, 2, 3...) to match PDF extraction format
                    # This matches how 17(a) -> 17.1, 17(b) -> 17.2 in pdf_answer_processor
                    subquestion_counter += 1
                    composite_id = f"{parent_id}.{subquestion_counter}"
                    # Create a copy of the question with the composite ID
                    q_copy = q.copy()
                    q_copy["id"] = composite_id
                    q_copy["original_id"] = original_id  # Keep original for reference
                    flattened.append(q_copy)
                else:
                    # Top-level question - keep original ID
                    flattened.append(q)
        return flattened

    def _flatten_answers(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively flatten student answers to match the structure of flattened questions.

        Handles different answer structures:
        - String answers: keep as-is
        - Object answers with text/diagram: keep object structure
        - Answers with subAnswers: recursively flatten at any depth
        - Direct answers to multi-part questions: stored at parent level for fallback
        """
        flattened: Dict[str, Any] = {}

        for question_id, answer in answers.items():
            if isinstance(answer, str):
                # Simple string answer - keep as-is
                flattened[question_id] = answer
            elif isinstance(answer, dict):
                if "subAnswers" in answer:
                    # Multi-part answer - recursively flatten subAnswers
                    sub_answers = answer.get("subAnswers", {})
                    sub_flattened = self._flatten_answers(sub_answers)
                    for sub_id, sub_answer in sub_flattened.items():
                        # Create composite question ID (e.g., "17.171", "29.294.1761205736950")
                        composite_id = f"{question_id}.{sub_id}"
                        flattened[composite_id] = sub_answer

                    # Also store the parent-level answer if it has text/diagram
                    # This handles cases where student provides answer at parent level
                    if answer.get("text") or answer.get("diagram"):
                        flattened[question_id] = {
                            k: v for k, v in answer.items() if k != "subAnswers"
                        }
                else:
                    # Object answer with text/diagram but no subAnswers
                    # This might be a direct answer to a multi-part question
                    # Store it at the parent level - the grading logic will handle it
                    flattened[question_id] = answer
            else:
                # Fallback - convert to string
                flattened[question_id] = str(answer)

        return flattened

    def _extract_answered_subquestion_ids(
        self, questions: List[Dict[str, Any]], answers: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract which subquestions were answered for optional parts questions.

        Returns a dictionary mapping question IDs to lists of answered subquestion IDs.
        """
        answered_map = {}

        for q in questions:
            q_id = str(q.get("id"))
            if q.get("type") == "multi-part" and q.get("optionalParts"):
                answer_obj = answers.get(q_id)
                if answer_obj and isinstance(answer_obj, dict):
                    subanswers = answer_obj.get("subAnswers", {})
                    # Track which subquestions have non-empty answers
                    answered_subq_ids = [k for k, v in subanswers.items() if v]
                    answered_map[q_id] = answered_subq_ids

                    # Recursively handle nested optional parts
                    if q.get("subquestions"):
                        for subq in q.get("subquestions"):
                            subq_id = str(subq.get("id"))
                            if subq.get("type") == "multi-part" and subq.get(
                                "optionalParts"
                            ):
                                subq_answer = subanswers.get(subq_id)
                                if subq_answer and isinstance(subq_answer, dict):
                                    subq_subanswers = subq_answer.get("subAnswers", {})
                                    answered_subsubq_ids = [
                                        k for k, v in subq_subanswers.items() if v
                                    ]
                                    answered_map[subq_id] = answered_subsubq_ids

        return answered_map

    def _build_bulk_prompt(
        self,
        flattened_questions: List[Dict[str, Any]],
        flattened_answers: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        prompt_parts = []
        diagram_s3_keys = []
        diagram_index = 0

        if self.provider == "gemini":
            prompt_parts.append(
                "You are an expert academic grader. Grade this student's submission.\n"
                "DO NOT OUTPUT JSON. Use the highly compact delimiter format below exactly as shown to save space.\n\n"
                "For EACH question, output:\n"
                "[Q:<id>]\n"
                "[R] <strictly 1-2 short sentences of logic against the rubric>\n"
                "[S] <score as a float>\n"
                "[STR] <brief strengths>\n"
                "[AFI] <areas to improve>\n"
                "[B] <detailed breakdown>\n"
                "[/Q]\n\n"
                "At the very end, output:\n"
                "[OVERALL]\n<overall assessment>\n[/OVERALL]\n\n"
                "CRITICAL: Keep all explanations extremely concise (under 30 words per field) to avoid output cutoff.\n"
                "Questions, reference answers, rubrics, max points, and student answers follow:\n"
            )
        else:
            # Keep your existing OpenAI/Anthropic JSON prompt here...
            prompt_parts.append(
                "You are an expert academic grader. Grade this student's submission for all questions. "
                "For each question, provide a score, strengths, areas for improvement, and detailed breakdown. "
                "Return your response as JSON with the following structure:\n"
                "{\n"
                '  "question_<id>": {\n'
                '    "score": <float in [0, max_points]>,\n'
                '    "strengths": "<brief strengths>",\n'
                '    "areas_for_improvement": "<areas to improve>",\n'
                '    "breakdown": "<detailed analysis>"\n'
                "  },\n"
                '  "overall_feedback": "<overall assessment>"\n'
                "}\n\n"
                "GRADING CRITERIA:\n"
                "- Grade strictly according to the provided rubric and max points.\n\n"
                "Questions, reference answers, rubrics, max points, and student answers follow:\n"
            )

        for question in flattened_questions:
            q_id = str(question.get("id"))
            q_type = question.get("type", "text")
            equations = question.get("equations", [])
            question_text = self._sanitize_text_for_prompt(
                question.get("question", ""), equations
            )
            rubric = self._sanitize_text_for_prompt(
                question.get("rubric", ""), equations
            )
            correct_answer = self._sanitize_text_for_prompt(
                question.get("correctAnswer", question.get("correct_answer", "")),
                equations,
            )
            max_points = float(question.get("points", 0) or 0)

            # Add question details
            prompt_parts.append(f"QUESTION {q_id} ({q_type}):")
            prompt_parts.append(f"{question_text}")

            if correct_answer:
                prompt_parts.append(f"REFERENCE ANSWER:\n{correct_answer}")

            if rubric:
                prompt_parts.append(f"RUBRIC:\n{rubric}")

            prompt_parts.append(f"MAX POINTS: {max_points}")

            # Add student answer
            answer_obj = flattened_answers.get(q_id)
            if answer_obj is not None:
                if isinstance(answer_obj, str):
                    prompt_parts.append(f"STUDENT ANSWER:\n{answer_obj}")
                elif isinstance(answer_obj, dict):
                    text_answer = answer_obj.get("text", "")
                    if text_answer:
                        prompt_parts.append(f"STUDENT ANSWER (text):\n{text_answer}")

                    # Check for diagram
                    diagram = answer_obj.get("diagram")
                    if diagram and isinstance(diagram, dict):
                        s3_key = diagram.get("s3_key")
                        if s3_key:
                            diagram_index = diagram_index + 1
                            ordinal_suffix = (
                                "st"
                                if diagram_index == 1
                                else "nd"
                                if diagram_index == 2
                                else "rd"
                                if diagram_index == 3
                                else "th"
                            )
                            diagram_s3_keys.append(s3_key)
                            prompt_parts.append(
                                f"STUDENT ANSWER (diagram): [Image attached - see {diagram_index}{ordinal_suffix} image]"
                            )
            else:
                prompt_parts.append("STUDENT ANSWER: <no answer provided>")

            prompt_parts.append("")  # Empty line between questions

        return "\n".join(prompt_parts), diagram_s3_keys

    def _parse_bulk_grading_response(
        self, response_text: str, flattened_questions: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """Parse JSON response from bulk grading LLM call.

        Returns:
            Tuple of (feedback_by_question, overall_feedback)
        """
        import json

        feedback_by_question = {}
        overall_feedback = ""

        try:
            # If it's not JSON (our new Gemini format), parse using Regex
            if self.provider == "gemini":
                import re
                feedback_by_question = {}
                
                # Extract Overall Feedback
                overall_match = re.search(r"\[OVERALL\](.*?)\[/OVERALL\]", response_text, re.DOTALL | re.IGNORECASE)
                overall_feedback = overall_match.group(1).strip() if overall_match else ""

                # Extract Each Question Block
                q_blocks = re.finditer(r'\[Q:(.*?)\](.*?)\[/Q\]', response_text, re.DOTALL | re.IGNORECASE)
                
                for block in q_blocks:
                    q_id = block.group(1).strip()
                    content = block.group(2)
                    
                    # Look up max points
                    max_points = 0.0
                    for q in flattened_questions:
                        if str(q.get("id")) == q_id:
                            max_points = float(q.get("points", 0) or 0)
                            break
                            
                    # Helper to grab text from a tag until the next tag bracket '[' or end of string
                    def extract_field(field, text):
                        match = re.search(rf"\[{field}\](.*?)(?=\[|$)", text, re.DOTALL | re.IGNORECASE)
                        return match.group(1).strip() if match else ""

                    try:
                        raw_score = extract_field("S", content)
                        score = float(raw_score) if raw_score else 0.0
                    except ValueError:
                        score = 0.0
                        
                    score = max(0.0, min(score, max_points))
                    reasoning = extract_field("R", content)
                    raw_breakdown = extract_field("B", content)
                    
                    feedback_by_question[q_id] = {
                        "score": score,
                        "max_points": max_points,
                        "strengths": extract_field("STR", content),
                        "areas_for_improvement": extract_field("AFI", content),
                        "breakdown": f"Reasoning: {reasoning}\n\n{raw_breakdown}" if reasoning else raw_breakdown
                    }

                return feedback_by_question, overall_feedback
            
            import re as _re
            # Strip markdown code fences and leading/trailing prose before parsing
            clean_text = response_text
            md_match = _re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_text)
            if md_match:
                clean_text = md_match.group(1)
            else:
                # Fall back to extracting the outermost JSON object
                first_brace = clean_text.find('{')
                last_brace = clean_text.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    clean_text = clean_text[first_brace:last_brace + 1]

            # Try to parse as JSON
            response_data = json.loads(clean_text)

            # Handle structured Gemini 'grades' array safely
            if "grades" in response_data and isinstance(response_data["grades"], list):
                overall_feedback = response_data.get("overall_feedback", "")
                
                for grade_item in response_data["grades"]:
                    q_id = str(grade_item.get("question_id"))
                    
                    # Look up max points dynamically to clamp safely
                    max_points = 0.0
                    for q in flattened_questions:
                        if str(q.get("id")) == q_id:
                            max_points = float(q.get("points", 0) or 0)
                            break

                    score = float(grade_item.get("score", 0.0))
                    score = max(0.0, min(score, max_points))

                    reasoning = grade_item.get("reasoning", "")
                    raw_breakdown = grade_item.get("breakdown", "")
                    # Prepend reasoning logic if present for visibility
                    breakdown = f"Reasoning: {reasoning}\n\n{raw_breakdown}" if reasoning else raw_breakdown

                    feedback_by_question[q_id] = {
                        "score": score,
                        "max_points": max_points,
                        "strengths": grade_item.get("strengths", ""),
                        "areas_for_improvement": grade_item.get("areas_for_improvement", ""),
                        "breakdown": breakdown,
                    }
                return feedback_by_question, overall_feedback

            # Existing generic parsing block for OpenAI and Anthropic
            overall_feedback = response_data.get("overall_feedback", "")

            for question in flattened_questions:
                q_id = str(question.get("id"))
                max_points = float(question.get("points", 0) or 0)

                question_key = f"question_{q_id}"
                question_data = response_data.get(question_key, {})

                # Extract score and feedback
                score = float(question_data.get("score", 0.0))
                score = max(0.0, min(score, max_points))  # Clamp to valid range

                feedback_by_question[q_id] = {
                    "score": score,
                    "max_points": max_points,
                    "strengths": question_data.get("strengths", ""),
                    "areas_for_improvement": question_data.get(
                        "areas_for_improvement", ""
                    ),
                    "breakdown": question_data.get("breakdown", ""),
                }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}. Attempting regex extraction.")
            
            # Fallback: try to extract information using regex patterns
            import re

            # Extract overall feedback
            overall_match = re.search(r'"overall_feedback":\s*"([^"]*)"', response_text)
            if overall_match:
                overall_feedback = overall_match.group(1)

            # Extract per-question scores and feedback
            for question in flattened_questions:
                q_id = str(question.get("id"))
                max_points = float(question.get("points", 0) or 0)

                # Try to find score for this question
                score_pattern = rf'"question_{q_id}":\s*{{[^}}]*"score":\s*([0-9.]+)'
                score_match = re.search(score_pattern, response_text)
                score = float(score_match.group(1)) if score_match else 0.0
                score = max(0.0, min(score, max_points))

                # Extract other fields
                strengths = ""
                areas = ""
                breakdown = ""

                strengths_match = re.search(
                    rf'"question_{q_id}":\s*{{[^}}]*"strengths":\s*"([^"]*)"',
                    response_text,
                )
                if strengths_match:
                    strengths = strengths_match.group(1)

                areas_match = re.search(
                    rf'"question_{q_id}":\s*{{[^}}]*"areas_for_improvement":\s*"([^"]*)"',
                    response_text,
                )
                if areas_match:
                    areas = areas_match.group(1)

                breakdown_match = re.search(
                    rf'"question_{q_id}":\s*{{[^}}]*"breakdown":\s*"([^"]*)"',
                    response_text,
                )
                if breakdown_match:
                    breakdown = breakdown_match.group(1)

                feedback_by_question[q_id] = {
                    "score": score,
                    "max_points": max_points,
                    "strengths": strengths,
                    "areas_for_improvement": areas,
                    "breakdown": breakdown,
                }
        
        except Exception as e:
            logger.error(f"Unexpected error while parsing LLM response: {e}")

        return feedback_by_question, overall_feedback

    def _extract_answer_text(self, answer_obj: Any) -> str:
        """Extract plain text from answer object (handles string or dict with 'text' field)."""
        if answer_obj is None:
            return ""
        if isinstance(answer_obj, str):
            return answer_obj
        if isinstance(answer_obj, dict):
            # Handle structured answer with 'text' field
            if "text" in answer_obj:
                return str(answer_obj["text"])
            # Handle subAnswers (multi-part questions)
            if "subAnswers" in answer_obj:
                sub_texts = []
                for sub_key, sub_val in answer_obj["subAnswers"].items():
                    sub_texts.append(self._extract_answer_text(sub_val))
                return " ".join(sub_texts)
        return str(answer_obj)

    def _extract_question_telemetry(
        self, question_id: str, telemetry_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract telemetry for a specific question, with fallback to submission-level.

        Args:
            question_id: Flattened question ID (e.g., "1", "17.2", "17.2.1")
            telemetry_data: Telemetry structure

        Returns:
            Question-specific telemetry dict, or submission-level telemetry, or None
        """
        if not telemetry_data:
            return None

        # Check if new format with per_question key
        if "per_question" in telemetry_data:
            per_question = telemetry_data.get("per_question", {})
            # Return telemetry for this specific question if available
            if question_id in per_question:
                return per_question[question_id]
            # No telemetry for this question (possibly unanswered optional part)
            return None

        # Legacy format: use submission-level telemetry for all questions
        # Check if this looks like submission-level telemetry (has expected keys)
        if any(
            key in telemetry_data
            for key in ["pasted", "pasteCount", "tabSwitches", "timeToComplete"]
        ):
            return telemetry_data

        return None
