"""
Enhanced LLM Judge Evaluator for Reranking Research

This module provides structured evaluation using LLM-as-a-judge with
True/False correctness assessment, confidence scoring, and detailed reasoning.
"""

import asyncio
import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from rich.console import Console
from retry import retry
from openai import RateLimitError
from ..models.interface import ModelInterface
from loguru import logger
import re

console = Console()


def extract_retry_delay(error_message: str) -> float:
    """Extract retry delay from rate limit error message.

    Args:
        error_message: Error message from API response

    Returns:
        Delay in seconds, or 0 if not found
    """
    # Look for patterns like "Try again after X seconds" or "retry after X seconds"
    patterns = [
        r"Try again after (\d+) seconds",
        r"retry after (\d+) seconds",
        r"Retry-After:\s*(\d+)",
        r"Please retry after (\d+) seconds",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            try:
                delay = float(match.group(1))
                # Cap the delay to reasonable maximum (5 minutes)
                return min(delay, 300)
            except (ValueError, IndexError):
                continue

    return 0.0


@dataclass
class JudgeEvaluation:
    """Result from LLM judge evaluation."""

    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    confidence: float
    reasoning: str
    answer_completeness: float
    context_utilization: float
    approach_used: str
    judge_model: str
    evaluation_time_ms: float
    raw_judge_response: str
    metadata: Dict[str, Any]
    # New dual correctness fields
    is_correct_given_context: bool = False
    context_grounded_confidence: float = 0.0
    context_grounded_reasoning: str = ""
    retrieval_gap: float = 0.0


from pydantic import BaseModel, Field
from typing import Union


class AbsoluteAssessment(BaseModel):
    is_correct: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation comparing to ground truth")


class ContextGroundedAssessment(BaseModel):
    is_correct_given_context: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation based only on provided context")


class RAGEvaluation(BaseModel):
    model_config = {"extra": "forbid"}
    absolute_assessment: AbsoluteAssessment
    context_grounded_assessment: ContextGroundedAssessment
    answer_completeness: float = Field(ge=0.0, le=1.0)
    context_utilization: float = Field(ge=0.0, le=1.0)


class JudgeEvaluator:
    """Enhanced LLM judge with structured True/False evaluation."""

    def __init__(self, model_interface: ModelInterface, judge_model: str | None = None, quiet_mode: bool = False):
        self.model_interface = model_interface
        self.judge_model = judge_model or "gpt-4.1-2025-04-14"
        self.quiet_mode = quiet_mode

        # Use standard prompt template for all models
        self.evaluation_prompt_template = """You are an expert evaluator assessing the correctness of answers to questions. You will perform TWO separate evaluations:

1. ABSOLUTE CORRECTNESS: Compare the model's answer to the expected ground truth answer
2. CONTEXT-GROUNDED CORRECTNESS: Evaluate if the model's answer is correct given only the provided context

Question: {question}

Expected Answer (Ground Truth): {expected_answer}

Model's Answer: {model_answer}

Context Used: {context_preview}

Please evaluate the model's answer and respond with a JSON object containing:
{{
    "absolute_assessment": {{
        "is_correct": true/false,
        "confidence": 0.0-1.0,
        "reasoning": "Detailed explanation comparing to ground truth"
    }},
    "context_grounded_assessment": {{
        "is_correct_given_context": true/false,
        "confidence": 0.0-1.0,
        "reasoning": "Detailed explanation based only on provided context"
    }},
    "answer_completeness": 0.0-1.0,
    "context_utilization": 0.0-1.0
}}

Guidelines:
- absolute_assessment.is_correct: true if the model's answer matches the expected ground truth answer
- context_grounded_assessment.is_correct_given_context: true if the model's answer is reasonable/correct based solely on the provided context, regardless of ground truth
- confidence: your confidence in each assessment (0.0 = no confidence, 1.0 = completely confident)
- reasoning: explain your evaluation decision for each assessment
- answer_completeness: how complete the model's answer is overall (0.0 = incomplete, 1.0 = comprehensive)
- context_utilization: how well the model used available context (0.0 = poor use, 1.0 = excellent use)

Key distinction: A model might give an incorrect answer compared to ground truth (absolute_assessment.is_correct = false) but still be reasonable given limited context (context_grounded_assessment.is_correct_given_context = true).

Be strict but fair in your evaluation. Minor phrasing differences are acceptable if the core information is correct.

Respond only with the JSON object:"""

        console.print(f"[green]JudgeEvaluator initialized with model: {self.judge_model}[/green]")

    async def evaluate_answer(
        self,
        question: str,
        expected_answer: str,
        model_answer: str,
        context_preview: str,
        max_retries: int = 3,
        **kwargs,
    ) -> RAGEvaluation | bool:
        """
        Evaluate a model's answer using LLM judge.

        Args:
            question: Original question
            expected_answer: Ground truth answer
            model_answer: Model's response to evaluate
            context_preview: Preview of context used (first 500 chars)
            approach_used: Which approach generated this answer
            **kwargs: Additional parameters for model interface

        Returns:
            JudgeEvaluation with structured assessment
        """

        # Prepare evaluation prompt
        prompt = self.evaluation_prompt_template.format(
            question=question,
            expected_answer=expected_answer,
            model_answer=model_answer,
            context_preview=context_preview,
        )
        for i in range(max_retries):
            try:
                completion = await self.model_interface.async_client.chat.completions.parse(
                    messages=[{"role": "user", "content": prompt}],
                    response_format=RAGEvaluation,
                    model=self.model_interface.get_model_config(self.judge_model).api_name,
                    temperature=0.0,
                )
                message = completion.choices[0].message
                if message.parsed:
                    return message.parsed
            except RateLimitError as e:  # noqa: PERF203
                if i == 0:
                    try:
                        logger.info("Trying call with OpenAI")
                        completion = await self.model_interface.async_client.chat.completions.parse(
                            messages=[{"role": "user", "content": prompt}],
                            response_format=RAGEvaluation,
                            model=self.model_interface.get_model_config("gpt-4.1-mini-oa").api_name,
                            temperature=0.0,
                        )
                        message = completion.choices[0].message
                        if message.parsed:
                            return message.parsed
                    except Exception as sube:
                        logger.warning(f"gpt-4.1-mini-oa call failed with {sube}")
                        await asyncio.sleep(60)
                sleep_duration = extract_retry_delay(error_message=str(e)) + 1
                logger.error(
                    f"Judge evaluation attempt {i + 1} failed. Sleeping for {sleep_duration} seconds: {type(e)} - {e}"
                )
                await asyncio.sleep(sleep_duration + random.randint(1, 10))  # Exponential backoff
                continue
            except Exception as e:
                if "400" in str(e):
                    return False
                sleep_duration = (5**i) + 1
                logger.error(
                    f"Judge evaluation attempt {i + 1} failed. Sleeping for {sleep_duration} seconds: {type(e)} - {e}"
                )
                await asyncio.sleep(sleep_duration + random.randint(1, 10))  # Exponential backoff
                continue
        else:
            raise ValueError("Failed to get succesful judge evaluation after retries")

    def batch_evaluate_answers(self, evaluations: list[dict[str, Any]], **kwargs) -> list[JudgeEvaluation]:
        """
        Evaluate multiple answers in batch.

        Args:
            evaluations: List of evaluation dictionaries with keys:
                        question, expected_answer, model_answer, context_preview, approach_used
            **kwargs: Additional parameters for evaluation

        Returns:
            List of JudgeEvaluation objects
        """
        console.print(f"[blue]Batch evaluating {len(evaluations)} answers[/blue]")

        results = []
        for i, eval_data in enumerate(evaluations):
            console.print(f"[dim]Evaluating {i + 1}/{len(evaluations)}[/dim]")

            evaluation = self.evaluate_answer(
                question=eval_data.get("question", ""),
                expected_answer=eval_data.get("expected_answer", ""),
                model_answer=eval_data.get("model_answer", ""),
                context_preview=eval_data.get("context_preview", ""),
                approach_used=eval_data.get("approach_used", ""),
                **kwargs,
            )
            results.append(evaluation)

        # Calculate batch statistics
        correct_count = sum(1 for r in results if r.is_correct)
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0

        console.print(
            f"[green]Batch evaluation complete: {correct_count}/{len(results)} correct, avg confidence: {avg_confidence:.3f}[/green]"
        )

        return results

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON object from judge response, handling various formats."""
        response = response.strip()

        # Try to find JSON block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
            else:
                json_str = response[start:].strip()
        elif response.startswith("{") and response.endswith("}"):
            json_str = response
        else:
            # Try to find JSON-like structure
            start_brace = response.find("{")
            end_brace = response.rfind("}")
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str = response[start_brace : end_brace + 1]
            else:
                raise json.JSONDecodeError("No JSON found in response", response, 0)

        return json.loads(json_str)

    def _create_fallback_evaluation(
        self,
        question: str,
        expected_answer: str,
        model_answer: str,
        approach_used: str,
        evaluation_time_ms: float,
        error: str,
        raw_response: str = "",
    ) -> JudgeEvaluation:
        """Create a fallback evaluation when judge fails."""
        # Simple heuristic fallback
        expected_lower = expected_answer.lower().strip()
        model_lower = model_answer.lower().strip()

        # Basic similarity check for absolute correctness
        is_correct = expected_lower in model_lower or model_lower in expected_lower
        confidence = 0.3 if is_correct else 0.1  # Low confidence for fallback

        # For fallback, assume context-grounded correctness is similar to absolute
        # but slightly more lenient
        is_correct_given_context = is_correct or len(model_answer.strip()) > 10  # Any substantial answer
        context_confidence = 0.2 if is_correct_given_context else 0.1

        retrieval_gap = float(is_correct_given_context) - float(is_correct)

        return JudgeEvaluation(
            question=question,
            expected_answer=expected_answer,
            model_answer=model_answer,
            is_correct=is_correct,
            confidence=confidence,
            reasoning=f"Fallback evaluation due to judge error: {error}",
            answer_completeness=0.5 if is_correct else 0.2,
            context_utilization=0.5,  # Unknown
            approach_used=approach_used,
            judge_model=self.judge_model,
            evaluation_time_ms=evaluation_time_ms,
            raw_judge_response=raw_response,
            metadata={"fallback_used": True, "error": error},
            # New dual correctness fields
            is_correct_given_context=is_correct_given_context,
            context_grounded_confidence=context_confidence,
            context_grounded_reasoning=f"Fallback context evaluation: {error}",
            retrieval_gap=retrieval_gap,
        )

    def analyze_evaluation_patterns(self, evaluations: List[JudgeEvaluation]) -> Dict[str, Any]:
        """
        Analyze patterns in a set of evaluations.

        Args:
            evaluations: List of JudgeEvaluation objects to analyze

        Returns:
            Dictionary with analysis results
        """
        if not evaluations:
            return {"error": "No evaluations to analyze"}

        # Basic statistics
        total_count = len(evaluations)
        correct_count = sum(1 for e in evaluations if e.is_correct)
        accuracy = correct_count / total_count

        # Context-grounded statistics
        context_correct_count = sum(1 for e in evaluations if e.is_correct_given_context)
        context_accuracy = context_correct_count / total_count

        # Retrieval gap analysis
        retrieval_gaps = [e.retrieval_gap for e in evaluations]
        avg_retrieval_gap = sum(retrieval_gaps) / len(retrieval_gaps)

        # Confidence statistics
        confidences = [e.confidence for e in evaluations]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_of_correct = [e.confidence for e in evaluations if e.is_correct]
        confidence_of_incorrect = [e.confidence for e in evaluations if not e.is_correct]

        # Context-grounded confidence statistics
        context_confidences = [e.context_grounded_confidence for e in evaluations]
        avg_context_confidence = sum(context_confidences) / len(context_confidences)
        context_confidence_of_correct = [
            e.context_grounded_confidence for e in evaluations if e.is_correct_given_context
        ]
        context_confidence_of_incorrect = [
            e.context_grounded_confidence for e in evaluations if not e.is_correct_given_context
        ]

        # By approach
        approach_stats = {}
        for evaluation in evaluations:
            approach = evaluation.approach_used or "unknown"
            if approach not in approach_stats:
                approach_stats[approach] = {
                    "total": 0,
                    "correct": 0,
                    "context_correct": 0,
                    "confidences": [],
                    "context_confidences": [],
                    "retrieval_gaps": [],
                }

            approach_stats[approach]["total"] += 1
            if evaluation.is_correct:
                approach_stats[approach]["correct"] += 1
            if evaluation.is_correct_given_context:
                approach_stats[approach]["context_correct"] += 1
            approach_stats[approach]["confidences"].append(evaluation.confidence)
            approach_stats[approach]["context_confidences"].append(evaluation.context_grounded_confidence)
            approach_stats[approach]["retrieval_gaps"].append(evaluation.retrieval_gap)

        # Calculate approach accuracies
        for approach, stats in approach_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            stats["context_accuracy"] = stats["context_correct"] / stats["total"] if stats["total"] > 0 else 0
            stats["avg_confidence"] = (
                sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0
            )
            stats["avg_context_confidence"] = (
                sum(stats["context_confidences"]) / len(stats["context_confidences"])
                if stats["context_confidences"]
                else 0
            )
            stats["avg_retrieval_gap"] = (
                sum(stats["retrieval_gaps"]) / len(stats["retrieval_gaps"]) if stats["retrieval_gaps"] else 0
            )

        analysis = {
            "total_evaluations": total_count,
            "overall_accuracy": accuracy,
            "correct_count": correct_count,
            "average_confidence": avg_confidence,
            "confidence_when_correct": sum(confidence_of_correct) / len(confidence_of_correct)
            if confidence_of_correct
            else 0,
            "confidence_when_incorrect": sum(confidence_of_incorrect) / len(confidence_of_incorrect)
            if confidence_of_incorrect
            else 0,
            # New dual correctness metrics
            "context_grounded_accuracy": context_accuracy,
            "context_correct_count": context_correct_count,
            "average_context_confidence": avg_context_confidence,
            "context_confidence_when_correct": sum(context_confidence_of_correct) / len(context_confidence_of_correct)
            if context_confidence_of_correct
            else 0,
            "context_confidence_when_incorrect": sum(context_confidence_of_incorrect)
            / len(context_confidence_of_incorrect)
            if context_confidence_of_incorrect
            else 0,
            "average_retrieval_gap": avg_retrieval_gap,
            "retrieval_quality_insight": self._interpret_retrieval_gap(avg_retrieval_gap),
            "approach_breakdown": approach_stats,
            "fallback_count": sum(1 for e in evaluations if e.metadata.get("fallback_used", False)),
        }

        return analysis

    def _interpret_retrieval_gap(self, avg_gap: float) -> str:
        """Interpret the retrieval gap metric."""
        if avg_gap > 0.3:
            return "Poor retrieval quality: Models can answer correctly with context but lack ground truth information"
        elif avg_gap > 0.1:
            return "Moderate retrieval gap: Some questions answered correctly given context but not absolutely"
        elif avg_gap > -0.1:
            return "Good retrieval quality: Context-grounded and absolute correctness are well aligned"
        else:
            return "Context may be misleading: Absolute answers are more correct than context-grounded ones"
