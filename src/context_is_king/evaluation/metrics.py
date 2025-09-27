#!/usr/bin/env python3
"""
Evaluation Metrics for Context Window Advantage Experiments

This module provides comprehensive evaluation metrics for both needle-in-haystack
and LongMemEval experiments, enabling consistent performance comparison across
different context window sizes and model capacities.

Usage:
    from metrics import NeedleEvaluator, LongMemEvalEvaluator

    evaluator = NeedleEvaluator()
    score = evaluator.evaluate_retrieval("Expected answer", "Model response")
"""

import re
import string
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
import difflib
from pathlib import Path
import json
import numpy as np


@dataclass
class RetrievalScore:
    """Score for a single needle retrieval task."""

    exact_match: bool
    partial_match_score: float  # 0.0 to 1.0
    semantic_similarity: float  # 0.0 to 1.0 (simplified)
    position_accuracy: bool  # Whether needle was found in correct position
    completeness_score: float  # How much of the needle was retrieved
    overall_score: float  # Combined score


@dataclass
class CrossReferenceScore:
    """Score for cross-reference tasks requiring multiple needles."""

    individual_retrievals: List[RetrievalScore]
    connection_accuracy: float  # How well connections were made
    synthesis_quality: float  # Quality of information synthesis
    overall_score: float


@dataclass
class PositionAnalysis:
    """Analysis of performance by needle position."""

    position_ranges: Dict[str, List[float]]  # {'beginning': [scores], 'middle': [scores], ...}
    position_bias: float  # Measure of position bias
    best_position: str
    worst_position: str


@dataclass
class ExperimentResults:
    """Complete results for an experiment run."""

    experiment_id: str
    model_name: str
    context_size: int
    composition: str  # 'pg_heavy', 'arxiv_heavy', 'mixed'

    # Individual test results
    retrieval_scores: List[RetrievalScore]
    cross_reference_scores: List[CrossReferenceScore]

    # Aggregate metrics
    avg_retrieval_accuracy: float
    avg_cross_reference_accuracy: float
    position_analysis: PositionAnalysis

    # Performance metrics
    avg_response_time: float
    total_cost: float
    success_rate: float


class NeedleEvaluator:
    """Evaluator for needle-in-haystack experiments."""

    def __init__(self):
        """Initialize the needle evaluator."""
        self.stop_words = set(
            [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
            ]
        )

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove punctuation at start/end
        text = text.strip(string.punctuation)

        return text

    def extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text, excluding stop words."""
        normalized = self.normalize_text(text)
        words = normalized.split()

        # Filter out stop words and short words
        key_terms = set()
        for word in words:
            word = word.strip(string.punctuation)
            if len(word) > 2 and word not in self.stop_words:
                key_terms.add(word)

        return key_terms

    def calculate_exact_match(self, expected: str, actual: str) -> bool:
        """Check for exact match after normalization."""
        return self.normalize_text(expected) == self.normalize_text(actual)

    def calculate_partial_match(self, expected: str, actual: str) -> float:
        """Calculate partial match score using token overlap."""
        expected_terms = self.extract_key_terms(expected)
        actual_terms = self.extract_key_terms(actual)

        if not expected_terms:
            return 1.0 if not actual_terms else 0.0

        # Calculate Jaccard similarity
        intersection = len(expected_terms.intersection(actual_terms))
        union = len(expected_terms.union(actual_terms))

        return intersection / union if union > 0 else 0.0

    def calculate_semantic_similarity(self, expected: str, actual: str) -> float:
        """Calculate semantic similarity (simplified version using string similarity)."""
        # This is a simplified version - in practice, you'd use sentence embeddings

        expected_norm = self.normalize_text(expected)
        actual_norm = self.normalize_text(actual)

        if not expected_norm or not actual_norm:
            return 0.0

        # Use difflib sequence matcher as a proxy for semantic similarity
        similarity = difflib.SequenceMatcher(None, expected_norm, actual_norm).ratio()

        # Boost score if key terms are present
        key_term_bonus = self.calculate_partial_match(expected, actual) * 0.3

        return min(similarity + key_term_bonus, 1.0)

    def calculate_completeness(self, expected: str, actual: str) -> float:
        """Calculate how complete the retrieved information is."""
        expected_terms = self.extract_key_terms(expected)
        actual_terms = self.extract_key_terms(actual)

        if not expected_terms:
            return 1.0

        # What fraction of expected terms were retrieved?
        retrieved_terms = len(expected_terms.intersection(actual_terms))
        return retrieved_terms / len(expected_terms)

    def evaluate_retrieval(
        self, expected_answer: str, actual_answer: str, needle_position: Optional[str] = None
    ) -> RetrievalScore:
        """Evaluate a single needle retrieval task."""

        exact_match = self.calculate_exact_match(expected_answer, actual_answer)
        partial_match = self.calculate_partial_match(expected_answer, actual_answer)
        semantic_similarity = self.calculate_semantic_similarity(expected_answer, actual_answer)
        completeness = self.calculate_completeness(expected_answer, actual_answer)

        # Position accuracy (simplified - assumes we know if needle was found in right spot)
        position_accuracy = True  # Default to True, would need more sophisticated checking

        # Calculate overall score (weighted combination)
        overall_score = (
            0.3 * (1.0 if exact_match else 0.0) + 0.25 * partial_match + 0.25 * semantic_similarity + 0.2 * completeness
        )

        return RetrievalScore(
            exact_match=exact_match,
            partial_match_score=partial_match,
            semantic_similarity=semantic_similarity,
            position_accuracy=position_accuracy,
            completeness_score=completeness,
            overall_score=overall_score,
        )

    def evaluate_cross_reference(
        self,
        expected_connections: List[str],
        actual_response: str,
        individual_needles: List[Tuple[str, str]],  # [(expected, actual), ...]
    ) -> CrossReferenceScore:
        """Evaluate a cross-reference task requiring multiple needles."""

        # Evaluate individual needle retrievals
        individual_scores = []
        for expected, actual in individual_needles:
            score = self.evaluate_retrieval(expected, actual)
            individual_scores.append(score)

        # Evaluate connection accuracy
        connection_accuracy = self._evaluate_connections(expected_connections, actual_response)

        # Evaluate synthesis quality
        synthesis_quality = self._evaluate_synthesis(expected_connections, actual_response)

        # Calculate overall score
        avg_individual = statistics.mean([s.overall_score for s in individual_scores])
        overall_score = 0.4 * avg_individual + 0.3 * connection_accuracy + 0.3 * synthesis_quality

        return CrossReferenceScore(
            individual_retrievals=individual_scores,
            connection_accuracy=connection_accuracy,
            synthesis_quality=synthesis_quality,
            overall_score=overall_score,
        )

    def _evaluate_connections(self, expected_connections: List[str], actual_response: str) -> float:
        """Evaluate how well connections between needles were made."""
        if not expected_connections:
            return 1.0

        # Simple keyword-based connection detection
        connection_score = 0.0
        for connection in expected_connections:
            connection_terms = self.extract_key_terms(connection)
            response_terms = self.extract_key_terms(actual_response)

            if connection_terms.intersection(response_terms):
                connection_score += 1.0

        return connection_score / len(expected_connections)

    def _evaluate_synthesis(self, expected_connections: List[str], actual_response: str) -> float:
        """Evaluate quality of information synthesis."""
        # Simplified synthesis evaluation
        response_length = len(actual_response.split())

        # Penalize too short responses
        if response_length < 20:
            length_penalty = response_length / 20.0
        else:
            length_penalty = 1.0

        # Reward use of connecting words
        connecting_words = [
            "however",
            "moreover",
            "furthermore",
            "in contrast",
            "similarly",
            "therefore",
            "consequently",
            "additionally",
            "because",
            "although",
        ]

        connecting_bonus = min(sum(1 for word in connecting_words if word in actual_response.lower()) / 3.0, 0.3)

        return min(length_penalty + connecting_bonus, 1.0)

    def analyze_position_bias(self, results: List[Tuple[str, RetrievalScore]]) -> PositionAnalysis:
        """Analyze performance by needle position."""
        position_scores = defaultdict(list)

        # Group scores by position
        for position, score in results:
            position_scores[position].append(score.overall_score)

        # Calculate statistics
        position_ranges = {}
        for position, scores in position_scores.items():
            position_ranges[position] = scores

        # Find best and worst positions
        avg_scores = {pos: statistics.mean(scores) for pos, scores in position_ranges.items()}
        best_position = max(avg_scores.keys(), key=lambda k: avg_scores[k])
        worst_position = min(avg_scores.keys(), key=lambda k: avg_scores[k])

        # Calculate position bias (standard deviation of position averages)
        position_bias = statistics.stdev(avg_scores.values()) if len(avg_scores) > 1 else 0.0

        return PositionAnalysis(
            position_ranges=dict(position_ranges),
            position_bias=position_bias,
            best_position=best_position,
            worst_position=worst_position,
        )


class LongMemEvalEvaluator:
    """Evaluator for LongMemEval experiments."""

    def __init__(self):
        """Initialize the LongMemEval evaluator."""
        self.needle_evaluator = NeedleEvaluator()

    def evaluate_answer(self, expected: str, actual: str, question_type: str) -> float:
        """Evaluate a LongMemEval answer."""
        if question_type == "factual":
            return self._evaluate_factual(expected, actual)
        elif question_type == "reasoning":
            return self._evaluate_reasoning(expected, actual)
        elif question_type == "synthesis":
            return self._evaluate_synthesis(expected, actual)
        elif question_type == "multi_hop":
            return self._evaluate_multi_hop(expected, actual)
        else:
            # Default to partial match evaluation
            return self.needle_evaluator.calculate_partial_match(expected, actual)

    def _evaluate_factual(self, expected: str, actual: str) -> float:
        """Evaluate factual questions (strict matching)."""
        retrieval_score = self.needle_evaluator.evaluate_retrieval(expected, actual)
        return retrieval_score.overall_score

    def _evaluate_reasoning(self, expected: str, actual: str) -> float:
        """Evaluate reasoning questions (more flexible)."""
        # For reasoning questions, focus on semantic similarity and completeness
        semantic_score = self.needle_evaluator.calculate_semantic_similarity(expected, actual)
        completeness_score = self.needle_evaluator.calculate_completeness(expected, actual)

        # Weight semantic understanding more heavily for reasoning
        return 0.6 * semantic_score + 0.4 * completeness_score

    def _evaluate_synthesis(self, expected: str, actual: str) -> float:
        """Evaluate synthesis questions (focus on connections)."""
        # Check for synthesis indicators
        synthesis_indicators = [
            "combine",
            "connect",
            "integrate",
            "relate",
            "compare",
            "contrast",
            "both",
            "either",
            "neither",
            "similar",
            "different",
            "however",
            "although",
        ]

        actual_lower = actual.lower()
        synthesis_bonus = min(sum(1 for indicator in synthesis_indicators if indicator in actual_lower) / 5.0, 0.3)

        base_score = self.needle_evaluator.calculate_semantic_similarity(expected, actual)
        return min(base_score + synthesis_bonus, 1.0)

    def _evaluate_multi_hop(self, expected: str, actual: str) -> float:
        """Evaluate multi-hop reasoning questions."""
        # Multi-hop questions require connecting information across documents
        # Look for evidence of multiple information sources

        # Count potential information hops (simplified)
        hop_indicators = actual.split(".")  # Sentences as proxy for reasoning steps
        hop_bonus = min(len([s for s in hop_indicators if len(s.strip()) > 10]) / 10.0, 0.2)

        base_score = self._evaluate_reasoning(expected, actual)
        return min(base_score + hop_bonus, 1.0)

    def evaluate_context_group_performance(
        self,
        results: List[Tuple[str, str, str, float]],  # [(question_type, expected, actual, context_tokens), ...]
        group_name: str,
    ) -> Dict[str, Any]:
        """Evaluate performance for a context group."""

        if not results:
            return {
                "group_name": group_name,
                "total_questions": 0,
                "avg_score": 0.0,
                "question_type_scores": {},
                "context_size_correlation": 0.0,
            }

        # Calculate scores by question type
        type_scores = defaultdict(list)
        context_scores = []

        for question_type, expected, actual, context_tokens in results:
            score = self.evaluate_answer(expected, actual, question_type)
            type_scores[question_type].append(score)
            context_scores.append((context_tokens, score))

        # Calculate averages
        avg_scores_by_type = {qtype: statistics.mean(scores) for qtype, scores in type_scores.items()}

        # Calculate overall average score
        all_scores = []
        for question_type, expected, actual, context_tokens in results:
            score = self.evaluate_answer(expected, actual, question_type)
            all_scores.append(score)
        overall_avg = statistics.mean(all_scores) if all_scores else 0.0

        # Calculate context size correlation (simplified)
        if len(context_scores) > 1:
            context_sizes = [size for size, _ in context_scores]
            scores = [score for _, score in context_scores]
            correlation = np.corrcoef(context_sizes, scores)[0, 1] if len(set(context_sizes)) > 1 else 0.0
        else:
            correlation = 0.0

        return {
            "group_name": group_name,
            "total_questions": len(results),
            "avg_score": overall_avg,
            "question_type_scores": avg_scores_by_type,
            "context_size_correlation": correlation,
            "score_distribution": {
                "min": min(scores),
                "max": max(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            },
        }


class ResultsAggregator:
    """Aggregates and analyzes experiment results."""

    @staticmethod
    def aggregate_model_performance(results: List[ExperimentResults]) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by model."""
        model_results = defaultdict(list)

        # Group by model
        for result in results:
            model_results[result.model_name].append(result)

        # Aggregate statistics
        aggregated = {}
        for model_name, model_experiments in model_results.items():
            avg_retrieval = statistics.mean([r.avg_retrieval_accuracy for r in model_experiments])
            avg_cross_ref = statistics.mean([r.avg_cross_reference_accuracy for r in model_experiments])
            avg_response_time = statistics.mean([r.avg_response_time for r in model_experiments])
            total_cost = sum([r.total_cost for r in model_experiments])
            avg_success_rate = statistics.mean([r.success_rate for r in model_experiments])

            # Context size performance
            context_performance = defaultdict(list)
            for exp in model_experiments:
                context_performance[exp.context_size].append(exp.avg_retrieval_accuracy)

            context_size_scores = {size: statistics.mean(scores) for size, scores in context_performance.items()}

            aggregated[model_name] = {
                "avg_retrieval_accuracy": avg_retrieval,
                "avg_cross_reference_accuracy": avg_cross_ref,
                "avg_response_time": avg_response_time,
                "total_cost": total_cost,
                "avg_success_rate": avg_success_rate,
                "context_size_performance": context_size_scores,
                "num_experiments": len(model_experiments),
            }

        return aggregated

    @staticmethod
    def compare_context_capacity_groups(
        high_capacity_results: Dict[str, Any], medium_capacity_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare high vs medium capacity model groups."""

        comparison = {
            "high_capacity": {
                "models": list(high_capacity_results.keys()),
                "avg_performance": statistics.mean(
                    [data["avg_retrieval_accuracy"] for data in high_capacity_results.values()]
                ),
                "avg_cost": statistics.mean([data["total_cost"] for data in high_capacity_results.values()]),
                "avg_response_time": statistics.mean(
                    [data["avg_response_time"] for data in high_capacity_results.values()]
                ),
            },
            "medium_capacity": {
                "models": list(medium_capacity_results.keys()),
                "avg_performance": statistics.mean(
                    [data["avg_retrieval_accuracy"] for data in medium_capacity_results.values()]
                ),
                "avg_cost": statistics.mean([data["total_cost"] for data in medium_capacity_results.values()]),
                "avg_response_time": statistics.mean(
                    [data["avg_response_time"] for data in medium_capacity_results.values()]
                ),
            },
        }

        # Performance difference
        perf_diff = comparison["high_capacity"]["avg_performance"] - comparison["medium_capacity"]["avg_performance"]

        # Cost efficiency
        high_efficiency = comparison["high_capacity"]["avg_performance"] / comparison["high_capacity"]["avg_cost"]
        medium_efficiency = comparison["medium_capacity"]["avg_performance"] / comparison["medium_capacity"]["avg_cost"]

        comparison["analysis"] = {
            "performance_advantage_high_capacity": perf_diff,
            "cost_efficiency_high_capacity": high_efficiency,
            "cost_efficiency_medium_capacity": medium_efficiency,
            "efficiency_ratio": high_efficiency / medium_efficiency if medium_efficiency > 0 else 0.0,
        }

        return comparison

    @staticmethod
    def save_results(results: Dict[str, Any], output_path: Path) -> None:
        """Save aggregated results to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj

        serializable_results = convert_types(results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"💾 Saved aggregated results to {output_path}")


def main():
    """Test the metrics module."""
    print("🧮 Testing Metrics Module")

    # Test needle evaluator
    evaluator = NeedleEvaluator()

    # Test exact match
    score1 = evaluator.evaluate_retrieval("Paul Graham founded Y Combinator", "Paul Graham founded Y Combinator")
    print(f"Exact match score: {score1.overall_score:.3f}")

    # Test partial match
    score2 = evaluator.evaluate_retrieval("Paul Graham founded Y Combinator", "Y Combinator was founded by Paul Graham")
    print(f"Partial match score: {score2.overall_score:.3f}")

    # Test LongMemEval evaluator
    longmem_evaluator = LongMemEvalEvaluator()
    score3 = longmem_evaluator.evaluate_answer(
        "The research was conducted in 2023", "According to the paper, the research took place in 2023", "factual"
    )
    print(f"LongMemEval factual score: {score3:.3f}")

    print("✅ Metrics module test completed")


if __name__ == "__main__":
    main()
