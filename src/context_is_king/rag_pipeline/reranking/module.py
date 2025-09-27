"""
Cross-encoder reranking module for improved retrieval relevance
"""

import time
from typing import Any

import torch
from rich.console import Console
from rich.progress import track
from sentence_transformers import CrossEncoder

from ..types import PipelineConfig, RetrievalResult

console = Console()


class RerankerModule:
    """Cross-encoder reranking for improved relevance"""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize cross-encoder with best available device
        device = self._get_best_device(config.reranker_device)
        console.print(f"[blue]Initializing reranker model on {device}[/blue]")

        self.reranker = CrossEncoder(config.reranker_model, device=device)

        console.print(f"[green]RerankerModule initialized with {config.reranker_model} on {device}[/green]")

    def rerank_results(
        self, query: str, retrieval_results: list[RetrievalResult], verbose: bool = False
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder

        Args:
            query: Original search query
            retrieval_results: Initial retrieval results to rerank

        Returns:
            Reranked results with updated scores and rankings
        """
        if not retrieval_results:
            console.print("[yellow]No results to rerank[/yellow]")
            return []

        if verbose:
            console.print(f"[blue]Reranking {len(retrieval_results)} results[/blue]")
        start_time = time.time()

        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [(query, result.chunk.content) for result in retrieval_results]

            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)

            # Update results with reranking scores
            reranked_results = []
            for i, (result, rerank_score) in enumerate(zip(retrieval_results, rerank_scores, strict=False)):
                reranked_result = result.model_copy(update={"rerank_score": float(rerank_score), "reranked": True})
                reranked_results.append(reranked_result)

            # Sort by reranking score (descending)
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

            # Update ranks
            for i, result in enumerate(reranked_results):
                reranked_results[i] = result.model_copy(update={"rank": i + 1})

            rerank_time = time.time() - start_time
            if verbose:
                console.print(f"[green]Reranking completed in {rerank_time:.3f}s[/green]")

            return reranked_results

        except Exception as e:
            console.print(f"[red]Reranking failed: {e}[/red]")
            # Return original results if reranking fails
            return retrieval_results

    def batch_rerank(
        self, queries: list[str], results_lists: list[list[RetrievalResult]]
    ) -> list[list[RetrievalResult]]:
        """
        Batch reranking for efficiency

        Args:
            queries: List of queries
            results_lists: List of retrieval result lists (one per query)

        Returns:
            List of reranked result lists
        """
        if len(queries) != len(results_lists):
            raise ValueError("Number of queries must match number of result lists")

        console.print(f"[blue]Batch reranking {len(queries)} query-result pairs[/blue]")

        reranked_lists = []
        for query, results in track(
            zip(queries, results_lists, strict=False), description="Reranking batches...", total=len(queries)
        ):
            reranked = self.rerank_results(query, results)
            reranked_lists.append(reranked)

        return reranked_lists

    def analyze_reranking_impact(
        self, original: list[RetrievalResult], reranked: list[RetrievalResult]
    ) -> dict[str, Any]:
        """
        Analyze how reranking changed the results

        Args:
            original: Original retrieval results
            reranked: Reranked results

        Returns:
            Analysis of reranking impact
        """
        if len(original) != len(reranked):
            return {"error": "Result lists must have same length"}

        if not original:
            return {"error": "No results to analyze"}

        # Create mapping from chunk_id to position for easy comparison
        original_positions = {result.chunk.full_id: result.rank for result in original}
        reranked_positions = {result.chunk.full_id: result.rank for result in reranked}

        # Calculate position changes
        position_changes = []
        for chunk_id in original_positions:
            original_pos = original_positions[chunk_id]
            reranked_pos = reranked_positions.get(chunk_id, original_pos)
            change = original_pos - reranked_pos  # Positive = moved up
            position_changes.append(change)

        # Calculate rank correlation
        rank_correlation = self._calculate_rank_correlation(original, reranked)

        # Identify biggest movers
        biggest_movers = self._identify_biggest_movers(original, reranked)

        # Score changes analysis
        score_changes = self._analyze_score_changes(original, reranked)

        analysis = {
            "total_results": len(original),
            "position_changes": {
                "mean_change": sum(position_changes) / len(position_changes) if position_changes else 0,
                "max_improvement": max(position_changes) if position_changes else 0,
                "max_degradation": min(position_changes) if position_changes else 0,
                "results_improved": len([c for c in position_changes if c > 0]),
                "results_degraded": len([c for c in position_changes if c < 0]),
                "results_unchanged": len([c for c in position_changes if c == 0]),
            },
            "rank_correlation": rank_correlation,
            "biggest_movers": biggest_movers,
            "score_changes": score_changes,
            "top_3_original": [
                {
                    "rank": r.rank,
                    "similarity": r.similarity_score,
                    "doc_id": r.chunk.doc_id,
                    "chunk_id": r.chunk.chunk_id,
                }
                for r in original[:3]
            ],
            "top_3_reranked": [
                {
                    "rank": r.rank,
                    "similarity": r.similarity_score,
                    "rerank_score": r.rerank_score,
                    "doc_id": r.chunk.doc_id,
                    "chunk_id": r.chunk.chunk_id,
                }
                for r in reranked[:3]
            ],
        }

        return analysis

    def evaluate_reranking_effectiveness(
        self,
        query: str,
        original_results: list[RetrievalResult],
        reranked_results: list[RetrievalResult],
        ground_truth_chunk_ids: list[str],
    ) -> dict[str, float]:
        """
        Evaluate reranking effectiveness against ground truth

        Args:
            query: Original query
            original_results: Results before reranking
            reranked_results: Results after reranking
            ground_truth_chunk_ids: List of relevant chunk IDs

        Returns:
            Comparison metrics
        """
        if not ground_truth_chunk_ids:
            return {"warning": "no_ground_truth"}

        ground_truth_set = set(ground_truth_chunk_ids)

        # Calculate metrics for both result sets
        original_metrics = self._calculate_ranking_metrics(original_results, ground_truth_set)
        reranked_metrics = self._calculate_ranking_metrics(reranked_results, ground_truth_set)

        # Calculate improvements
        improvements = {}
        for metric in original_metrics:
            if metric in reranked_metrics:
                improvement = reranked_metrics[metric] - original_metrics[metric]
                improvements[f"{metric}_improvement"] = improvement

        return {
            "query": query,
            "ground_truth_count": len(ground_truth_chunk_ids),
            "original_metrics": original_metrics,
            "reranked_metrics": reranked_metrics,
            "improvements": improvements,
        }

    def warm_up_model(self) -> None:
        """Warm up the reranker model with dummy predictions"""
        console.print("[blue]Warming up reranker model...[/blue]")
        start_time = time.time()

        # Dummy query-document pairs
        dummy_pairs = [
            ("What is machine learning?", "Machine learning is a subset of artificial intelligence."),
            ("How does deep learning work?", "Deep learning uses neural networks with multiple layers."),
            (
                "What is natural language processing?",
                "NLP focuses on the interaction between computers and human language.",
            ),
        ]

        self.reranker.predict(dummy_pairs)

        warmup_time = time.time() - start_time
        console.print(f"[green]Reranker warmed up in {warmup_time:.2f}s[/green]")

    def get_model_stats(self) -> dict[str, Any]:
        """Get reranker model statistics"""
        device_str = str(self.reranker.device)
        stats = {
            "model_name": self.config.reranker_model,
            "device": device_str,
            "max_length": getattr(self.reranker, "max_length", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }

        # Add GPU memory stats if using CUDA
        if torch.cuda.is_available() and "cuda" in device_str:
            stats.update(
                {
                    "gpu_memory_allocated": torch.cuda.memory_allocated(),
                    "gpu_memory_cached": torch.cuda.memory_reserved(),
                }
            )

        return stats

    def _calculate_rank_correlation(self, original: list[RetrievalResult], reranked: list[RetrievalResult]) -> float:
        """Calculate Spearman rank correlation between original and reranked results"""
        if len(original) != len(reranked):
            return 0.0

        # Create rank mappings
        original_ranks = {result.chunk.full_id: result.rank for result in original}
        reranked_ranks = {result.chunk.full_id: result.rank for result in reranked}

        # Get paired ranks
        paired_ranks = [
            (original_ranks[chunk_id], reranked_ranks[chunk_id])
            for chunk_id in original_ranks
            if chunk_id in reranked_ranks
        ]

        if len(paired_ranks) < 2:
            return 0.0

        # Simple Spearman correlation calculation
        n = len(paired_ranks)
        sum_d_squared = sum((x - y) ** 2 for x, y in paired_ranks)
        correlation = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))

        return correlation

    def _identify_biggest_movers(
        self, original: list[RetrievalResult], reranked: list[RetrievalResult]
    ) -> dict[str, list[dict]]:
        """Identify results that moved most in ranking"""
        # Create position mappings
        original_positions = {result.chunk.full_id: result.rank for result in original}
        reranked_positions = {result.chunk.full_id: result.rank for result in reranked}

        # Calculate moves
        moves = []
        for chunk_id in original_positions:
            if chunk_id in reranked_positions:
                original_pos = original_positions[chunk_id]
                reranked_pos = reranked_positions[chunk_id]
                move = original_pos - reranked_pos  # Positive = moved up

                # Find the actual result objects
                original_result = next(r for r in original if r.chunk.full_id == chunk_id)
                reranked_result = next(r for r in reranked if r.chunk.full_id == chunk_id)

                moves.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": original_result.chunk.doc_id,
                        "move": move,
                        "original_rank": original_pos,
                        "reranked_rank": reranked_pos,
                        "original_similarity": original_result.similarity_score,
                        "rerank_score": reranked_result.rerank_score,
                        "content_preview": original_result.chunk.content[:100] + "...",
                    }
                )

        # Sort by absolute move size
        moves.sort(key=lambda x: abs(x["move"]), reverse=True)

        return {
            "biggest_improvements": [m for m in moves[:5] if m["move"] > 0],
            "biggest_degradations": [m for m in moves[:5] if m["move"] < 0],
        }

    def _analyze_score_changes(
        self, original: list[RetrievalResult], reranked: list[RetrievalResult]
    ) -> dict[str, float]:
        """Analyze how scores changed between original and reranked results"""
        if not reranked or not all(r.rerank_score is not None for r in reranked):
            return {"error": "Missing rerank scores"}

        similarity_scores = [r.similarity_score for r in original]
        rerank_scores = [r.rerank_score for r in reranked if r.rerank_score is not None]

        return {
            "similarity_score_mean": sum(similarity_scores) / len(similarity_scores),
            "similarity_score_max": max(similarity_scores),
            "similarity_score_min": min(similarity_scores),
            "rerank_score_mean": sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0,
            "rerank_score_max": max(rerank_scores) if rerank_scores else 0,
            "rerank_score_min": min(rerank_scores) if rerank_scores else 0,
            "score_correlation": self._calculate_score_correlation(similarity_scores, rerank_scores),
        }

    def _calculate_ranking_metrics(self, results: list[RetrievalResult], ground_truth_set: set) -> dict[str, float]:
        """Calculate ranking metrics for a result set"""
        if not results:
            return {}

        result_ids = [r.chunk.full_id for r in results]

        metrics = {}

        # Precision@k and Recall@k for k=1,3,5,10
        for k in [1, 3, 5, 10]:
            if k <= len(result_ids):
                top_k = set(result_ids[:k])
                precision = len(top_k.intersection(ground_truth_set)) / k
                recall = len(top_k.intersection(ground_truth_set)) / len(ground_truth_set)

                metrics[f"precision@{k}"] = precision
                metrics[f"recall@{k}"] = recall

        # MRR
        mrr = 0.0
        for i, result_id in enumerate(result_ids):
            if result_id in ground_truth_set:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr"] = mrr

        return metrics

    def _calculate_score_correlation(self, scores1: list[float], scores2: list[float]) -> float:
        """Calculate Pearson correlation between two score lists"""
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0

        mean1 = sum(scores1) / len(scores1)
        mean2 = sum(scores2) / len(scores2)

        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(scores1, scores2, strict=False))

        sum_sq1 = sum((x - mean1) ** 2 for x in scores1)
        sum_sq2 = sum((y - mean2) ** 2 for y in scores2)

        denominator = (sum_sq1 * sum_sq2) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _get_best_device(self, preferred_device: str = "auto") -> str:
        """Determine the best device for the reranker model."""
        if preferred_device != "auto":
            return preferred_device

        # Check for MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            console.print("[green]MPS GPU detected and available[/green]")
            return "mps"

        # Check for CUDA
        elif torch.cuda.is_available():
            console.print("[green]CUDA GPU detected and available[/green]")
            return "cuda"

        # Fallback to CPU
        else:
            console.print("[yellow]No GPU available, using CPU[/yellow]")
            return "cpu"
