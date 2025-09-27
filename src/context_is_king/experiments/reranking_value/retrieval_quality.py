"""
Retrieval Quality Assessment System for Reranking Value Experiment.

Provides comprehensive retrieval quality metrics (Recall@K, MRR, NDCG) by:
1. Generating ground truth chunk annotations through semantic similarity
2. Calculating standard information retrieval metrics
3. Analyzing retrieval effectiveness across different approaches

This addresses the research scientist's recommendation for intermediate
retrieval quality assessment beyond just final answer correctness.
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

console = Console()


@dataclass
class GroundTruthChunk:
    """Represents a chunk with ground truth relevance annotation."""
    chunk_id: str
    content: str
    relevance_score: float  # 0.0-1.0, higher = more relevant
    is_relevant: bool  # Binary relevance (usually >0.5 threshold)
    semantic_similarity: float  # Similarity to expected answer
    contains_answer_keywords: bool  # Whether chunk contains key answer terms


@dataclass
class RetrievalQualityResult:
    """Results from retrieval quality assessment."""
    question_id: str
    approach: str
    
    # Retrieval metrics
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    
    # Ranking quality metrics
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_3: float  # Normalized Discounted Cumulative Gain
    ndcg_at_5: float
    ndcg_at_10: float
    
    # Detailed analysis
    total_retrieved: int
    relevant_retrieved: int
    total_relevant: int
    precision_at_5: float
    
    # Ground truth details
    ground_truth_chunks: List[str]  # IDs of relevant chunks
    retrieved_chunks: List[str]     # IDs of retrieved chunks
    relevance_scores: List[float]   # Relevance scores of retrieved chunks


class GroundTruthAnnotator:
    """
    Creates ground truth chunk-level relevance annotations for retrieval evaluation.
    
    Uses semantic similarity and keyword matching to identify which chunks
    contain information relevant to answering each question.
    """
    
    def __init__(self, similarity_threshold: float = 0.6, keyword_weight: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self.keyword_weight = keyword_weight
        
        # Initialize semantic similarity model
        console.print("[blue]Loading semantic similarity model...[/blue]")
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            console.print("[green]✓ Semantic similarity model loaded[/green]")
        except Exception as e:
            console.print(f"[red]⚠ Could not load similarity model: {e}[/red]")
            console.print("[yellow]Using keyword-only annotation[/yellow]")
            self.similarity_model = None
    
    def annotate_chunks_for_question(
        self, 
        question: str, 
        expected_answer: str, 
        chunks: List[Dict[str, Any]]
    ) -> List[GroundTruthChunk]:
        """
        Create ground truth annotations for chunks relative to a question.
        
        Args:
            question: The question being asked
            expected_answer: Expected answer for the question
            chunks: List of chunk dictionaries with 'id', 'content', and metadata
            
        Returns:
            List of GroundTruthChunk objects with relevance annotations
        """
        
        if not chunks:
            return []
        
        annotated_chunks = []
        
        # Extract answer keywords for keyword-based matching
        answer_keywords = self._extract_answer_keywords(expected_answer)
        
        # Get embeddings for semantic similarity (if model available)
        if self.similarity_model:
            answer_embedding = self.similarity_model.encode([expected_answer])
            chunk_contents = [chunk.get('content', '') for chunk in chunks]
            chunk_embeddings = self.similarity_model.encode(chunk_contents)
            similarities = util.cos_sim(answer_embedding, chunk_embeddings)[0]
        else:
            similarities = [0.0] * len(chunks)
        
        # Annotate each chunk
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get('content', '')
            chunk_id = chunk.get('id', f'chunk_{i}')
            
            # Calculate semantic similarity score
            semantic_sim = float(similarities[i]) if self.similarity_model else 0.0
            
            # Calculate keyword overlap score
            keyword_score = self._calculate_keyword_overlap(chunk_content, answer_keywords)
            
            # Combine scores with weighting
            if self.similarity_model:
                relevance_score = (1 - self.keyword_weight) * semantic_sim + self.keyword_weight * keyword_score
            else:
                relevance_score = keyword_score
            
            # Determine binary relevance
            is_relevant = relevance_score > self.similarity_threshold
            
            # Check for answer keywords
            contains_keywords = any(keyword.lower() in chunk_content.lower() 
                                  for keyword in answer_keywords)
            
            annotated_chunk = GroundTruthChunk(
                chunk_id=chunk_id,
                content=chunk_content[:500] + "..." if len(chunk_content) > 500 else chunk_content,
                relevance_score=relevance_score,
                is_relevant=is_relevant,
                semantic_similarity=semantic_sim,
                contains_answer_keywords=contains_keywords
            )
            
            annotated_chunks.append(annotated_chunk)
        
        # Sort by relevance score (highest first)
        annotated_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return annotated_chunks
    
    def _extract_answer_keywords(self, answer: str) -> List[str]:
        """Extract key terms from expected answer for keyword matching."""
        
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 
            'will', 'just', 'don', 'should', 'now', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could'
        }
        
        # Extract words, filter stop words, and focus on meaningful terms
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', answer.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Also extract quoted phrases and named entities (capitalized sequences)
        quoted_phrases = re.findall(r'"([^"]+)"', answer)
        named_entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', answer)
        
        # Add numbers and years
        numbers = re.findall(r'\b\d{4}\b|\b\d+\.\d+\b|\b\d+%?\b', answer)
        
        # Combine all extracted terms
        all_keywords = keywords + [phrase.lower() for phrase in quoted_phrases] + \
                      [entity.lower() for entity in named_entities] + numbers
        
        # Remove duplicates and return top terms
        return list(set(all_keywords))[:15]  # Limit to top 15 keywords
    
    def _calculate_keyword_overlap(self, chunk_content: str, answer_keywords: List[str]) -> float:
        """Calculate overlap between chunk content and answer keywords."""
        
        if not answer_keywords:
            return 0.0
        
        chunk_lower = chunk_content.lower()
        matches = sum(1 for keyword in answer_keywords if keyword in chunk_lower)
        
        return matches / len(answer_keywords)


class RetrievalQualityEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.
    
    Calculates Recall@K, MRR, NDCG, and other metrics to assess
    how well different retrieval approaches find relevant content.
    """
    
    def __init__(self, annotator: GroundTruthAnnotator):
        self.annotator = annotator
    
    def evaluate_retrieval(
        self,
        question: str,
        expected_answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        approach: str,
        question_id: str
    ) -> RetrievalQualityResult:
        """
        Evaluate retrieval quality for a single question.
        
        Args:
            question: The question being asked
            expected_answer: Expected answer
            retrieved_chunks: Chunks returned by retrieval system
            approach: Name of retrieval approach being evaluated
            question_id: Unique identifier for question
            
        Returns:
            RetrievalQualityResult with computed metrics
        """
        
        # Create ground truth annotations
        ground_truth_chunks = self.annotator.annotate_chunks_for_question(
            question, expected_answer, retrieved_chunks
        )
        
        # Extract relevant information
        relevant_chunk_ids = [chunk.chunk_id for chunk in ground_truth_chunks if chunk.is_relevant]
        retrieved_chunk_ids = [chunk.get('id', f'chunk_{i}') for i, chunk in enumerate(retrieved_chunks)]
        relevance_scores = [chunk.relevance_score for chunk in ground_truth_chunks]
        
        # Calculate recall metrics
        recall_at_1 = self._calculate_recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, 1)
        recall_at_3 = self._calculate_recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, 3)
        recall_at_5 = self._calculate_recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, 5)
        recall_at_10 = self._calculate_recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, 10)
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(relevant_chunk_ids, retrieved_chunk_ids)
        
        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        ndcg_at_3 = self._calculate_ndcg(relevance_scores, 3)
        ndcg_at_5 = self._calculate_ndcg(relevance_scores, 5)
        ndcg_at_10 = self._calculate_ndcg(relevance_scores, 10)
        
        # Calculate precision
        precision_at_5 = self._calculate_precision_at_k(relevant_chunk_ids, retrieved_chunk_ids, 5)
        
        return RetrievalQualityResult(
            question_id=question_id,
            approach=approach,
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg_at_3=ndcg_at_3,
            ndcg_at_5=ndcg_at_5,
            ndcg_at_10=ndcg_at_10,
            total_retrieved=len(retrieved_chunk_ids),
            relevant_retrieved=len(set(relevant_chunk_ids) & set(retrieved_chunk_ids)),
            total_relevant=len(relevant_chunk_ids),
            precision_at_5=precision_at_5,
            ground_truth_chunks=relevant_chunk_ids,
            retrieved_chunks=retrieved_chunk_ids,
            relevance_scores=relevance_scores
        )
    
    def _calculate_recall_at_k(self, relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Recall@K metric."""
        
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = len(retrieved_at_k & relevant_set)
        return intersection / len(relevant_set)
    
    def _calculate_precision_at_k(self, relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Precision@K metric."""
        
        if not retrieved_ids or k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = len(retrieved_at_k & relevant_set)
        return intersection / min(k, len(retrieved_ids))
    
    def _calculate_mrr(self, relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        
        if not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg(self, relevance_scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        
        if not relevance_scores or k == 0:
            return 0.0
        
        # Calculate DCG@k for retrieved ranking
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            dcg += relevance_scores[i] / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@k (ideal ranking)
        sorted_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(sorted_scores))):
            idcg += sorted_scores[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0


class RetrievalQualityAnalyzer:
    """
    Analyzes retrieval quality results across experiments.
    
    Provides aggregate statistics and identifies patterns in
    retrieval effectiveness across different approaches.
    """
    
    def __init__(self):
        self.results: List[RetrievalQualityResult] = []
    
    def add_result(self, result: RetrievalQualityResult):
        """Add a retrieval quality result to the analysis."""
        self.results.append(result)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary of retrieval quality across approaches."""
        
        if not self.results:
            return {"error": "No retrieval quality results available"}
        
        # Group results by approach
        approach_results = {}
        for result in self.results:
            if result.approach not in approach_results:
                approach_results[result.approach] = []
            approach_results[result.approach].append(result)
        
        # Calculate aggregate metrics per approach
        approach_metrics = {}
        for approach, results in approach_results.items():
            metrics = self._calculate_aggregate_metrics(results)
            approach_metrics[approach] = metrics
        
        # Generate comparison table
        comparison_table = self._create_comparison_table(approach_metrics)
        
        # Identify best performing approach per metric
        best_approaches = self._identify_best_approaches(approach_metrics)
        
        return {
            "total_questions_evaluated": len(self.results),
            "approaches_compared": list(approach_metrics.keys()),
            "approach_metrics": approach_metrics,
            "comparison_table": comparison_table,
            "best_approaches": best_approaches,
            "detailed_results": [asdict(result) for result in self.results]
        }
    
    def _calculate_aggregate_metrics(self, results: List[RetrievalQualityResult]) -> Dict[str, float]:
        """Calculate average metrics across multiple results."""
        
        if not results:
            return {}
        
        return {
            "avg_recall_at_1": np.mean([r.recall_at_1 for r in results]),
            "avg_recall_at_3": np.mean([r.recall_at_3 for r in results]),
            "avg_recall_at_5": np.mean([r.recall_at_5 for r in results]),
            "avg_recall_at_10": np.mean([r.recall_at_10 for r in results]),
            "avg_mrr": np.mean([r.mrr for r in results]),
            "avg_ndcg_at_3": np.mean([r.ndcg_at_3 for r in results]),
            "avg_ndcg_at_5": np.mean([r.ndcg_at_5 for r in results]),
            "avg_ndcg_at_10": np.mean([r.ndcg_at_10 for r in results]),
            "avg_precision_at_5": np.mean([r.precision_at_5 for r in results]),
            "questions_evaluated": len(results)
        }
    
    def _create_comparison_table(self, approach_metrics: Dict[str, Dict[str, float]]) -> Table:
        """Create Rich table comparing approaches across metrics."""
        
        table = Table(title="Retrieval Quality Comparison")
        
        # Add columns
        table.add_column("Approach", style="cyan")
        table.add_column("Recall@1", justify="right")
        table.add_column("Recall@5", justify="right") 
        table.add_column("MRR", justify="right")
        table.add_column("NDCG@5", justify="right")
        table.add_column("Precision@5", justify="right")
        table.add_column("Questions", justify="right")
        
        # Add rows
        for approach, metrics in approach_metrics.items():
            table.add_row(
                approach,
                f"{metrics.get('avg_recall_at_1', 0):.3f}",
                f"{metrics.get('avg_recall_at_5', 0):.3f}",
                f"{metrics.get('avg_mrr', 0):.3f}",
                f"{metrics.get('avg_ndcg_at_5', 0):.3f}",
                f"{metrics.get('avg_precision_at_5', 0):.3f}",
                str(metrics.get('questions_evaluated', 0))
            )
        
        return table
    
    def _identify_best_approaches(self, approach_metrics: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Identify the best performing approach for each metric."""
        
        metrics_to_check = ['avg_recall_at_5', 'avg_mrr', 'avg_ndcg_at_5', 'avg_precision_at_5']
        best_approaches = {}
        
        for metric in metrics_to_check:
            best_approach = max(
                approach_metrics.keys(),
                key=lambda approach: approach_metrics[approach].get(metric, 0)
            )
            best_approaches[metric] = best_approach
        
        return best_approaches
    
    def display_results_summary(self):
        """Display a summary of retrieval quality results using Rich."""
        
        summary = self.generate_summary_report()
        
        if "error" in summary:
            console.print(f"[red]{summary['error']}[/red]")
            return
        
        # Display main summary panel
        summary_panel = Panel.fit(
            f"[bold green]📊 Retrieval Quality Summary[/bold green]\n"
            f"[dim]Questions Evaluated:[/dim] [cyan]{summary['total_questions_evaluated']}[/cyan]\n"
            f"[dim]Approaches Compared:[/dim] [yellow]{len(summary['approaches_compared'])}[/yellow]\n"
            f"[dim]Best Overall (NDCG@5):[/dim] [magenta]{summary['best_approaches'].get('avg_ndcg_at_5', 'TBD')}[/magenta]",
            title="🎯 Retrieval Quality Analysis",
            border_style="green"
        )
        console.print(summary_panel)
        
        # Display comparison table
        console.print(summary['comparison_table'])
        
        # Display best approaches
        best_table = Table(title="Best Performing Approaches by Metric")
        best_table.add_column("Metric", style="cyan")
        best_table.add_column("Best Approach", style="green")
        
        for metric, approach in summary['best_approaches'].items():
            metric_display = metric.replace('avg_', '').replace('_', '@').title()
            best_table.add_row(metric_display, approach)
        
        console.print(best_table)


# Export key classes for use in main experiment
__all__ = [
    'GroundTruthAnnotator',
    'RetrievalQualityEvaluator', 
    'RetrievalQualityAnalyzer',
    'RetrievalQualityResult',
    'GroundTruthChunk'
]