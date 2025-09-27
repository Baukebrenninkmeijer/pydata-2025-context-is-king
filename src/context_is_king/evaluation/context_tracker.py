"""
Context Tracker for RAG Experiments

This module provides comprehensive context size tracking and analysis
for comparing different RAG approaches and full context retrieval.
"""

import tiktoken
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class ContextMeasurement:
    """Measurement of context composition and size."""
    approach: str
    total_tokens: int
    query_tokens: int
    retrieved_context_tokens: int
    system_prompt_tokens: int
    context_composition: Dict[str, int]
    efficiency_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class ContextTracker:
    """Tracks and analyzes context size and composition across different approaches."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize context tracker.
        
        Args:
            encoding_name: tiktoken encoding to use (cl100k_base for GPT-4, etc.)
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            console.print(f"[yellow]Failed to load encoding {encoding_name}, using default: {e}[/yellow]")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.encoding_name = encoding_name
        console.print(f"[green]ContextTracker initialized with encoding: {encoding_name}[/green]")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured encoding.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            console.print(f"[yellow]Token counting error: {e}[/yellow]")
            # Fallback to rough estimation
            return len(text.split()) * 1.3  # Rough approximation
    
    def measure_full_context_approach(self,
                                    query: str,
                                    full_document_context: str,
                                    system_prompt: str = "",
                                    **kwargs) -> ContextMeasurement:
        """
        Measure context for full document context approach.
        
        Args:
            query: User query
            full_document_context: Complete document context
            system_prompt: System prompt if any
            **kwargs: Additional metadata
            
        Returns:
            ContextMeasurement for full context approach
        """
        query_tokens = self.count_tokens(query)
        context_tokens = self.count_tokens(full_document_context)
        system_tokens = self.count_tokens(system_prompt)
        total_tokens = query_tokens + context_tokens + system_tokens
        
        # Context composition breakdown
        composition = {
            "query": query_tokens,
            "document_context": context_tokens,
            "system_prompt": system_tokens,
            "overhead": 0  # No retrieval overhead for full context
        }
        
        # Efficiency metrics
        efficiency = {
            "context_utilization_ratio": 1.0,  # Full context is 100% utilized by definition
            "retrieval_overhead_ratio": 0.0,   # No retrieval overhead
            "context_density": context_tokens / total_tokens if total_tokens > 0 else 0,
            "query_to_context_ratio": query_tokens / context_tokens if context_tokens > 0 else 0
        }
        
        return ContextMeasurement(
            approach="full_context",
            total_tokens=total_tokens,
            query_tokens=query_tokens,
            retrieved_context_tokens=context_tokens,
            system_prompt_tokens=system_tokens,
            context_composition=composition,
            efficiency_metrics=efficiency,
            metadata={
                "document_length_chars": len(full_document_context),
                "query_length_chars": len(query),
                **kwargs
            }
        )
    
    def measure_rag_approach(self,
                           query: str,
                           retrieved_chunks: List[str],
                           system_prompt: str = "",
                           approach_name: str = "rag",
                           retrieval_metadata: Optional[Dict[str, Any]] = None,
                           **kwargs) -> ContextMeasurement:
        """
        Measure context for RAG-based approaches.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved text chunks
            system_prompt: System prompt if any
            approach_name: Name of the RAG approach (e.g., "rag_with_rerank")
            retrieval_metadata: Additional retrieval information
            **kwargs: Additional metadata
            
        Returns:
            ContextMeasurement for RAG approach
        """
        query_tokens = self.count_tokens(query)
        system_tokens = self.count_tokens(system_prompt)
        
        # Count tokens in each retrieved chunk
        chunk_tokens = [self.count_tokens(chunk) for chunk in retrieved_chunks]
        total_context_tokens = sum(chunk_tokens)
        total_tokens = query_tokens + total_context_tokens + system_tokens
        
        # Context composition breakdown
        composition = {
            "query": query_tokens,
            "retrieved_context": total_context_tokens,
            "system_prompt": system_tokens,
            "chunks_count": len(retrieved_chunks)
        }
        
        # Add per-chunk breakdown
        for i, chunk_token_count in enumerate(chunk_tokens[:5]):  # Top 5 chunks
            composition[f"chunk_{i+1}"] = chunk_token_count
        
        # Efficiency metrics
        retrieval_meta = retrieval_metadata or {}
        
        # Estimate context utilization based on chunk relevance scores
        avg_relevance = 0.5  # Default if no scores available
        if "relevance_scores" in retrieval_meta:
            scores = retrieval_meta["relevance_scores"]
            avg_relevance = sum(scores) / len(scores) if scores else 0.5
        
        efficiency = {
            "context_utilization_ratio": avg_relevance,
            "retrieval_overhead_ratio": len(retrieved_chunks) / 100.0,  # Normalized overhead
            "context_density": total_context_tokens / total_tokens if total_tokens > 0 else 0,
            "query_to_context_ratio": query_tokens / total_context_tokens if total_context_tokens > 0 else 0,
            "avg_chunk_size": total_context_tokens / len(retrieved_chunks) if retrieved_chunks else 0,
            "chunk_size_variance": self._calculate_variance(chunk_tokens) if len(chunk_tokens) > 1 else 0
        }
        
        metadata = {
            "chunks_count": len(retrieved_chunks),
            "total_retrieved_chars": sum(len(chunk) for chunk in retrieved_chunks),
            "query_length_chars": len(query),
            "retrieval_metadata": retrieval_meta,
            **kwargs
        }
        
        return ContextMeasurement(
            approach=approach_name,
            total_tokens=total_tokens,
            query_tokens=query_tokens,
            retrieved_context_tokens=total_context_tokens,
            system_prompt_tokens=system_tokens,
            context_composition=composition,
            efficiency_metrics=efficiency,
            metadata=metadata
        )
    
    def compare_measurements(self,
                           measurements: List[ContextMeasurement]) -> Dict[str, Any]:
        """
        Compare multiple context measurements.
        
        Args:
            measurements: List of ContextMeasurement objects to compare
            
        Returns:
            Comparative analysis of measurements
        """
        if not measurements:
            return {"error": "No measurements to compare"}
        
        # Group measurements by approach
        by_approach = {}
        for measurement in measurements:
            approach = measurement.approach
            if approach not in by_approach:
                by_approach[approach] = []
            by_approach[approach].append(measurement)
        
        # Calculate statistics for each approach
        approach_stats = {}
        for approach, measures in by_approach.items():
            total_tokens = [m.total_tokens for m in measures]
            context_tokens = [m.retrieved_context_tokens for m in measures]
            utilization_ratios = [m.efficiency_metrics.get("context_utilization_ratio", 0) for m in measures]
            
            approach_stats[approach] = {
                "count": len(measures),
                "avg_total_tokens": sum(total_tokens) / len(total_tokens),
                "avg_context_tokens": sum(context_tokens) / len(context_tokens),
                "avg_utilization_ratio": sum(utilization_ratios) / len(utilization_ratios),
                "min_total_tokens": min(total_tokens),
                "max_total_tokens": max(total_tokens),
                "token_efficiency": sum(context_tokens) / sum(total_tokens) if sum(total_tokens) > 0 else 0
            }
        
        # Overall comparison
        all_measurements = measurements
        total_tokens_all = [m.total_tokens for m in all_measurements]
        
        comparison = {
            "approaches_compared": list(approach_stats.keys()),
            "total_measurements": len(measurements),
            "approach_statistics": approach_stats,
            "overall_stats": {
                "avg_total_tokens": sum(total_tokens_all) / len(total_tokens_all),
                "min_total_tokens": min(total_tokens_all),
                "max_total_tokens": max(total_tokens_all),
                "token_range": max(total_tokens_all) - min(total_tokens_all)
            }
        }
        
        # Find most/least efficient approaches
        if approach_stats:
            most_efficient = max(approach_stats.items(), key=lambda x: x[1]["token_efficiency"])
            least_efficient = min(approach_stats.items(), key=lambda x: x[1]["token_efficiency"])
            
            comparison["efficiency_ranking"] = {
                "most_efficient": {"approach": most_efficient[0], **most_efficient[1]},
                "least_efficient": {"approach": least_efficient[0], **least_efficient[1]}
            }
        
        return comparison
    
    def analyze_context_scaling(self, 
                              measurements: List[ContextMeasurement],
                              group_by: str = "approach") -> Dict[str, Any]:
        """
        Analyze how context size scales across different conditions.
        
        Args:
            measurements: List of measurements to analyze
            group_by: Field to group by ("approach", "query_type", etc.)
            
        Returns:
            Context scaling analysis
        """
        if not measurements:
            return {"error": "No measurements to analyze"}
        
        # Group measurements
        groups = {}
        for measurement in measurements:
            if group_by == "approach":
                key = measurement.approach
            else:
                key = measurement.metadata.get(group_by, "unknown")
            
            if key not in groups:
                groups[key] = []
            groups[key].append(measurement)
        
        # Analyze scaling for each group
        scaling_analysis = {}
        for group_name, group_measurements in groups.items():
            tokens = [m.total_tokens for m in group_measurements]
            context_ratios = [m.efficiency_metrics.get("context_density", 0) for m in group_measurements]
            
            if len(tokens) > 1:
                scaling_analysis[group_name] = {
                    "measurement_count": len(tokens),
                    "token_stats": {
                        "mean": sum(tokens) / len(tokens),
                        "min": min(tokens),
                        "max": max(tokens),
                        "variance": self._calculate_variance(tokens)
                    },
                    "context_density_stats": {
                        "mean": sum(context_ratios) / len(context_ratios),
                        "min": min(context_ratios),
                        "max": max(context_ratios)
                    },
                    "scaling_trend": self._analyze_trend(tokens)
                }
            else:
                scaling_analysis[group_name] = {
                    "measurement_count": len(tokens),
                    "single_measurement": tokens[0] if tokens else 0
                }
        
        return {
            "group_by": group_by,
            "groups_analyzed": list(scaling_analysis.keys()),
            "scaling_analysis": scaling_analysis
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance
    
    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in values (increasing, decreasing, stable)."""
        if len(values) < 3:
            return "insufficient_data"
        
        increases = 0
        decreases = 0
        
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                increases += 1
            elif values[i] < values[i-1]:
                decreases += 1
        
        if increases > decreases * 1.5:
            return "increasing"
        elif decreases > increases * 1.5:
            return "decreasing"
        else:
            return "stable"
    
    def get_tracker_stats(self) -> Dict[str, Any]:
        """Get context tracker statistics and configuration."""
        return {
            "encoding_name": self.encoding_name,
            "encoding_vocab_size": self.encoding.n_vocab if hasattr(self.encoding, 'n_vocab') else "unknown"
        }