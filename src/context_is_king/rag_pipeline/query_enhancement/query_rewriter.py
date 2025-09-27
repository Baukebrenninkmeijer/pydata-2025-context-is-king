"""
Query Rewriter for Enhanced RAG Retrieval

This module provides LLM-based query expansion and reformulation to improve
retrieval quality in RAG pipelines. It generates alternative formulations
of user queries to capture different aspects of information need.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rich.console import Console

from ...models.interface import ModelInterface
from ..types import PipelineConfig
from loguru import logger

console = Console()


@dataclass
class QueryRewriteResult:
    """Result from query rewriting operation."""

    original_query: str
    rewritten_query: str
    rewrite_time_ms: float
    success: bool
    metadata: Dict[str, Any]


class QueryRewriter:
    """LLM-based query rewriter for enhanced retrieval."""

    def __init__(self, pipeline_config: PipelineConfig, model_interface: ModelInterface | None = None):
        self.model_interface = model_interface or ModelInterface()
        self.model = pipeline_config.small_llm

        self.rewrite_prompt_template = """Given the following question, generate an alternative search query that might find relevant information using different keywords or phrasing.

The alternative query should:
1. Use synonyms or related terms
2. Rephrase the question structure
3. Focus on key concepts from a different angle
4. Maintain the same information need

Original question: {question}

Generate only the alternative search query without additional text:"""

        console.print("[green]QueryRewriter initialized[/green]")

    async def rewrite_query(self, query: str, verbose: bool = False, **kwargs) -> QueryRewriteResult:  # noqa: ANN003
        """
        Generate an alternative formulation of the input query.

        Args:
            query: Original user query
            **kwargs: Additional parameters for generation

        Returns:
            QueryRewriteResult with original and rewritten queries
        """
        start_time = time.time()

        try:
            # Prepare the rewriting prompt
            prompt = self.rewrite_prompt_template.format(question=query)

            # Generate rewritten query using the model interface
            gen_result = await self.model_interface.query_model_async(
                model_name=self.model,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 100),
                temperature=0.0,
            )

            if not gen_result.success or not gen_result.response:
                logger.warning(f"Query rewriting failed for: {query[:50]}...")
                # console.print(f"[yellow]Query rewriting failed for: {query[:50]}...[/yellow]")
                return QueryRewriteResult(
                    original_query=query,
                    rewritten_query=query,  # Fallback to original
                    rewrite_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    metadata={"error": "Empty response from LLM"},
                )

            rewritten_query = gen_result.response.strip()

            # Basic validation - ensure we got a reasonable query back
            if len(rewritten_query) < 5 or len(rewritten_query) > 500:
                console.print(f"[yellow]Invalid rewritten query length: {len(rewritten_query)}[/yellow]")
                rewritten_query = query  # Fallback to original
                success = False
            else:
                success = True

            rewrite_time = (time.time() - start_time) * 1000

            result = QueryRewriteResult(
                original_query=query,
                rewritten_query=rewritten_query,
                rewrite_time_ms=rewrite_time,
                success=success,
                metadata={
                    "model_used": kwargs.get("model", "gpt-4-turbo"),
                    "token_usage": getattr(gen_result, "token_usage", {}),
                    "latency_ms": getattr(gen_result, "latency_ms", None),
                    "generation_metadata": getattr(gen_result, "metadata", {}),
                },
            )
            if verbose:
                console.print(f"[blue]Query rewritten in {rewrite_time:.1f}ms[/blue]")
            return result  # noqa: TRY300

        except Exception as e:
            console.print(f"[red]Query rewriting error: {e}[/red]")
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,  # Fallback to original
                rewrite_time_ms=(time.time() - start_time) * 1000,
                success=False,
                metadata={"error": str(e)},
            )

    def batch_rewrite_queries(self, queries: List[str], **kwargs) -> List[QueryRewriteResult]:
        """
        Rewrite multiple queries in batch.

        Args:
            queries: List of queries to rewrite
            **kwargs: Additional parameters for generation

        Returns:
            List of QueryRewriteResult objects
        """
        console.print(f"[blue]Batch rewriting {len(queries)} queries[/blue]")

        results = []
        for i, query in enumerate(queries):
            console.print(f"[dim]Rewriting query {i + 1}/{len(queries)}[/dim]")
            result = self.rewrite_query(query, **kwargs)
            results.append(result)

        success_count = sum(1 for r in results if r.success)
        console.print(f"[green]Batch rewriting complete: {success_count}/{len(queries)} successful[/green]")

        return results

    def analyze_rewrite_quality(self, result: QueryRewriteResult) -> Dict[str, Any]:
        """
        Analyze the quality of a query rewrite.

        Args:
            result: QueryRewriteResult to analyze

        Returns:
            Quality analysis metrics
        """
        if not result.success:
            return {"error": "Rewrite failed", "quality_score": 0.0}

        original = result.original_query.lower()
        rewritten = result.rewritten_query.lower()

        # Calculate basic similarity metrics
        original_words = set(original.split())
        rewritten_words = set(rewritten.split())

        # Word overlap (lower is better for diversity)
        word_overlap = len(original_words.intersection(rewritten_words)) / len(original_words.union(rewritten_words))

        # Length similarity
        length_ratio = min(len(rewritten), len(original)) / max(len(rewritten), len(original))

        # Diversity score (higher is better)
        diversity_score = 1.0 - word_overlap

        # Quality heuristics
        has_question_words = any(word in rewritten for word in ["what", "how", "when", "where", "why", "who"])
        is_reasonable_length = 10 <= len(rewritten) <= 200
        is_different = rewritten != original

        quality_indicators = {
            "word_overlap": word_overlap,
            "length_ratio": length_ratio,
            "diversity_score": diversity_score,
            "has_question_words": has_question_words,
            "is_reasonable_length": is_reasonable_length,
            "is_different": is_different,
        }

        # Simple quality score
        quality_score = (
            diversity_score * 0.4
            + length_ratio * 0.2
            + (1.0 if has_question_words else 0.0) * 0.2
            + (1.0 if is_reasonable_length else 0.0) * 0.1
            + (1.0 if is_different else 0.0) * 0.1
        )

        return {
            "quality_score": quality_score,
            "indicators": quality_indicators,
            "original_query": result.original_query,
            "rewritten_query": result.rewritten_query,
        }

    def get_rewriter_stats(self) -> Dict[str, Any]:
        """Get query rewriter statistics and configuration."""
        return {
            "model_interface": "OpenAI SDK via ORQ proxy",
            "prompt_template_length": len(self.rewrite_prompt_template),
            "available_models": list(self.model_interface.MODELS.keys())[:5],  # Show first 5
        }
