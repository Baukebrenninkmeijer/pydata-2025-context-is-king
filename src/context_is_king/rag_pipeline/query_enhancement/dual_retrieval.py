"""
Dual Retrieval Engine for Enhanced RAG

This module combines retrieval results from original and rewritten queries,
providing improved recall and relevance through query diversity.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from chromadb import Collection
from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from context_is_king.models import ModelInterface
from context_is_king.rag_pipeline.retrieval.engine import RetrievalEngine

from ..types import PipelineConfig, RetrievalResult
from .query_rewriter import QueryRewriter, QueryRewriteResult

console = Console()


class HypotheticalAnswerer:
    """LLM-based hypothetical answer generator for query enhancement."""

    def __init__(self, pipeline_config: PipelineConfig, model_interface: ModelInterface | None = None):
        self.model_interface = model_interface or ModelInterface()
        self.model = pipeline_config.small_llm

        self.hypothetical_prompt_template = """You are a professional question answerer, with the goal of giving a probably answer to questions. Given the following question, generate a concise hypothetical answer that captures the key information need. Try to base your answer on common knowledge, but it does not need to be accurate."""

    async def generate_hypothetical_answer(
        self,
        query: str,
        verbose: bool = False,
        use_async: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> str | None:
        """
        Generate a hypothetical answer to the input query.

        Args:
            query: The user query to answer
            verbose: Whether to print detailed logs
            **kwargs: Additional parameters for LLM generation

        Returns:
            Hypothetical answer string or None if generation failed
        """
        prompt = self.hypothetical_prompt_template + f"\n\nQuestion: {query}\nHypothetical Answer:"

        if verbose:
            console.print(f"[blue]Generating hypothetical answer for: {query[:50]}...[/blue]")

        gen_result = await self.model_interface.query_model_async(
            model_name=self.model,
            prompt=prompt,
            max_tokens=1000,
            temperature=0.0,
        )
        # print(f'{gen_result=}')
        if not gen_result.success or not gen_result.response:
            logger.warning("Hypothetical answer generation failed")
            return None

        hypothetical_answer = gen_result.response.strip().split("\n")[0]
        if verbose:
            console.print(f"[green]Hypothetical answer generated: {hypothetical_answer}[/green]")

        return hypothetical_answer


class DualRetrievalResult(BaseModel):
    """Result from dual retrieval operation."""

    original_query: str
    rewritten_query: str
    original_results: List[RetrievalResult]
    rewritten_results: List[RetrievalResult]
    combined_results: List[RetrievalResult]
    fusion_metadata: Dict[str, Any]
    total_time_ms: float


class DualRetrieval:
    """Enhanced retrieval combining original and rewritten query results."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        retrieval_engine: RetrievalEngine,
        query_rewriter: QueryRewriter | None = None,
        hypothetical_answerer: HypotheticalAnswerer | None = None,
    ):
        self.retrieval_engine = retrieval_engine
        self.query_rewriter = query_rewriter or QueryRewriter(pipeline_config)
        self.hypothetical_answerer = hypothetical_answerer or HypotheticalAnswerer(pipeline_config)

        console.print("[green]DualRetrieval initialized[/green]")

    async def retrieve_dual(
        self,
        query: str,
        collection: Collection,
        k: int = 20,
        fusion_method: str = "rrf",  # "rrf", "concat", "interleave"
        dimensions: int = 768,
        *,
        verbose: bool = False,
        retries: int = 3,
        **kwargs,  # noqa: ANN003
    ) -> DualRetrievalResult:
        """
        Perform dual retrieval with original and rewritten queries.

        Args:
            query: Original user query
            collection: ChromaDB collection to search
            k: Total number of results to return after fusion
            fusion_method: Method to combine results ("rrf", "concat", "interleave")
            **kwargs: Additional parameters for retrieval and rewriting

        Returns:
            DualRetrievalResult with combined and individual results
        """
        start_time = time.time()
        if verbose:
            console.print(f"[blue]Starting dual retrieval for: {query[:50]}...[/blue]")

        # Step 1: Rewrite the query
        for _ in range(retries):
            try:
                rewrite_result = await self.query_rewriter.rewrite_query(query, **kwargs)
                hypothetical_answer = await self.hypothetical_answerer.generate_hypothetical_answer(query)

                if not rewrite_result.success:
                    logger.warning("Query rewriting failed, using original query only.")
                    # console.print("[yellow]Query rewriting failed, using original query only[/yellow]")
                    original_results = await self.retrieval_engine.retrieve_chunks(
                        query=query, collection=collection, k=k, dimensions=dimensions
                    )

                    return DualRetrievalResult(
                        original_query=query,
                        rewritten_query=query,
                        original_results=original_results,
                        rewritten_results=[],
                        combined_results=original_results,
                        fusion_metadata={"error": "Rewriting failed", "fallback_used": True},
                        total_time_ms=(time.time() - start_time) * 1000,
                    )

                # Step 2: Retrieve with both queries in parallel
                # Retrieve more results initially to allow for better fusion
                # retrieve_k = min(k * 2, 50)  # Get more results for better fusion
                if verbose:
                    console.print(f"[dim]Retrieving {k} results each for original and rewritten queries[/dim]")

                original_results = await self.retrieval_engine.retrieve_chunks(
                    query=query, collection=collection, k=k, dimensions=dimensions
                )
                rewritten_results = await self.retrieval_engine.retrieve_chunks(
                    query=rewrite_result.rewritten_query, collection=collection, k=k, dimensions=dimensions
                )

                # Handle failed hypothetical answer generation
                if hypothetical_answer and isinstance(hypothetical_answer, str) and hypothetical_answer.strip():
                    hypothetical_results = await self.retrieval_engine.retrieve_chunks(
                        query=hypothetical_answer, collection=collection, k=k, dimensions=dimensions
                    )
                else:
                    if verbose:
                        console.print("[dim]Skipping hypothetical retrieval - invalid answer generated[/dim]")
                    hypothetical_results = []
                # Step 3: Fuse the results
                combined_results, fusion_metadata = self._fuse_results(
                    original_results, rewritten_results, fusion_method, target_k=k
                )

                # Only fuse hypothetical results if we have any
                if hypothetical_results:
                    combined_results, fusion_metadata = self._fuse_results(
                        combined_results, hypothetical_results, fusion_method, target_k=k
                    )

                total_time = (time.time() - start_time) * 1000

                result = DualRetrievalResult(
                    original_query=query,
                    rewritten_query=rewrite_result.rewritten_query,
                    original_results=original_results,
                    rewritten_results=rewritten_results,
                    combined_results=combined_results,
                    fusion_metadata=fusion_metadata,
                    total_time_ms=total_time,
                )
                if verbose:
                    console.print(
                        f"[green]Dual retrieval completed in {total_time:.1f}ms, {len(combined_results)} results[/green]"
                    )
                return result
            except Exception as e:
                logger.error(f"Dual retrieval attempt failed: {e}")

    def _fuse_results(
        self,
        original_results: list[RetrievalResult],
        rewritten_results: list[RetrievalResult],
        fusion_method: str,
        target_k: int,
    ) -> tuple[list[RetrievalResult], Dict[str, Any]]:
        """
        Fuse results from original and rewritten queries.

        Args:
            original_results: Results from original query
            rewritten_results: Results from rewritten query
            fusion_method: Fusion method to use
            target_k: Target number of results to return

        Returns:
            Tuple of (fused_results, fusion_metadata)
        """
        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(original_results, rewritten_results, target_k)
        elif fusion_method == "concat":
            return self._concatenate_fusion(original_results, rewritten_results, target_k)
        elif fusion_method == "interleave":
            return self._interleave_fusion(original_results, rewritten_results, target_k)
        else:
            console.print(f"[yellow]Unknown fusion method {fusion_method}, using RRF[/yellow]")
            return self._reciprocal_rank_fusion(original_results, rewritten_results, target_k)

    def _reciprocal_rank_fusion(
        self, original_results: List[RetrievalResult], rewritten_results: List[RetrievalResult], target_k: int
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) - combines rankings using reciprocal ranks.

        RRF Score = 1/(k + rank_in_original) + 1/(k + rank_in_rewritten)
        where k is typically 60.
        """
        k_constant = 60
        chunk_scores = defaultdict(lambda: {"score": 0.0, "result": None, "sources": []})

        # Process original results
        for i, result in enumerate(original_results):
            chunk_id = result.chunk.full_id
            rrf_score = 1.0 / (k_constant + i + 1)
            chunk_scores[chunk_id]["score"] += rrf_score
            chunk_scores[chunk_id]["result"] = result
            chunk_scores[chunk_id]["sources"].append(f"original_rank_{i + 1}")

        # Process rewritten results
        for i, result in enumerate(rewritten_results):
            chunk_id = result.chunk.full_id
            rrf_score = 1.0 / (k_constant + i + 1)
            chunk_scores[chunk_id]["score"] += rrf_score
            if chunk_scores[chunk_id]["result"] is None:
                chunk_scores[chunk_id]["result"] = result
            chunk_scores[chunk_id]["sources"].append(f"rewritten_rank_{i + 1}")

        # Sort by RRF score and create final results
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        fused_results = []
        for i, (chunk_id, data) in enumerate(sorted_chunks[:target_k]):
            result = data["result"]
            # Update result with RRF score and new rank
            fused_result = result.model_copy(
                update={
                    "rank": i + 1,
                    "similarity_score": data["score"],  # Use RRF score as similarity
                }
            )
            fused_results.append(fused_result)

        # Calculate fusion statistics
        original_ids = {r.chunk.full_id for r in original_results}
        rewritten_ids = {r.chunk.full_id for r in rewritten_results}
        fused_ids = {r.chunk.full_id for r in fused_results}

        fusion_metadata = {
            "method": "reciprocal_rank_fusion",
            "k_constant": k_constant,
            "original_count": len(original_results),
            "rewritten_count": len(rewritten_results),
            "fused_count": len(fused_results),
            "unique_chunks": len(chunk_scores),
            "overlap_count": len(original_ids.intersection(rewritten_ids)),
            "from_original_only": len(fused_ids.intersection(original_ids) - rewritten_ids),
            "from_rewritten_only": len(fused_ids.intersection(rewritten_ids) - original_ids),
            "from_both": len(fused_ids.intersection(original_ids.intersection(rewritten_ids))),
        }

        return fused_results, fusion_metadata

    def _concatenate_fusion(
        self, original_results: List[RetrievalResult], rewritten_results: List[RetrievalResult], target_k: int
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Simple concatenation fusion - original results first, then rewritten.
        Deduplicates by chunk ID.
        """
        seen_chunks: Set[str] = set()
        fused_results = []

        # Add original results first
        for result in original_results:
            if len(fused_results) >= target_k:
                break
            if result.chunk.full_id not in seen_chunks:
                fused_result = result.model_copy(update={"rank": len(fused_results) + 1})
                fused_results.append(fused_result)
                seen_chunks.add(result.chunk.full_id)

        # Add rewritten results
        for result in rewritten_results:
            if len(fused_results) >= target_k:
                break
            if result.chunk.full_id not in seen_chunks:
                fused_result = result.model_copy(update={"rank": len(fused_results) + 1})
                fused_results.append(fused_result)
                seen_chunks.add(result.chunk.full_id)

        fusion_metadata = {
            "method": "concatenate",
            "original_count": len(original_results),
            "rewritten_count": len(rewritten_results),
            "fused_count": len(fused_results),
            "duplicates_removed": (len(original_results) + len(rewritten_results)) - len(fused_results),
        }

        return fused_results, fusion_metadata

    def _interleave_fusion(
        self, original_results: List[RetrievalResult], rewritten_results: List[RetrievalResult], target_k: int
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Interleave fusion - alternate between original and rewritten results.
        Deduplicates by chunk ID.
        """
        seen_chunks: Set[str] = set()
        fused_results = []

        max_len = max(len(original_results), len(rewritten_results))

        for i in range(max_len):
            if len(fused_results) >= target_k:
                break

            # Try to add from original results
            if i < len(original_results):
                result = original_results[i]
                if result.chunk.full_id not in seen_chunks:
                    fused_result = result.model_copy(update={"rank": len(fused_results) + 1})
                    fused_results.append(fused_result)
                    seen_chunks.add(result.chunk.full_id)

            if len(fused_results) >= target_k:
                break

            # Try to add from rewritten results
            if i < len(rewritten_results):
                result = rewritten_results[i]
                if result.chunk.full_id not in seen_chunks:
                    fused_result = result.model_copy(update={"rank": len(fused_results) + 1})
                    fused_results.append(fused_result)
                    seen_chunks.add(result.chunk.full_id)

        fusion_metadata = {
            "method": "interleave",
            "original_count": len(original_results),
            "rewritten_count": len(rewritten_results),
            "fused_count": len(fused_results),
            "duplicates_removed": (len(original_results) + len(rewritten_results)) - len(fused_results),
        }

        return fused_results, fusion_metadata

    def analyze_retrieval_diversity(self, result: DualRetrievalResult) -> Dict[str, Any]:
        """
        Analyze the diversity and complementarity of dual retrieval results.

        Args:
            result: DualRetrievalResult to analyze

        Returns:
            Dictionary with diversity analysis metrics
        """
        original_ids = {r.chunk.full_id for r in result.original_results}
        rewritten_ids = {r.chunk.full_id for r in result.rewritten_results}
        fused_ids = {r.chunk.full_id for r in result.combined_results}

        # Calculate overlap and diversity metrics
        total_unique = len(original_ids.union(rewritten_ids))
        overlap = len(original_ids.intersection(rewritten_ids))

        diversity_metrics = {
            "original_count": len(original_ids),
            "rewritten_count": len(rewritten_ids),
            "total_unique_chunks": total_unique,
            "overlapping_chunks": overlap,
            "diversity_ratio": 1.0 - (overlap / total_unique) if total_unique > 0 else 0.0,
            "fused_count": len(fused_ids),
            "fusion_efficiency": len(fused_ids) / max(len(result.original_results), len(result.rewritten_results))
            if max(len(result.original_results), len(result.rewritten_results)) > 0
            else 0.0,
        }

        # Analyze score distributions
        if result.original_results:
            original_scores = [r.similarity_score for r in result.original_results]
            diversity_metrics["original_score_stats"] = {
                "mean": sum(original_scores) / len(original_scores),
                "max": max(original_scores),
                "min": min(original_scores),
            }

        if result.rewritten_results:
            rewritten_scores = [r.similarity_score for r in result.rewritten_results]
            diversity_metrics["rewritten_score_stats"] = {
                "mean": sum(rewritten_scores) / len(rewritten_scores),
                "max": max(rewritten_scores),
                "min": min(rewritten_scores),
            }

        diversity_metrics["fusion_metadata"] = result.fusion_metadata

        return diversity_metrics
