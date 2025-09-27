"""
Retrieval engine with semantic similarity search and performance metrics
"""

import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import asdict

from chromadb import Collection
import openai
from rich.console import Console

from context_is_king.evaluation.judge_evaluator import extract_retry_delay

from ..types import DocumentChunk, RetrievalResult, PipelineConfig
from ..embedding_storage.manager import EmbeddingStorageManager
from openai import OpenAI
import os
from loguru import logger
from retry import retry

console = Console()


class RetrievalEngine:
    """Handles semantic similarity search and retrieval metrics"""

    def __init__(self, config: PipelineConfig, embedding_manager: EmbeddingStorageManager | None = None):
        self.embedding_manager = embedding_manager
        self.openai_client = OpenAI(
            base_url=os.getenv("ORQ_BASE_URL"),
            api_key=os.getenv("ORQ_API_KEY"),
        )

        self.config = config

        console.print("[green]RetrievalEngine initialized[/green]")

    async def get_embedding_for_query(self, query: str, dimensions: int, retries: int = 3) -> list[float] | None:
        for i in range(retries):
            try:
                query_embedding = (
                    self.openai_client.embeddings.create(
                        input=query.strip(), dimensions=dimensions, model=os.environ["EMBEDDING_MODEL"]
                    )
                    .data[0]
                    .embedding
                )
                return query_embedding
            except openai.RateLimitError as e:  # noqa: PERF203
                sleep_duration = extract_retry_delay(error_message=str(e))
                logger.error(
                    f"Query embedding retrieval {i + 1} failed. Sleeping for {sleep_duration} seconds: {type(e)} - {e}"
                )
                await asyncio.sleep(sleep_duration + random.randint(1, 10))  # Exponential backoff
            except Exception as e:
                if "400" in str(e):
                    logger.error(f"Bad request error, not retrying: {e}")
                    return None
                sleep_duration = 2**i
                logger.error(f"Query embedding retrieval {i} failed. Sleeping for {sleep_duration} type={type(e)}: {e}")
                await asyncio.sleep(sleep_duration + random.randint(1, 10))

    async def retrieve_chunks(
        self,
        collection: Collection,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        k: int = -1,
        dimensions: int = 768,
        retries: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k chunks based on semantic similarity

        Args:
            query: Search query
            collection: ChromaDB collection to search
            k: Number of results to return (defaults to config.default_k)

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        if k == -1:
            k = self.config.default_k

        start_time = time.time()

        # Validate query input
        if (not query or not isinstance(query, str) or query.strip() == "") and query_embedding is None:
            logger.error(f"Invalid query provided: {repr(query)} (type: {type(query)})")
            return []

        for i in range(retries):
            try:
                # Generate query embedding
                if query_embedding is None and query is not None:
                    query_embedding = await self.get_embedding_for_query(
                        query=query, retries=retries, dimensions=dimensions
                    )

                # Query ChromaDB
                results = collection.query(
                    query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"]
                )

                # Convert to RetrievalResult objects
                retrieval_results = self._convert_to_retrieval_results(chroma_results=results, query=query)

                # Filter by similarity threshold if configured
                if self.config.similarity_threshold > 0:
                    retrieval_results = [
                        r for r in retrieval_results if r.similarity_score >= self.config.similarity_threshold
                    ]

                retrieval_time = time.time() - start_time

                if not getattr(self.config, "quiet_mode", False):
                    console.print(f"[blue]Retrieved {len(retrieval_results)} chunks in {retrieval_time:.3f}s[/blue]")
                return retrieval_results
            except openai.RateLimitError as e:  # noqa: PERF203
                sleep_duration = extract_retry_delay(error_message=str(e))
                logger.error(f"Query embedding {i + 1} failed. Sleeping for {sleep_duration} seconds: {type(e)} - {e}")
                await asyncio.sleep(sleep_duration + random.randint(1, 10))  # Exponential backoff
            except Exception as e:
                if "400" in str(e):
                    logger.error(f"Bad request error, not retrying: {e}")
                    return []
                sleep_duration = 2**i
                logger.error(f"llm call iteration {i} failed. Sleeping for {sleep_duration} type={type(e)}: {e}")
                await asyncio.sleep(sleep_duration + random.randint(1, 10))
        return []

    def retrieve_with_metadata_filter(
        self, query: str, collection: Collection, filters: Dict[str, Any], k: int = None
    ) -> List[RetrievalResult]:
        """
        Retrieve with metadata-based filtering

        Args:
            query: Search query
            collection: ChromaDB collection to search
            filters: Metadata filters (e.g., {"doc_type": "paul_graham"})
            k: Number of results to return

        Returns:
            Filtered retrieval results
        """
        if k is None:
            k = self.config.default_k

        try:
            query_embedding = self.embedding_manager.embed_query(query)

            # Build ChromaDB where clause
            where_clause = self._build_where_clause(filters)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            retrieval_results = self._convert_to_retrieval_results(results, query)

            if not getattr(self.config, "quiet_mode", False):
                console.print(f"[blue]Retrieved {len(retrieval_results)} filtered chunks[/blue]")
            return retrieval_results

        except Exception as e:
            console.print(f"[red]Filtered retrieval failed: {e}[/red]")
            return []

    def calculate_retrieval_metrics(
        self,
        retrieved_chunks: list[RetrievalResult],
        ground_truth_chunk_ids: list[str],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Calculate precision, recall, and other retrieval metrics

        Args:
            retrieved_chunks: Retrieved results from search
            ground_truth_chunk_ids: List of chunk IDs that should be retrieved
            k_values: Different k values to calculate metrics for

        Returns:
            Dictionary with retrieval metrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        metrics = {}
        # logger.debug(f"Ground truth chunk IDs: {ground_truth_chunk_ids}")
        if not ground_truth_chunk_ids:
            console.print("[yellow]No ground truth provided for metrics calculation[/yellow]")
            return {"warning": "no_ground_truth"}

        ground_truth_set = set(ground_truth_chunk_ids)
        retrieved_ids = [chunk.chunk.chunk_id for chunk in retrieved_chunks]
        # logger.debug(f"Ground retrieved IDS: {retrieved_ids}")

        # Calculate metrics for different k values
        for k in k_values:
            if k > len(retrieved_chunks):
                continue

            top_k_ids = set(retrieved_ids[:k])

            # Precision@k = relevant_retrieved / total_retrieved
            precision = len(top_k_ids.intersection(ground_truth_set)) / k if k > 0 else 0

            # Recall@k = relevant_retrieved / total_relevant
            recall = (
                len(top_k_ids.intersection(ground_truth_set)) / len(ground_truth_set)
                if len(ground_truth_set) > 0
                else 0
            )

            # F1@k
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics.update({f"precision@{k}": precision, f"recall@{k}": recall, f"f1@{k}": f1})

        # Mean Reciprocal Rank (MRR)
        mrr = self._calculate_mrr(retrieved_ids, ground_truth_set)
        metrics["mrr"] = mrr

        # Hit rate (whether any relevant document was retrieved)
        hit_rate = 1.0 if len(set(retrieved_ids).intersection(ground_truth_set)) > 0 else 0.0
        metrics["hit_rate"] = hit_rate

        # Average similarity score of retrieved chunks
        if retrieved_chunks:
            avg_similarity = sum(chunk.similarity_score for chunk in retrieved_chunks) / len(retrieved_chunks)
            metrics["avg_similarity"] = avg_similarity
            metrics["max_similarity"] = max(chunk.similarity_score for chunk in retrieved_chunks)
            metrics["min_similarity"] = min(chunk.similarity_score for chunk in retrieved_chunks)

        return metrics

    def analyze_retrieval_patterns(self, results: List[RetrievalResult], query: str) -> Dict[str, Any]:
        """
        Analyze retrieval patterns for debugging and optimization

        Args:
            results: Retrieval results to analyze
            query: Original query

        Returns:
            Analysis of retrieval patterns
        """
        if not results:
            return {"error": "no_results"}

        analysis = {
            "query": query,
            "total_results": len(results),
            "similarity_distribution": {
                "mean": sum(r.similarity_score for r in results) / len(results),
                "max": max(r.similarity_score for r in results),
                "min": min(r.similarity_score for r in results),
                "std": self._calculate_std([r.similarity_score for r in results]),
            },
            "document_distribution": self._analyze_document_distribution(results),
            "content_length_stats": self._analyze_content_lengths(results),
            "top_3_chunks": [
                {
                    "rank": r.rank,
                    "similarity": r.similarity_score,
                    "doc_id": r.chunk.doc_id,
                    "chunk_id": r.chunk.chunk_id,
                    "content_preview": r.chunk.content[:100] + "..." if len(r.chunk.content) > 100 else r.chunk.content,
                }
                for r in results[:3]
            ],
        }

        return analysis

    def benchmark_retrieval_speed(self, queries: List[str], collection: Collection, k: int = 10) -> Dict[str, float]:
        """
        Benchmark retrieval speed with multiple queries

        Args:
            queries: List of test queries
            collection: Collection to search
            k: Number of results per query

        Returns:
            Speed benchmarking results
        """
        console.print(f"[blue]Benchmarking retrieval speed with {len(queries)} queries[/blue]")

        times = []
        result_counts = []

        for query in queries:
            start_time = time.time()
            results = self.retrieve_chunks(query, collection, k)
            end_time = time.time()

            times.append(end_time - start_time)
            result_counts.append(len(results))

        benchmark_results = {
            "total_queries": len(queries),
            "total_time": sum(times),
            "avg_time_per_query": sum(times) / len(times) if times else 0,
            "min_time": min(times) if times else 0,
            "max_time": max(times) if times else 0,
            "avg_results_per_query": sum(result_counts) / len(result_counts) if result_counts else 0,
            "queries_per_second": len(queries) / sum(times) if sum(times) > 0 else 0,
        }

        console.print(
            f"[green]Benchmark complete: {benchmark_results['avg_time_per_query']:.3f}s avg, {benchmark_results['queries_per_second']:.1f} QPS[/green]"
        )

        return benchmark_results

    def _convert_to_retrieval_results(self, chroma_results: Dict, query: str) -> List[RetrievalResult]:
        """Convert ChromaDB results to RetrievalResult objects"""
        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return []
        # print(chroma_results[0])
        results = []
        for i, (chunk_id, document, metadata, distance) in enumerate(
            zip(
                chroma_results["ids"][0],
                chroma_results["documents"][0],
                chroma_results["metadatas"][0],
                chroma_results["distances"][0],
                strict=True,
            )
        ):
            # Convert distance to similarity score (ChromaDB uses L2 distance by default)
            similarity_score = 1.0 / (1.0 + distance)

            # Reconstruct DocumentChunk from stored data
            chunk = DocumentChunk(
                content=document,
                doc_id=metadata.get("doc_id", "") if metadata is not None else None,
                chunk_id=chunk_id,
                start_char=metadata.get("start_char", 0) if metadata is not None else None,
                end_char=metadata.get("end_char", 0) if metadata is not None else None,
                metadata=metadata if metadata is not None else {},
            )

            result = RetrievalResult(chunk=chunk, similarity_score=similarity_score, rank=i + 1, reranked=False)

            results.append(result)

        return results

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        where_clause = {}

        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = value

        return where_clause

    def _calculate_mrr(self, retrieved_ids: List[str], ground_truth_set: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_set:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _analyze_document_distribution(self, results: List[RetrievalResult]) -> Dict[str, int]:
        """Analyze distribution of results across documents"""
        doc_counts = {}
        for result in results:
            doc_id = result.chunk.doc_id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        return dict(sorted(doc_counts.items(), key=lambda x: x[1], reverse=True))

    def _analyze_content_lengths(self, results: List[RetrievalResult]) -> Dict[str, float]:
        """Analyze content length statistics"""
        lengths = [len(result.chunk.content) for result in results]
        if not lengths:
            return {}

        return {
            "mean": sum(lengths) / len(lengths),
            "max": max(lengths),
            "min": min(lengths),
            "std": self._calculate_std([float(l) for l in lengths]),
        }
