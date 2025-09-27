"""Document filtering with LLM-based evaluation and async processing."""

import asyncio
import re
import time
from typing import Any

import polars as pl
import tiktoken
from loguru import logger
from openai import AsyncOpenAI
from openai import OpenAI as OpenAIClient
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import WikiTextConfig

# Initialize rich console for progress tracking
console = Console()


class DocumentFilter:
    """Filter documents using LLM evaluation with rate limiting and semaphore control."""

    def __init__(self, config: WikiTextConfig):
        self.config = config

        # Setup clients
        self.client = OpenAIClient(api_key=config.orq_api_key, base_url=config.orq_base_url)
        self.async_client = AsyncOpenAI(api_key=config.orq_api_key, base_url=config.orq_base_url)

        # Setup logging
        logger.configure(
            handlers=[
                {
                    "sink": "logs/document_filtering.log",
                    "level": config.log_level,
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                    "rotation": "10 MB",
                }
            ]
        )

        # Initialize token encoding
        try:
            self.encoding = tiktoken.encoding_for_model(config.filter_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _extract_retry_delay(self, error_message: str) -> float:
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

    def create_filter_batch(self, documents: list[str], ids: list[str]) -> str:
        """Create a batch job for document filtering.
        
        NOTE: This method is deprecated - use the new consecutive chunks approach instead.
        """
        logger.warning("create_filter_batch is deprecated. Use filter_documents_async instead.")
        raise NotImplementedError("Legacy batch filtering not supported with new consecutive chunks approach")

    def _create_batch_file(self, requests: list[dict]) -> str:
        """Create a file for batch processing."""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
            return f.name

    async def filter_documents_async(
        self, documents: list[str], ids: list[str], documents_df: pl.DataFrame = None
    ) -> tuple[list[str], dict[str, dict[str, bool]]]:
        """Filter documents using consecutive chunk analysis and question generation.

        Args:
            documents: List of document texts to filter
            ids: List of document IDs
            documents_df: Original DataFrame with metadata (optional)

        Returns:
            Tuple of (filtered_document_ids, evaluation_results)
        """
        logger.info(f"Starting async filtering of {len(documents)} documents using consecutive chunks")

        # Simple semaphore-only rate limiting
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Group documents into consecutive chunks
        chunk_groups = self._create_consecutive_chunk_groups(documents, ids, documents_df)
        logger.info(f"Created {len(chunk_groups)} consecutive chunk groups")

        async def evaluate_chunk_group(chunk_group: dict):
            """Evaluate a group of consecutive chunks by generating and evaluating questions."""
            async with semaphore:
                # Combine consecutive chunks
                combined_text = "\n\n".join(chunk_group["texts"])
                
                # Generate question from the combined text with reasoning
                question, question_reasoning = await self._generate_question_from_chunks(combined_text)
                if not question:
                    # If question generation fails, mark all chunks as failed
                    return {
                        doc_id: {
                            "question_quality": False,
                            "generated_question": "",
                            "question_generation_reasoning": question_reasoning,
                            "human_likeness_evaluation": False,
                            "evaluation_reasoning": "Question generation failed, so evaluation was not performed",
                            "chunk_ids": chunk_group["ids"],
                            "chunk_metadata": chunk_group.get("metadata", []),
                            "combined_text_preview": combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
                        } for doc_id in chunk_group["ids"]
                    }
                
                # Evaluate if the question seems like something a human would ask
                is_human_like, evaluation_reasoning = await self._evaluate_question_human_likeness(question, combined_text)
                
                # All chunks in the group get the same evaluation result with question info
                return {
                    doc_id: {
                        "question_quality": is_human_like,
                        "generated_question": question,
                        "question_generation_reasoning": question_reasoning,
                        "human_likeness_evaluation": is_human_like,
                        "evaluation_reasoning": evaluation_reasoning,
                        "chunk_ids": chunk_group["ids"],
                        "chunk_metadata": chunk_group.get("metadata", []),
                        "combined_text_preview": combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
                    } for doc_id in chunk_group["ids"]
                }

        # Process all chunk groups concurrently
        tasks = [evaluate_chunk_group(group) for group in chunk_groups]
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results from all groups
        evaluation_results = {}
        filtered_ids = []

        for result in group_results:
            if isinstance(result, Exception):
                logger.error(f"Chunk group evaluation failed: {result}")
                continue
            
            evaluation_results.update(result)
            
            # Add IDs where the question quality is good
            for doc_id, criteria in result.items():
                if criteria.get("question_quality", False):
                    filtered_ids.append(doc_id)

        logger.info(f"Filtering complete: {len(filtered_ids)}/{len(documents)} documents passed")
        return filtered_ids, evaluation_results

    def _create_consecutive_chunk_groups(self, documents: list[str], ids: list[str], documents_df: pl.DataFrame = None) -> list[dict]:
        """Create groups of consecutive chunks for processing.
        
        Args:
            documents: List of document texts
            ids: List of document IDs
            documents_df: Original DataFrame with metadata
            
        Returns:
            List of dictionaries with 'texts', 'ids', and 'metadata' for each group
        """
        groups = []
        chunk_count = self.config.consecutive_chunks_count
        
        # Create mapping from unique_id to row data for quick lookup
        metadata_map = {}
        if documents_df is not None:
            for row in documents_df.to_dicts():
                metadata_map[row["unique_id"]] = {
                    "unique_id": row["unique_id"],
                    "chunk_index": row.get("chunk_index"),
                    "document_url": row.get("document_url", ""),
                    "document_title": row.get("document_title", ""),
                    "source_title": row.get("source_title", ""),
                    "context_length": row.get("context_length", 0),
                    "token_count": row.get("token_count", 0)
                }
        
        for i in range(0, len(documents), chunk_count):
            end_idx = min(i + chunk_count, len(documents))
            group_ids = ids[i:end_idx]
            
            # Get metadata for each chunk in the group
            group_metadata = []
            for chunk_id in group_ids:
                if chunk_id in metadata_map:
                    group_metadata.append(metadata_map[chunk_id])
                else:
                    # Fallback metadata if not found
                    group_metadata.append({
                        "unique_id": chunk_id,
                        "chunk_index": None,
                        "document_url": "",
                        "document_title": "",
                        "source_title": "",
                        "context_length": 0,
                        "token_count": 0
                    })
            
            groups.append({
                "texts": documents[i:end_idx],
                "ids": group_ids,
                "metadata": group_metadata
            })
            
        return groups

    async def _generate_question_from_chunks(self, combined_text: str) -> tuple[str, str]:
        """Generate a question from the combined chunk text with reasoning.
        
        Args:
            combined_text: Combined text from consecutive chunks
            
        Returns:
            Tuple of (generated_question, reasoning_for_question)
        """
        QUESTION_GENERATION_PROMPT = """
You are an expert at creating natural questions based on text content.

Given the text below, generate ONE specific, well-formed question that a real human might ask about the information in this text. The question should:
1. Be about factual information contained in the text
2. Be specific and focused (not overly broad)
3. Sound natural and conversational
4. Be answerable from the provided text

Provide your response in this exact format:

QUESTION: [Your generated question here]

REASONING: [Brief explanation of why this question is appropriate for this text - what key information it targets and why a human would naturally ask this]

Text:
{text}"""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.question_generation_model,
                messages=[
                    {"role": "user", "content": QUESTION_GENERATION_PROMPT.format(text=combined_text[:3000])}
                ],
                max_tokens=200,
                temperature=0.7,
                timeout=self.config.request_timeout
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the structured response
            question = ""
            reasoning = ""
            
            if "QUESTION:" in content and "REASONING:" in content:
                parts = content.split("REASONING:", 1)
                question_part = parts[0].replace("QUESTION:", "").strip()
                reasoning_part = parts[1].strip()
                
                question = question_part
                reasoning = reasoning_part
            else:
                # Fallback if format isn't followed
                question = content
                reasoning = "No structured reasoning provided"
            
            return question, reasoning
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return "", f"Generation failed: {str(e)}"

    async def _evaluate_question_human_likeness(self, question: str, context: str) -> tuple[bool, str]:
        """Evaluate if a generated question seems like something a human would ask.
        
        Args:
            question: The generated question
            context: The source text context
            
        Returns:
            Tuple of (is_human_like, evaluation_reasoning)
        """
        EVALUATION_PROMPT = """
You are an expert at evaluating the quality and naturalness of questions.

Evaluate whether the following question sounds like something a real human would naturally ask when reading the given text. Consider:

1. **Naturalness**: Does it sound conversational and human-like?
2. **Relevance**: Is it about important/interesting information in the text?
3. **Specificity**: Is it focused on specific facts rather than overly broad?
4. **Clarity**: Is it clear and well-formed?
5. **Human curiosity**: Would a human actually be curious about this?

Provide your response in this exact format:

DECISION: [yes/no]

REASONING: [Detailed explanation of your decision, addressing the 5 criteria above. Explain specifically what makes this question human-like or not, and which criteria it passes or fails.]

Text context:
{context}

Question to evaluate:
{question}"""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.question_evaluation_model,
                messages=[
                    {"role": "user", "content": EVALUATION_PROMPT.format(
                        context=context[:2000], 
                        question=question
                    )}
                ],
                max_tokens=300,
                temperature=0.0,
                timeout=self.config.request_timeout
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the structured response
            decision = False
            reasoning = ""
            
            if "DECISION:" in content and "REASONING:" in content:
                parts = content.split("REASONING:", 1)
                decision_part = parts[0].replace("DECISION:", "").strip().lower()
                reasoning_part = parts[1].strip()
                
                decision = decision_part == "yes"
                reasoning = reasoning_part
            else:
                # Fallback parsing
                decision = "yes" in content.lower()
                reasoning = content if content else "No structured reasoning provided"
            
            return decision, reasoning
            
        except Exception as e:
            logger.error(f"Question evaluation failed: {e}")
            return False, f"Evaluation failed: {str(e)}"


    def filter_documents_sync(
        self, documents: list[str], ids: list[str]
    ) -> tuple[list[str], dict[str, dict[str, bool]]]:
        """Synchronous version of document filtering for smaller datasets."""
        logger.info(f"Starting sync filtering of {len(documents)} documents")

        SYSTEM_INSTRUCTION = """
        You are an assistant specialized in filtering documents based on specific criteria.

        Given a document and a criterion, evaluate whether the document meets the criterion and output a single word: "yes" if the document meets the criterion, or "no" if it does not. Do not include any extra text or formatting, simply "yes" or "no".
        """

        evaluation_results = {}
        filtered_ids = []

        logger.info(f"Starting synchronous filtering of {len(documents)} documents")

        for document, doc_id in zip(documents, ids, strict=False):
            evaluation_results[doc_id] = {}

            # Simple sync filtering - basic quality check
            is_substantial = len(document.strip()) > 50 and len(document.split()) > 10
            
            evaluation_results[doc_id][criterion_label] = is_substantial
            
            if is_substantial:
                filtered_ids.append(doc_id)

        logger.info(
            f"Filtered {len(filtered_ids)} documents out of {len(documents)} ({len(filtered_ids) / len(documents) * 100:.1f}%)"
        )

        return filtered_ids, evaluation_results

    async def run_filtering_pipeline(
        self, documents_df: pl.DataFrame, *, use_async: bool = True, batch_size: int = 100, checkpoint_path: str = None
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Run the complete document filtering pipeline with batching and checkpointing.

        Args:
            documents_df: DataFrame with documents to filter
            use_async: Whether to use async processing
            batch_size: Number of documents to process per batch (default: 100)
            checkpoint_path: Path to save/load checkpoints (optional)

        Returns:
            Tuple of (filtered_dataframe, filtering_metadata)
        """
        from datetime import datetime
        
        # Track filtering run metadata
        filtering_run_id = f"filtering_{int(time.time())}"
        start_time = datetime.now()
        
        logger.info(f"Starting filtering pipeline run: {filtering_run_id}")
        
        # Use simple async processing without batching bottleneck
        documents = documents_df.select("chunked_prompt").to_series().to_list()
        ids = documents_df.select("unique_id").to_series().to_list()

        if use_async:
            filtered_ids, evaluation_results = await self.filter_documents_async(documents, ids, documents_df)
        else:
            filtered_ids, evaluation_results = self.filter_documents_sync(documents, ids)

        # Filter DataFrame
        filtered_df = documents_df.filter(pl.col("unique_id").is_in(filtered_ids))
        
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()

        # Create comprehensive metadata
        metadata = {
            "run_id": filtering_run_id,
            "timestamp": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_duration_seconds": processing_duration,
            "total_documents": len(documents_df),
            "filtered_documents": len(filtered_df),
            "rejected_documents": len(documents_df) - len(filtered_df),
            "filter_rate": len(filtered_df) / len(documents_df) if documents_df.height > 0 else 0,
            "evaluation_results": evaluation_results,
            "filtering_approach": "consecutive_chunks_with_question_generation",
            "consecutive_chunks_count": self.config.consecutive_chunks_count,
            "filter_model": self.config.filter_model,
            "question_generation_model": self.config.question_generation_model,
            "question_evaluation_model": self.config.question_evaluation_model,
            "batch_size": batch_size,
            "checkpoint_path": checkpoint_path,
            "use_async": use_async,
        }
        
        # Save full filtering details to Excel BEFORE filtering decision
        await self._save_excel_filtering_details(documents_df, evaluation_results, metadata)
        
        # Save detailed filtering results for later inspection
        await self._save_filtering_results(documents_df, filtered_df, evaluation_results, metadata)
        
        # Save detailed CSV/JSON with questions and chunk metadata
        await self._save_detailed_question_results(documents_df, evaluation_results, metadata)

        return filtered_df, metadata

    async def filter_documents_batched(
        self, documents_df: pl.DataFrame, *, batch_size: int = 100, checkpoint_path: str = None
    ) -> tuple[list[str], dict[str, dict[str, bool]]]:
        """Filter documents in batches with checkpointing for resilience.

        Args:
            documents_df: DataFrame with documents to filter
            batch_size: Number of documents per batch
            checkpoint_path: Path to save/load progress checkpoints

        Returns:
            Tuple of (filtered_document_ids, evaluation_results)
        """
        import json
        import os
        from pathlib import Path

        # Setup checkpoint management
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else Path("logs/filtering_checkpoint.json")
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing progress if checkpoint exists
        processed_ids = set()
        evaluation_results = {}
        start_batch = 0

        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            try:
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                    processed_ids = set(checkpoint_data.get("processed_ids", []))
                    evaluation_results = checkpoint_data.get("evaluation_results", {})
                    start_batch = checkpoint_data.get("last_batch", 0)
                    logger.info(f"Resuming from batch {start_batch}, {len(processed_ids)} documents already processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")

        # Get remaining documents to process
        total_docs = len(documents_df)
        if processed_ids:
            remaining_df = documents_df.filter(~pl.col("unique_id").is_in(list(processed_ids)))
        else:
            remaining_df = documents_df

        logger.info(f"Processing {len(remaining_df)} remaining documents in batches of {batch_size}")
        logger.info(f"LLM Model: {self.config.filter_model}")
        logger.info(f"Filter Approach: {getattr(self.config, 'filtering_approach', 'consecutive_chunks_with_question_generation')}")
        logger.info(
            f"Concurrency: {self.config.max_concurrent_requests} | Rate limit: {self.config.tokens_per_minute:,} tokens/min"
        )

        # Process documents in batches with circuit breaker
        total_batches = (len(remaining_df) + batch_size - 1) // batch_size
        consecutive_failures = 0
        max_consecutive_failures = 3

        logger.info(f"Starting {total_batches} batches of LLM filtering with checkpointing enabled")

        # Use main console for filtering progress (visible in stdout)
        filtering_console = console

        # Create rich progress bar for batch processing with dedicated console
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=filtering_console,
            transient=False,
        ) as progress:
            batch_task = progress.add_task(f"🔍 LLM Filtering ({self.config.filter_model})", total=total_batches)

            # Update progress to show already completed batches if resuming
            if start_batch > 0:
                progress.update(batch_task, advance=start_batch)

            for batch_idx in range(start_batch, total_batches):
                # Check for cancellation at start of each batch
                try:
                    batch_start = (batch_idx - start_batch) * batch_size
                    batch_end = min(batch_start + batch_size, len(remaining_df))

                    batch_df = remaining_df[batch_start:batch_end]
                    batch_documents = batch_df.select("chunked_prompt").to_series().to_list()
                    batch_ids = batch_df.select("unique_id").to_series().to_list()

                    # Update progress bar description with current batch info
                    progress.update(
                        batch_task,
                        description=f"🔍 LLM Filtering - Batch {batch_idx + 1}/{total_batches}",
                    )

                    # Estimate processing time and tokens (debug level only)
                    estimated_tokens = sum(
                        len(self.encoding.encode(doc[:1000])) for doc in batch_documents[:5]
                    )  # Sample estimation
                    estimated_total_tokens = estimated_tokens * len(batch_documents) // 5
                    logger.debug(
                        f"Batch {batch_idx + 1} estimated tokens: ~{estimated_total_tokens:,} (for question generation and evaluation)"
                    )
                except KeyboardInterrupt:
                    logger.info(f"Filtering interrupted by user at batch {batch_idx + 1}")
                    # Save current progress before raising
                    checkpoint_data = {
                        "processed_ids": list(processed_ids),
                        "evaluation_results": evaluation_results,
                        "last_batch": batch_idx,
                        "total_batches": total_batches,
                        "timestamp": time.time(),
                        "interrupted": True,
                    }
                    with open(checkpoint_file, "w") as f:
                        json.dump(checkpoint_data, f)
                    logger.info("Progress saved before interruption")
                    raise

                try:
                    # Process this batch with timeout and cancellation handling
                    batch_task_async = asyncio.create_task(self.filter_documents_async(batch_documents, batch_ids))

                    try:
                        batch_filtered_ids, batch_results = await asyncio.wait_for(
                            batch_task_async,
                            timeout=300,  # 5 minute timeout per batch
                        )
                    except KeyboardInterrupt:
                        logger.info(f"Filtering interrupted during batch {batch_idx + 1} processing")
                        batch_task_async.cancel()  # Cancel the async task
                        try:
                            await batch_task_async  # Wait for cancellation to complete
                        except asyncio.CancelledError:
                            pass
                        raise

                    # Update results
                    evaluation_results.update(batch_results)
                    processed_ids.update(batch_ids)

                    # Save checkpoint after each batch
                    checkpoint_data = {
                        "processed_ids": list(processed_ids),
                        "evaluation_results": evaluation_results,
                        "last_batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "timestamp": time.time(),
                        "consecutive_failures": 0,
                    }

                    with open(checkpoint_file, "w") as f:
                        json.dump(checkpoint_data, f)

                    consecutive_failures = 0  # Reset on success

                    # Update progress bar
                    progress.update(batch_task, advance=1)

                    logger.info(f"Batch {batch_idx + 1}/{total_batches} completed successfully ✓ Checkpoint saved")

                except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                    logger.error(f"Batch {batch_idx + 1} failed with {type(e).__name__}: {e}")
                    consecutive_failures += 1

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive failures ({consecutive_failures}), aborting")
                        raise

                    # Save checkpoint with failure info
                    checkpoint_data = {
                        "processed_ids": list(processed_ids),
                        "evaluation_results": evaluation_results,
                        "last_batch": batch_idx,  # Don't increment, retry this batch
                        "total_batches": total_batches,
                        "timestamp": time.time(),
                        "consecutive_failures": consecutive_failures,
                    }

                    with open(checkpoint_file, "w") as f:
                        json.dump(checkpoint_data, f)

                    logger.info(f"Checkpoint saved after batch failure, will retry batch {batch_idx + 1}")
                    await asyncio.sleep(30)  # Wait before retrying

                except Exception as e:
                    logger.error(f"Unexpected error in batch {batch_idx + 1}: {e}")
                    consecutive_failures += 1

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive failures, aborting")
                        raise

                    logger.info("Continuing to next batch after error")
                    await asyncio.sleep(10)

        # Determine final filtered IDs
        filtered_ids = []
        for doc_id in processed_ids:
            if doc_id in evaluation_results:
                if evaluation_results[doc_id].get("question_quality", False):
                    filtered_ids.append(doc_id)

        # Clean up checkpoint file on successful completion
        if checkpoint_file.exists():
            os.remove(checkpoint_file)
            logger.info("Processing completed successfully, checkpoint file removed")

        logger.info(
            f"Batched filtering completed: {len(filtered_ids)} documents out of {total_docs} passed filters "
            f"({len(filtered_ids) / total_docs * 100:.1f}%)"
        )

        return filtered_ids, evaluation_results

    async def _save_filtering_results(
        self, 
        original_df: pl.DataFrame, 
        filtered_df: pl.DataFrame, 
        evaluation_results: dict[str, dict[str, bool]], 
        metadata: dict[str, Any]
    ) -> None:
        """Save comprehensive filtering results for later inspection."""
        import json
        from pathlib import Path
        
        # Create results directory
        results_dir = Path("logs/filtering_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        run_id = metadata["run_id"]
        timestamp = metadata["timestamp"][:19].replace(":", "-")  # Format for filename
        
        try:
            # 1. Save detailed evaluation results with document content
            detailed_results = self._create_detailed_results_data(original_df, evaluation_results, metadata)
            detailed_file = results_dir / f"{run_id}_{timestamp}_detailed.json"
            
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Detailed filtering results saved: {detailed_file}")
            
            # 2. Save filtering summary report
            summary_report = self._create_filtering_summary(original_df, filtered_df, evaluation_results, metadata)
            summary_file = results_dir / f"{run_id}_{timestamp}_summary.md"
            
            with open(summary_file, 'w') as f:
                f.write(summary_report)
            
            logger.info(f"✓ Filtering summary report saved: {summary_file}")
            
            # 3. Save criteria-specific analysis
            criteria_analysis = self._create_criteria_analysis(evaluation_results, metadata)
            criteria_file = results_dir / f"{run_id}_{timestamp}_criteria_analysis.json"
            
            with open(criteria_file, 'w') as f:
                json.dump(criteria_analysis, f, indent=2)
            
            logger.info(f"✓ Criteria analysis saved: {criteria_file}")
            
            # 4. Save rejected documents for inspection
            rejected_data = self._create_rejected_documents_data(original_df, filtered_df, evaluation_results)
            rejected_file = results_dir / f"{run_id}_{timestamp}_rejected_documents.json"
            
            with open(rejected_file, 'w') as f:
                json.dump(rejected_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Rejected documents data saved: {rejected_file}")
            
            # 5. Export filtered documents DataFrame for further analysis
            filtered_csv = results_dir / f"{run_id}_{timestamp}_filtered_documents.csv"
            filtered_df.write_csv(filtered_csv)
            
            logger.info(f"✓ Filtered documents CSV saved: {filtered_csv}")
            
            console.print(f"[green]📁 Filtering results saved to: {results_dir}[/green]")
            
        except Exception as e:
            logger.error(f"Failed to save filtering results: {e}")
            console.print(f"[red]⚠️ Failed to save filtering results: {e}[/red]")

    def _create_detailed_results_data(
        self, 
        original_df: pl.DataFrame, 
        evaluation_results: dict[str, dict[str, bool]], 
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Create detailed results data with document content and evaluations."""
        detailed_results = {
            "metadata": metadata,
            "documents": []
        }
        
        # Get document data for analysis - handle missing columns gracefully
        available_columns = ["unique_id", "chunked_prompt"]
        if "source_title" in original_df.columns:
            available_columns.append("source_title")
        doc_data = original_df.select(available_columns).to_dicts()
        
        for doc in doc_data:
            doc_id = doc["unique_id"]
            doc_evaluations = evaluation_results.get(doc_id, {})
            
            # Determine if document passed the question quality check
            all_passed = doc_evaluations.get("question_quality", False)
            
            document_result = {
                "document_id": doc_id,
                "source_title": doc.get("source_title", "Unknown"),
                "content_preview": doc["chunked_prompt"][:500] + "..." if len(doc["chunked_prompt"]) > 500 else doc["chunked_prompt"],
                "content_length": len(doc["chunked_prompt"]),
                "passed_filtering": all_passed,
                "criteria_evaluations": doc_evaluations,
                "failed_criteria": [] if all_passed else ["question_quality"]
            }
            
            detailed_results["documents"].append(document_result)
        
        return detailed_results

    def _create_filtering_summary(
        self, 
        original_df: pl.DataFrame, 
        filtered_df: pl.DataFrame, 
        evaluation_results: dict[str, dict[str, bool]], 
        metadata: dict[str, Any]
    ) -> str:
        """Create a human-readable filtering summary report."""
        from datetime import datetime
        
        total_docs = len(original_df)
        passed_docs = len(filtered_df)
        rejected_docs = total_docs - passed_docs
        pass_rate = (passed_docs / total_docs * 100) if total_docs > 0 else 0
        
        # Calculate question quality statistics
        passed_count = sum(
            1 for doc_results in evaluation_results.values() 
            if doc_results.get("question_quality", False)
        )
        criteria_stats = {
            "question_quality": {
                "passed": passed_count,
                "failed": total_docs - passed_count,
                "pass_rate": (passed_count / total_docs * 100) if total_docs > 0 else 0
            }
        }
        
        # Create markdown report
        report = f"""# WikiText Filtering Results Summary

## Run Information
- **Run ID**: {metadata['run_id']}
- **Timestamp**: {metadata['timestamp']}
- **Duration**: {metadata['processing_duration_seconds']:.2f} seconds
- **Filter Model**: {metadata['filter_model']}
- **Processing Mode**: {'Async' if metadata['use_async'] else 'Sync'}

## Overall Results
- **Total Documents Processed**: {total_docs:,}
- **Documents Passed**: {passed_docs:,} ({pass_rate:.1f}%)
- **Documents Rejected**: {rejected_docs:,} ({100-pass_rate:.1f}%)

## Filter Criteria Results

| Criterion | Passed | Failed | Pass Rate |
|-----------|---------|--------|-----------|
"""
        
        for criterion_label, stats in criteria_stats.items():
            report += f"| **{criterion_label}** | {stats['passed']:,} | {stats['failed']:,} | {stats['pass_rate']:.1f}% |\n"
        
        report += f"""
## Filtering Approach Used

**Method:** Consecutive chunks with question generation and human-likeness evaluation

**Parameters:**
- Consecutive chunks count: {metadata.get('consecutive_chunks_count', 3)}
- Question generation model: {metadata.get('question_generation_model', 'unknown')}
- Question evaluation model: {metadata.get('question_evaluation_model', 'unknown')}

"""
        
        # Find failure patterns for question quality
        failure_patterns = {}
        for doc_results in evaluation_results.values():
            if not doc_results.get("question_quality", False):
                failure_key = "question_quality_failed"
                failure_patterns[failure_key] = failure_patterns.get(failure_key, 0) + 1
        
        if failure_patterns:
            report += "## Common Failure Patterns\n\n"
            for pattern, count in sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
                if pattern:  # Only show actual failures
                    pattern_str = ", ".join(pattern)
                    percentage = (count / rejected_docs * 100) if rejected_docs > 0 else 0
                    report += f"- **{pattern_str}**: {count:,} documents ({percentage:.1f}% of rejected)\n"
        
        report += f"""
## Files Generated
- **Detailed Results**: `{metadata['run_id']}_{metadata['timestamp'][:19].replace(':', '-')}_detailed.json`
- **Criteria Analysis**: `{metadata['run_id']}_{metadata['timestamp'][:19].replace(':', '-')}_criteria_analysis.json`
- **Rejected Documents**: `{metadata['run_id']}_{metadata['timestamp'][:19].replace(':', '-')}_rejected_documents.json`
- **Filtered Documents CSV**: `{metadata['run_id']}_{metadata['timestamp'][:19].replace(':', '-')}_filtered_documents.csv`

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

    def _create_criteria_analysis(
        self, 
        evaluation_results: dict[str, dict[str, bool]], 
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Create detailed analysis of question quality filtering performance."""
        total_docs = len(evaluation_results)
        
        analysis = {
            "run_metadata": {
                "run_id": metadata["run_id"],
                "timestamp": metadata["timestamp"],
                "total_documents": total_docs,
                "filter_model": metadata["filter_model"],
                "consecutive_chunks_count": metadata.get("consecutive_chunks_count", 3),
                "question_generation_model": metadata.get("question_generation_model", "unknown"),
                "question_evaluation_model": metadata.get("question_evaluation_model", "unknown")
            },
            "question_quality_performance": {},
            "filtering_summary": {}
        }
        
        # Analyze question quality criterion performance
        passed_docs = [
            doc_id for doc_id, results in evaluation_results.items() 
            if results.get("question_quality", False)
        ]
        failed_docs = [
            doc_id for doc_id, results in evaluation_results.items() 
            if not results.get("question_quality", False)
        ]
        
        analysis["question_quality_performance"] = {
            "criterion_description": "Generated question passes human-likeness evaluation",
            "passed_count": len(passed_docs),
            "failed_count": len(failed_docs),
            "pass_rate": len(passed_docs) / total_docs if total_docs > 0 else 0,
            "passed_doc_ids": passed_docs[:100],  # Sample for inspection
            "failed_doc_ids": failed_docs[:100]   # Sample for inspection
        }
        
        # Summary statistics
        analysis["filtering_summary"] = {
            "total_chunk_groups_evaluated": total_docs // metadata.get("consecutive_chunks_count", 3),
            "total_chunks_processed": total_docs,
            "chunks_passed": len(passed_docs),
            "chunks_failed": len(failed_docs),
            "overall_pass_rate": len(passed_docs) / total_docs if total_docs > 0 else 0,
            "filtering_approach": "consecutive_chunks_with_question_generation"
        }
        
        return analysis

    def _create_rejected_documents_data(
        self, 
        original_df: pl.DataFrame, 
        filtered_df: pl.DataFrame, 
        evaluation_results: dict[str, dict[str, bool]]
    ) -> dict[str, Any]:
        """Create detailed data about rejected documents for inspection."""
        # Get IDs of filtered (passed) documents
        passed_ids = set(filtered_df.select("unique_id").to_series().to_list())
        
        # Get data for all documents - handle missing columns gracefully
        available_columns = ["unique_id", "chunked_prompt"]
        if "source_title" in original_df.columns:
            available_columns.append("source_title")
        all_docs = original_df.select(available_columns).to_dicts()
        
        # Filter to get only rejected documents
        rejected_documents = []
        for doc in all_docs:
            doc_id = doc["unique_id"]
            if doc_id not in passed_ids:
                doc_evaluations = evaluation_results.get(doc_id, {})
                
                rejected_doc = {
                    "document_id": doc_id,
                    "source_title": doc.get("source_title", "Unknown"),
                    "content_preview": doc["chunked_prompt"][:300] + "..." if len(doc["chunked_prompt"]) > 300 else doc["chunked_prompt"],
                    "content_length": len(doc["chunked_prompt"]),
                    "criteria_evaluations": doc_evaluations,
                    "failed_criteria": [
                        label for label, passed in doc_evaluations.items() 
                        if not passed
                    ]
                }
                
                rejected_documents.append(rejected_doc)
        
        # Sort by number of failed criteria (most problematic first)
        rejected_documents.sort(key=lambda x: len(x["failed_criteria"]), reverse=True)
        
        return {
            "total_rejected": len(rejected_documents),
            "documents": rejected_documents
        }
    
    async def _save_detailed_question_results(
        self,
        original_df: pl.DataFrame,
        evaluation_results: dict[str, dict[str, bool]],
        metadata: dict[str, Any]
    ) -> None:
        """Save detailed results with generated questions and chunk metadata to CSV/JSON."""
        import json
        from pathlib import Path
        
        # Create results directory
        results_dir = Path("logs/filtering_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        run_id = metadata["run_id"]
        timestamp = metadata["timestamp"][:19].replace(":", "-")  # Format for filename
        
        try:
            # Prepare detailed data for each chunk
            detailed_data = []
            
            for row in original_df.to_dicts():
                chunk_id = row["unique_id"]
                evaluation = evaluation_results.get(chunk_id, {})
                
                chunk_record = {
                    # Chunk identification
                    "unique_id": chunk_id,
                    "chunk_index": row.get("chunk_index"),
                    "document_url": row.get("document_url", ""),
                    "document_title": row.get("document_title", ""),
                    "source_title": row.get("source_title", ""),
                    
                    # Content information
                    "chunked_prompt": row.get("chunked_prompt", ""),
                    "context_length": row.get("context_length", 0),
                    "token_count": row.get("token_count", 0),
                    
                    # Filtering results
                    "question_quality_passed": evaluation.get("question_quality", False),
                    "generated_question": evaluation.get("generated_question", ""),
                    "question_generation_reasoning": evaluation.get("question_generation_reasoning", ""),
                    "human_likeness_evaluation": evaluation.get("human_likeness_evaluation", False),
                    "evaluation_reasoning": evaluation.get("evaluation_reasoning", ""),
                    "combined_text_preview": evaluation.get("combined_text_preview", ""),
                    
                    # Group information
                    "chunk_group_ids": evaluation.get("chunk_ids", []),
                    "chunk_group_size": len(evaluation.get("chunk_ids", [])),
                    
                    # Metadata from chunk group
                    "group_metadata": evaluation.get("chunk_metadata", []),
                    
                    # Run metadata
                    "filtering_run_id": run_id,
                    "filtering_timestamp": metadata["timestamp"],
                    "consecutive_chunks_count": metadata.get("consecutive_chunks_count", 3)
                }
                
                detailed_data.append(chunk_record)
            
            # Save as JSON
            json_file = results_dir / f"{run_id}_{timestamp}_detailed_questions.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "run_metadata": {
                        "run_id": run_id,
                        "timestamp": metadata["timestamp"],
                        "filtering_approach": metadata.get("filtering_approach", "consecutive_chunks_with_question_generation"),
                        "consecutive_chunks_count": metadata.get("consecutive_chunks_count", 3),
                        "question_generation_model": metadata.get("question_generation_model", "unknown"),
                        "question_evaluation_model": metadata.get("question_evaluation_model", "unknown"),
                        "total_chunks": len(detailed_data),
                        "passed_chunks": sum(1 for d in detailed_data if d["question_quality_passed"])
                    },
                    "chunk_results": detailed_data
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Detailed question results JSON saved: {json_file}")
            
            # Save as CSV for easy analysis
            csv_file = results_dir / f"{run_id}_{timestamp}_detailed_questions.csv"
            
            # Flatten the data for CSV (remove complex nested structures)
            csv_data = []
            for record in detailed_data:
                csv_record = record.copy()
                # Convert lists to string representations for CSV
                csv_record["chunk_group_ids"] = "|".join(record["chunk_group_ids"])
                # Remove complex nested metadata for CSV
                del csv_record["group_metadata"]
                csv_data.append(csv_record)
            
            # Create DataFrame and save as CSV
            import polars as pl
            results_df = pl.DataFrame(csv_data)
            results_df.write_csv(csv_file)
            
            logger.info(f"✓ Detailed question results CSV saved: {csv_file}")
            
            # Create summary statistics
            summary_stats = {
                "total_chunks_processed": len(detailed_data),
                "chunks_passed": sum(1 for d in detailed_data if d["question_quality_passed"]),
                "chunks_failed": sum(1 for d in detailed_data if not d["question_quality_passed"]),
                "pass_rate": sum(1 for d in detailed_data if d["question_quality_passed"]) / len(detailed_data) if detailed_data else 0,
                "total_questions_generated": sum(1 for d in detailed_data if d["generated_question"]),
                "unique_chunk_groups": len(set("|".join(d["chunk_group_ids"]) for d in detailed_data if d["chunk_group_ids"])),
                "average_group_size": sum(d["chunk_group_size"] for d in detailed_data) / len(detailed_data) if detailed_data else 0
            }
            
            summary_file = results_dir / f"{run_id}_{timestamp}_question_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            logger.info(f"✓ Question filtering summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save detailed question results: {e}")
            raise
    
    async def _save_excel_filtering_details(
        self,
        original_df: pl.DataFrame,
        evaluation_results: dict[str, dict[str, bool]],
        metadata: dict[str, Any]
    ) -> None:
        """Save complete filtering details with reasoning to Excel file BEFORE filtering decision."""
        try:
            import pandas as pd
            from pathlib import Path
            
            # Create results directory
            results_dir = Path("logs/filtering_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            run_id = metadata["run_id"]
            timestamp = metadata["timestamp"][:19].replace(":", "-")
            
            # Prepare comprehensive data for each chunk
            excel_data = []
            
            for row in original_df.to_dicts():
                chunk_id = row["unique_id"]
                evaluation = evaluation_results.get(chunk_id, {})
                
                # Determine filtering decision and create recommendation
                question_quality_passed = evaluation.get("question_quality", False)
                filtering_decision = "PASS" if question_quality_passed else "FAIL"
                
                excel_record = {
                    # == CHUNK IDENTIFICATION ==
                    "unique_id": chunk_id,
                    "chunk_index": row.get("chunk_index", ""),
                    "document_url": row.get("document_url", ""),
                    "document_title": row.get("document_title", ""),
                    "source_title": row.get("source_title", ""),
                    
                    # == CONTENT DETAILS ==
                    "chunked_prompt": row.get("chunked_prompt", ""),
                    "context_length": row.get("context_length", 0),
                    "token_count": row.get("token_count", 0),
                    "combined_text_preview": evaluation.get("combined_text_preview", ""),
                    
                    # == CHUNK GROUP INFORMATION ==
                    "chunk_group_ids": "|".join(evaluation.get("chunk_ids", [])),
                    "chunk_group_size": len(evaluation.get("chunk_ids", [])),
                    "consecutive_chunks_count": metadata.get("consecutive_chunks_count", 3),
                    
                    # == QUESTION GENERATION ==
                    "generated_question": evaluation.get("generated_question", ""),
                    "question_generation_reasoning": evaluation.get("question_generation_reasoning", ""),
                    
                    # == HUMAN-LIKENESS EVALUATION ==
                    "human_likeness_evaluation": evaluation.get("human_likeness_evaluation", False),
                    "evaluation_reasoning": evaluation.get("evaluation_reasoning", ""),
                    
                    # == FILTERING DECISION ==
                    "final_filtering_decision": filtering_decision,
                    "question_quality_passed": question_quality_passed,
                    
                    # == MODEL INFORMATION ==
                    "question_generation_model": metadata.get("question_generation_model", ""),
                    "question_evaluation_model": metadata.get("question_evaluation_model", ""),
                    "filter_model": metadata.get("filter_model", ""),
                    
                    # == RUN METADATA ==
                    "filtering_run_id": run_id,
                    "filtering_timestamp": metadata["timestamp"],
                    "filtering_approach": metadata.get("filtering_approach", "consecutive_chunks_with_question_generation")
                }
                
                excel_data.append(excel_record)
            
            # Create pandas DataFrame
            df = pd.DataFrame(excel_data)
            
            # Save to Excel with multiple sheets
            excel_file = results_dir / f"{run_id}_{timestamp}_FULL_filtering_analysis.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Filtering_Analysis', index=False)
                
                # Summary statistics sheet
                summary_data = {
                    "Metric": [
                        "Total Chunks Processed",
                        "Chunks Passed",
                        "Chunks Failed", 
                        "Pass Rate (%)",
                        "Questions Successfully Generated",
                        "Questions Failed Generation",
                        "Human-like Questions",
                        "Non-human-like Questions",
                        "Unique Chunk Groups",
                        "Average Group Size",
                        "Question Generation Model",
                        "Question Evaluation Model",
                        "Consecutive Chunks Count",
                        "Run Timestamp"
                    ],
                    "Value": [
                        len(excel_data),
                        sum(1 for d in excel_data if d["question_quality_passed"]),
                        sum(1 for d in excel_data if not d["question_quality_passed"]),
                        round((sum(1 for d in excel_data if d["question_quality_passed"]) / len(excel_data) * 100), 2) if excel_data else 0,
                        sum(1 for d in excel_data if d["generated_question"]),
                        sum(1 for d in excel_data if not d["generated_question"]),
                        sum(1 for d in excel_data if d["human_likeness_evaluation"]),
                        sum(1 for d in excel_data if not d["human_likeness_evaluation"]),
                        len(set(d["chunk_group_ids"] for d in excel_data if d["chunk_group_ids"])),
                        round(sum(d["chunk_group_size"] for d in excel_data) / len(excel_data), 2) if excel_data else 0,
                        metadata.get("question_generation_model", "unknown"),
                        metadata.get("question_evaluation_model", "unknown"),
                        metadata.get("consecutive_chunks_count", 3),
                        metadata["timestamp"]
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Failed cases analysis sheet (for easy debugging)
                failed_df = df[df["question_quality_passed"] == False].copy()
                if not failed_df.empty:
                    # Select key columns for failed case analysis
                    failed_analysis = failed_df[[
                        "unique_id", "document_title", "source_title",
                        "generated_question", "question_generation_reasoning",
                        "human_likeness_evaluation", "evaluation_reasoning",
                        "combined_text_preview"
                    ]].copy()
                    
                    failed_analysis.to_excel(writer, sheet_name='Failed_Cases_Analysis', index=False)
                
                # Question samples sheet (mix of passed and failed)
                sample_size = min(50, len(excel_data))
                sample_indices = list(range(0, len(excel_data), max(1, len(excel_data) // sample_size)))[:sample_size]
                sample_df = df.iloc[sample_indices][[
                    "unique_id", "document_title", "generated_question", 
                    "question_generation_reasoning", "evaluation_reasoning",
                    "final_filtering_decision", "combined_text_preview"
                ]].copy()
                
                sample_df.to_excel(writer, sheet_name='Question_Samples', index=False)
            
            logger.info(f"✅ Complete Excel filtering analysis saved: {excel_file}")
            logger.info(f"   📊 {len(excel_data)} chunks analyzed across {len(set(d['chunk_group_ids'] for d in excel_data if d['chunk_group_ids']))} chunk groups")
            logger.info(f"   ✅ {sum(1 for d in excel_data if d['question_quality_passed'])} chunks passed, {sum(1 for d in excel_data if not d['question_quality_passed'])} chunks failed")
            
        except Exception as e:
            logger.error(f"Failed to save Excel filtering details: {e}")
            # Don't raise - this is supplementary output
            logger.warning("Continuing with filtering process despite Excel export failure")
