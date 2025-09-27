"""Simple async filtering without lock contention bottleneck."""

import asyncio
import time
from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()

async def filter_documents_simple_async(
    self, documents: list[str], ids: list[str]
) -> tuple[list[str], dict[str, dict[str, bool]]]:
    """Simple async filtering - just semaphore rate limiting, no token locks."""
    
    logger.info(f"Starting simple async filtering of {len(documents)} documents")
    
    # Simple semaphore-only rate limiting
    semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
    
    SYSTEM_INSTRUCTION = """
You are an assistant specialized in filtering documents based on specific criteria.
Given a document and a criterion, evaluate whether the document meets the criterion and output a single word: "yes" if the document meets the criterion, or "no" if it does not. Do not include any extra text or formatting, simply "yes" or "no".
    """.strip()
    
    async def evaluate_document_criterion(document: str, doc_id: str, criterion: str, criterion_label: str):
        """Simple evaluation with just semaphore + 429 retry."""
        async with semaphore:
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"Document: {document}\n\nCriterion: {criterion}"}
            ]
            
            # Simple retry on 429 only
            for retry in range(3):
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.config.filter_model, 
                        messages=messages, 
                        timeout=self.config.request_timeout,
                        max_tokens=10,
                        temperature=0.0
                    )
                    
                    content = response.choices[0].message.content
                    result = content.strip().lower() == "yes" if content else False
                    return doc_id, criterion_label, result
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate limit" in error_str.lower():
                        if retry < 2:  # Don't wait on last retry
                            wait_time = (retry + 1) * 2  # 2s, 4s
                            console.print(f"[yellow]⚠️  429 Rate Limit - waiting {wait_time}s[/yellow]")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    logger.error(f"API error for {doc_id} - {criterion_label}: {e}")
                    return doc_id, criterion_label, False
            
            return doc_id, criterion_label, False

    # Create all evaluation tasks
    tasks = []
    for doc_id, document in zip(ids, documents):
        for criterion, criterion_label in zip(self.config.filter_criteria, self.config.filter_criteria_labels):
            task = evaluate_document_criterion(document, doc_id, criterion, criterion_label)
            tasks.append(task)

    # Run all tasks with progress tracking
    logger.info(f"Processing {len(tasks)} evaluation tasks with {self.config.max_concurrent_requests} concurrent requests")
    
    results = []
    completed_chunks = set()
    total_chunks = len(documents)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        filter_task = progress.add_task(f"🔍 LLM Filtering ({self.config.filter_model})", total=len(tasks))
        
        for coro in asyncio.as_completed(tasks):
            try:
                doc_id, criterion_label, passed = await coro
                results.append((doc_id, criterion_label, passed))
                completed_chunks.add(doc_id)
                
                # Update description with chunk progress
                chunks_done = len(completed_chunks)
                progress.update(
                    filter_task, 
                    advance=1,
                    description=f"🔍 LLM Filtering ({self.config.filter_model}) - {chunks_done}/{total_chunks} chunks"
                )
            except Exception as e:
                logger.error(f"Task failed: {e}")
                progress.update(filter_task, advance=1)

    # Process results
    evaluation_results = {}
    for doc_id, criterion_label, passed in results:
        if doc_id not in evaluation_results:
            evaluation_results[doc_id] = {}
        evaluation_results[doc_id][criterion_label] = passed

    # Filter documents that passed all criteria
    filtered_ids = []
    for doc_id in ids:
        if doc_id in evaluation_results:
            if all(evaluation_results[doc_id].get(label, False) for label in self.config.filter_criteria_labels):
                filtered_ids.append(doc_id)

    logger.info(f"Simple filtering completed: {len(filtered_ids)}/{len(documents)} passed ({len(filtered_ids) / len(documents) * 100:.1f}%)")
    
    return filtered_ids, evaluation_results