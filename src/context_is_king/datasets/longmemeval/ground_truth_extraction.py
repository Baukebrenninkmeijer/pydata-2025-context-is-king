"""
Ground Truth Extraction for LongMemEval Dataset

Extracts ground truth chunk relevance annotations for retrieval evaluation
by leveraging the focused vs full conversation structure in LongMemEval.

This module is SPECIFICALLY designed for the LongMemEval dataset structure:
- focused_prompt: Contains only relevant conversation sections
- full_prompt: Contains complete conversation history with distractors
- Questions with expected answers for evaluation

Key Components:
- GroundTruthExtractor: Main orchestrator for ChromaDB-based extraction
- ConversationChunker: Splits conversations into overlapping chunks
- ContentOverlapMetrics: Calculates Jaccard similarity and ROUGE-L metrics

Usage:
    from context_is_king.datasets.longmemeval import GroundTruthExtractor

    extractor = GroundTruthExtractor()
    extractor.ingest_conversations(full_conversations_df)
    ground_truth = extractor.extract_ground_truth(focused_conversations_df)
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Literal

import chromadb
import dotenv
import pandas as pd
import tiktoken
from chromadb.utils.embedding_functions.huggingface_embedding_function import HuggingFaceEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from openai import OpenAI as OpenAIClient
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

root_dir = Path(__file__).parents[4]

dotenv.load_dotenv(root_dir / ".env", override=True)

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

console = Console()


class LongMemEvalPromptCleaner:
    """
    Cleans LongMemEval prompts to extract only the conversation data.

    Removes the instruction prompt header and question/answer footer,
    leaving just the conversation history for embedding and similarity search.
    """

    @staticmethod
    def clean_prompt(prompt_text: str) -> str:
        """
        Extract only conversation data from LongMemEval prompts.

        Removes:
        - Header: "I will give you several user/history chats. Please answer..."
        - Footer: "Current Date: ... Question: ... Only output the answer..."

        Args:
            prompt_text: Raw LongMemEval prompt with instructions

        Returns:
            Clean conversation content only
        """

        if not prompt_text or not isinstance(prompt_text, str):
            return ""

        # Find the start of conversation data
        # Look for "History Chats:" followed by session info
        history_start_patterns = [
            r"History Chats:\s*Session",
            r"History Chats:\s*\[",  # For full prompts that start with [
            r"\n\n\[{'role':",  # Direct JSON conversation start
        ]

        conversation_start = None
        for pattern in history_start_patterns:
            match = re.search(pattern, prompt_text)
            if match:
                # Start from "History Chats:" if found, otherwise from the match
                history_match = re.search(r"History Chats:\s*", prompt_text[: match.start() + 50])
                if history_match:
                    conversation_start = history_match.end()
                else:
                    conversation_start = match.start()
                break

        if conversation_start is None:
            console.print("[yellow]⚠️ Could not find conversation start pattern, using full text[/yellow]")
            return prompt_text

        # Find the end of conversation data
        # Look for "Current Date:" or "Question:" patterns
        end_patterns = [
            r"\n\nCurrent Date:",
            r"\nQuestion:",
            r"\n\nQuestion:",
        ]

        conversation_end = len(prompt_text)  # Default to end of text
        for pattern in end_patterns:
            match = re.search(pattern, prompt_text[conversation_start:])
            if match:
                conversation_end = conversation_start + match.start()
                break

        # Extract the conversation content
        conversation_content = prompt_text[conversation_start:conversation_end].strip()

        # Additional cleaning: remove any stray instruction fragments
        conversation_content = re.sub(
            r"Please answer the question based on.*?(?=\n|$)", "", conversation_content, flags=re.IGNORECASE
        )

        # Clean up extra whitespace
        conversation_content = re.sub(r"\n\n+", "\n\n", conversation_content)
        conversation_content = conversation_content.strip()

        return conversation_content

    @staticmethod
    def validate_cleaning(original: str, cleaned: str, sample_id: str) -> bool:
        """
        Validate that prompt cleaning worked correctly.

        Args:
            original: Original prompt text
            cleaned: Cleaned conversation content
            sample_id: Sample identifier for logging

        Returns:
            True if cleaning appears successful
        """

        # Check that we removed significant content (instructions)
        size_reduction = len(original) - len(cleaned)
        reduction_ratio = size_reduction / len(original) if original else 0

        # Should remove some content but not too much (keep most conversation)
        if reduction_ratio < 0.05:  # Less than 5% removed - might not have cleaned properly
            console.print(f"[yellow]⚠️ {sample_id}: Low reduction ratio ({reduction_ratio:.2%})[/yellow]")
        elif reduction_ratio > 0.8:  # More than 80% removed - might have over-cleaned
            console.print(f"[red]⚠️ {sample_id}: High reduction ratio ({reduction_ratio:.2%})[/red]")
            return False

        # Check that instructions are removed
        instruction_fragments = [
            "I will give you several",
            "Please answer the question",
            "Only output the answer",
            "Current Date:",
        ]

        for fragment in instruction_fragments:
            if fragment.lower() in cleaned.lower():
                console.print(f"[yellow]⚠️ {sample_id}: Instruction fragment remains: {fragment}[/yellow]")

        # Check that conversation content remains
        conversation_indicators = [
            "role",
            "content",
            "user",
            "assistant",
            "Session",
        ]

        has_conversation = any(indicator in cleaned for indicator in conversation_indicators)
        if not has_conversation:
            console.print(f"[red]⚠️ {sample_id}: No conversation indicators found[/red]")
            return False

        return True


class ContentOverlapMetrics:
    """
    Calculate content overlap metrics for LongMemEval ground truth extraction.

    Computes multiple overlap metrics between focused content and conversation chunks:
    - Jaccard Similarity: Standard token-level intersection over union
    - ROUGE-L: Longest common subsequence based similarity (if available)

    Optimized for LongMemEval's focused vs full conversation structure.
    """

    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            console.print("[green]✓ ROUGE-L scorer initialized[/green]")
        else:
            self.rouge_scorer = None
            console.print("[yellow]ℹ️  ROUGE-L not available (optional dependency)[/yellow]")

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Token-level Jaccard similarity (intersection over union).

        Well-suited for LongMemEval because it measures how much the focused
        content overlaps with chunks, regardless of order.

        Args:
            text1: Focused content text
            text2: Conversation chunk text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    def rouge_l_similarity(self, text1: str, text2: str) -> float:
        """
        ROUGE-L similarity using longest common subsequence.

        Particularly good for LongMemEval because it captures sequential
        overlap, which is important for conversation flow.

        Args:
            text1: Focused content text
            text2: Conversation chunk text

        Returns:
            ROUGE-L F1 score between 0.0 and 1.0, or 0.0 if ROUGE unavailable
        """
        if not self.rouge_scorer:
            return 0.0

        try:
            scores = self.rouge_scorer.score(text1, text2)
            return scores["rougeL"].fmeasure
        except Exception as e:
            console.print(f"[yellow]⚠ ROUGE-L calculation failed: {e}[/yellow]")
            return 0.0

    def token_containment(self, focused_text: str, chunk_text: str) -> float:
        """
        Fraction of focused content tokens that appear in the chunk.

        Useful for LongMemEval to measure how completely the chunk
        covers the focused content. This is asymmetric - it doesn't penalize
        the chunk for containing additional information.

        Args:
            focused_text: The focused (relevant) content
            chunk_text: The conversation chunk

        Returns:
            Containment ratio between 0.0 and 1.0
        """
        focused_tokens = set(focused_text.lower().split())
        chunk_tokens = set(chunk_text.lower().split())
        intersection = len(focused_tokens & chunk_tokens)
        return intersection / len(focused_tokens) if focused_tokens else 0.0

    def asymmetric_jaccard(self, query_text: str, chunk_text: str, weight: float = 0.8) -> float:
        """
        Asymmetric Jaccard similarity that emphasizes query coverage.

        This metric combines:
        - Token containment (how much of query is in chunk)
        - Standard Jaccard (overall similarity)

        The weight parameter controls the balance:
        - weight=1.0: Pure token containment (fully asymmetric)
        - weight=0.5: Equal balance
        - weight=0.0: Pure Jaccard (symmetric)

        Args:
            query_text: The query or focused content
            chunk_text: The conversation chunk
            weight: Weight for token containment (0.0 to 1.0)

        Returns:
            Asymmetric similarity score between 0.0 and 1.0
        """
        containment = self.token_containment(query_text, chunk_text)
        jaccard = self.jaccard_similarity(query_text, chunk_text)
        return weight * containment + (1 - weight) * jaccard

    def calculate_primary_metrics(self, focused_text: str, chunk_text: str) -> dict[str, float]:
        """
        Calculate primary overlap metrics for LongMemEval ground truth.

        Uses Jaccard similarity as the main metric, with ROUGE-L as secondary
        if available. These two metrics provide complementary information:
        - Jaccard: Token-level overlap regardless of order
        - ROUGE-L: Sequential overlap preserving conversation flow

        Args:
            focused_text: The focused (relevant) conversation content
            chunk_text: The conversation chunk to evaluate

        Returns:
            Dictionary with overlap metrics
        """
        return {
            "jaccard_similarity": self.jaccard_similarity(focused_text, chunk_text),
            "rouge_l_similarity": self.rouge_l_similarity(focused_text, chunk_text),
            "token_containment": self.token_containment(focused_text, chunk_text),
            "asymmetric_jaccard": self.asymmetric_jaccard(focused_text, chunk_text),
        }


class ConversationChunker:
    """
    Conversation chunker optimized for LongMemEval dataset.

    Creates overlapping chunks from LongMemEval conversation data,
    preserving conversation structure and maintaining context windows
    suitable for retrieval evaluation.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        """
        Initialize chunker with LongMemEval-appropriate parameters.

        Args:
            chunk_size: Target chunk size in tokens (500 works well for LongMemEval)
            overlap: Token overlap between chunks (50 tokens = ~12% overlap)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_conversation(self, conversation_text: str, custom_id: str) -> list[dict[str, Any]]:
        """
        Split LongMemEval conversation into overlapping chunks.

        Designed specifically for LongMemEval's conversation format,
        maintaining conversation flow and creating meaningful chunks
        for similarity search.

        Args:
            conversation_text: Full LongMemEval conversation prompt
            custom_id: LongMemEval custom_id for chunk identification

        Returns:
            List of chunk dictionaries with metadata
        """

        # Tokenize the conversation
        # Use disallowed_special=() to treat special tokens (like <|endoftext|>) as regular text
        tokens = self.encoding.encode(conversation_text, disallowed_special=())

        chunks = []
        chunk_id = 0

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                {
                    "id": f"{custom_id}_chunk_{chunk_id}",
                    "content": chunk_text,
                    "token_count": len(chunk_tokens),
                    "start_token": i,
                    "end_token": i + len(chunk_tokens),
                    "source_conversation": custom_id,
                    "chunk_position": chunk_id,  # Useful for LongMemEval analysis
                }
            )

            chunk_id += 1

            # Break if we've processed all tokens
            if i + self.chunk_size >= len(tokens):
                break

        return chunks


class GroundTruthExtractor:
    """
    Ground truth extraction system for LongMemEval dataset.

    This class is SPECIFICALLY designed for LongMemEval's structure:
    - Uses focused vs full conversation pairs to identify relevant chunks
    - Leverages ChromaDB for efficient similarity search
    - Applies LongMemEval-appropriate similarity thresholds
    - Generates ground truth suitable for retrieval evaluation

    NOT suitable for other datasets without modification.
    """

    def __init__(
        self, chroma_path: str = "chroma_longmemeval", embedding_dimensions: int = 768, embedding_model: str = "openai"
    ):
        """
        Initialize extractor for LongMemEval dataset with per-conversation collections.

        Args:
            chroma_path: ChromaDB storage path (defaults to LongMemEval-specific path)
            embedding_dimensions: Embedding dimensions (768 recommended for quality/cost balance)
            embedding_model: Model type - 'openai', 'sentence-transformer', or 'huggingface'
        """
        import os

        self.chroma_path = chroma_path
        self.embedding_dimensions = embedding_dimensions
        self.embedding_model = embedding_model
        self.client: chromadb.ClientAPI = chromadb.PersistentClient(path=chroma_path)
        self.chunker = ConversationChunker()
        self.overlap_metrics = ContentOverlapMetrics()
        self.prompt_cleaner = LongMemEvalPromptCleaner()
        self.openai_client = OpenAIClient(api_key=os.environ["ORQ_API_KEY"], base_url=os.environ["ORQ_BASE_URL"])

        # Store collections per conversation (lazy loading)
        self.collections: dict[str, Any] = {}
        self.collection_prefix = "longmemeval_conv_"

        console.print("[blue]Initialized LongMemEval ground truth extractor (per-conversation collections)[/blue]")
        console.print(f"[dim]ChromaDB path: {chroma_path}[/dim]")
        console.print(f"[dim]Collection strategy: separate collection per conversation[/dim]")
        console.print(f"[dim]Target dimensions: {embedding_dimensions}d[/dim]")

    def _create_embedding_function(
        self,
    ) -> OpenAIEmbeddingFunction | SentenceTransformerEmbeddingFunction | HuggingFaceEmbeddingFunction:
        """Create embedding function based on specified model type."""
        import os

        from chromadb.utils import embedding_functions

        if self.embedding_model == "openai":
            # OpenAI embeddings with API batching support
            openai_api_key = os.getenv("ORQ_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or ORQ_API_KEY environment variable.")

            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=self._get_model_name(),  # From centralized config
                dimensions=self.embedding_dimensions,
                api_base=os.getenv("ORQ_BASE_URL"),
                # OpenAI API handles batching automatically for efficiency
            )

        if self.embedding_model == "sentence-transformer":
            # Local sentence-transformers with GPU support
            console.print("[blue]🔧 Configuring sentence-transformer for GPU acceleration...[/blue]")

            # Warn if dimensions exceed sentence-transformer limits
            if self.embedding_dimensions > 768:
                console.print(
                    f"[yellow]⚠️  Warning: Sentence-transformer max dimensions is 768, requested {self.embedding_dimensions}[/yellow]"
                )

            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._get_model_name(),  # From centralized config
                device="cuda" if self._has_cuda() else ("mps" if self._has_mps() else "cpu"),
                # Batching handled by sentence-transformers internally
            )

        if self.embedding_model == "huggingface":
            # HuggingFace embeddings with GPU support
            console.print("[blue]🔧 Configuring HuggingFace model for GPU acceleration...[/blue]")

            return embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=os.getenv("HUGGINGFACE_API_KEY") or "",
                model_name=self._get_model_name(),  # From centralized config
            )

        raise ValueError(
            f"Unsupported embedding model: {self.embedding_model}. Use 'openai', 'sentence-transformer', or 'huggingface'"
        )

    def _get_model_config(self) -> dict[str, str]:
        """Get centralized model configuration for the specified embedding type and dimensions."""
        if self.embedding_model == "openai":
            return {
                "model_name": "azure/text-embedding-3-small",
                "provider": "openai",
                "description": "Azure OpenAI text-embedding-3-small via ORQ proxy",
            }
        if self.embedding_model == "sentence-transformer":
            if self.embedding_dimensions <= 384:
                return {
                    "model_name": "all-MiniLM-L6-v2",
                    "provider": "sentence-transformer",
                    "description": "Local sentence-transformer (384d, fast)",
                }
            return {
                "model_name": "all-mpnet-base-v2",
                "provider": "sentence-transformer",
                "description": "Local sentence-transformer (768d, balanced)",
            }
        if self.embedding_model == "huggingface":
            return {
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "provider": "huggingface",
                "description": "HuggingFace API sentence-transformer",
            }
        return {
            "model_name": "unknown",
            "provider": "unknown",
            "description": f"Unsupported model type: {self.embedding_model}",
        }

    def _get_model_name(
        self,
    ) -> Literal[
        "azure/text-embedding-3-small",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "unknown",
    ]:
        """Get the actual model name being used."""
        return self._get_model_config()["model_name"]

    def _has_cuda(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except (ImportError, AttributeError, RuntimeError):
            return False

    def _has_mps(self):
        """Check if MacOS MPS GPU is available."""
        try:
            import torch

            return torch.backends.mps.is_available()
        except (ImportError, AttributeError, RuntimeError):
            return False

    def _get_or_create_collection(self, conversation_id: str):
        """Get existing collection or create new one for a specific conversation."""
        collection_name = f"{self.collection_prefix}{conversation_id}"
        
        # Return cached collection if available
        if collection_name in self.collections:
            return self.collections[collection_name]
            
        embedding_function = self._create_embedding_function()

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={
                "description": f"LongMemEval conversation {conversation_id} chunks for ground truth extraction",
                "dataset": "LongMemEval",
                "conversation_id": conversation_id,
                "chunk_size": self.chunker.chunk_size,
                "overlap": self.chunker.overlap,
                "embedding_provider": embedding_function.__class__.__name__,
                "embedding_model": self._get_model_name(),
                "embedding_dimensions": self.embedding_dimensions,
                "embedding_type": self.embedding_model,
                "created_timestamp": pd.Timestamp.now().isoformat(),
            },
        )

        # Cache the collection
        self.collections[collection_name] = collection

        # Check if it's an existing collection and warn about potential mismatches
        existing_model = collection.metadata.get("embedding_model", "unknown")
        existing_dims = collection.metadata.get("embedding_dimensions", "unknown")
        existing_type = collection.metadata.get("embedding_type", "unknown")

        if existing_model != "unknown" and len(self.collections) == 1:  # Only show for first collection
            console.print(f"[blue]✓ Using existing ChromaDB collections for conversations[/blue]")
            console.print(f"[dim]Existing model: {existing_model} ({existing_dims}d)[/dim]")

            # Check for dimension mismatch
            if str(existing_dims) != str(self.embedding_dimensions):
                console.print(
                    f"[yellow]⚠️  Warning: Requested dimensions ({self.embedding_dimensions}) differs from existing ({existing_dims})[/yellow]"
                )
                console.print("[yellow]Use --force-reingest to recreate with new dimensions[/yellow]")

            # Check for model type mismatch
            if existing_type != self.embedding_model:
                console.print(
                    f"[yellow]⚠️  Warning: Requested model type ({self.embedding_model}) differs from existing ({existing_type})[/yellow]"
                )
                console.print("[yellow]Use --force-reingest to recreate with new model type[/yellow]")
        elif existing_model == "unknown" and len(self.collections) == 1:  # Only show for first collection
            console.print(f"[green]✓ Creating new ChromaDB collections with {self.embedding_model} embeddings[/green]")
            console.print(f"[dim]Model: {self._get_model_name()} ({self.embedding_dimensions}d)[/dim]")

        return collection

    def _extract_retry_delay_from_error(self, error_message: str, base_delay: float, attempt: int) -> float:
        """Extract retry delay from error message, fallback to exponential backoff."""
        import random
        
        patterns = [
            r"retry after (\d+) seconds?",
            r"wait (\d+) seconds?",
            r"retry in (\d+) seconds?",
            r"after (\d+)s",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # No specific retry time found, use exponential backoff with jitter
        return base_delay * (2 ** attempt) + random.uniform(0, 1)

    def _retry_with_backoff(self, func, *args, max_retries=5, base_delay=1, **kwargs):
        """Retry function with exponential backoff for API stability and 429 rate limiting."""
        import random

        import openai

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError as e:
                # Special handling for 429 rate limit errors
                if attempt == max_retries - 1:
                    console.print(f"[red]❌ Rate limit exceeded after {max_retries} attempts[/red]")
                    raise e

                # Extract retry time from error message: "Please retry after 60 seconds"
                retry_delay = self._extract_retry_delay_from_error(str(e), base_delay, attempt)
                
                console.print(f"[yellow]⚠️  Rate limit hit (429) - attempt {attempt + 1}/{max_retries}[/yellow]")
                if "retry after" in str(e).lower() or "wait" in str(e).lower():
                    console.print(f"[dim]Server requested retry after: {retry_delay}s[/dim]")
                else:
                    console.print(f"[dim]Using backoff delay: {retry_delay:.1f}s[/dim]")

                time.sleep(retry_delay)

            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                # Network/timeout errors - shorter delay
                if attempt == max_retries - 1:
                    console.print(f"[red]❌ API connection failed after {max_retries} attempts[/red]")
                    raise e

                delay = base_delay * (1.5**attempt)
                console.print(f"[yellow]⚠️  API connection issue (attempt {attempt + 1}/{max_retries}): {e}[/yellow]")
                console.print(f"[dim]Retrying in {delay:.1f}s...[/dim]")
                time.sleep(delay)

            except openai.APIError as e:
                # Other API errors
                if "rate" in str(e).lower() or "429" in str(e):
                    # Treat as rate limit even if not caught above
                    if attempt == max_retries - 1:
                        console.print(f"[red]❌ Rate limiting error after {max_retries} attempts[/red]")
                        raise e

                    # Extract retry time from error message
                    retry_delay = self._extract_retry_delay_from_error(str(e), base_delay, attempt)
                    
                    console.print(f"[yellow]⚠️  Rate limiting detected - attempt {attempt + 1}/{max_retries}[/yellow]")
                    if "retry after" in str(e).lower() or "wait" in str(e).lower():
                        console.print(f"[dim]Server requested retry after: {retry_delay}s[/dim]")
                    else:
                        console.print(f"[dim]Using backoff delay: {retry_delay:.1f}s[/dim]")

                    time.sleep(retry_delay)
                else:
                    # Non-retryable API error
                    console.print(f"[red]❌ Non-retryable API error: {e}[/red]")
                    raise e

            except Exception as e:
                # Generic errors - use original logic
                if attempt == max_retries - 1:
                    console.print(f"[red]❌ Unexpected error after {max_retries} attempts: {e}[/red]")
                    raise e
                delay = base_delay * (2**attempt)
                console.print(f"[yellow]⚠️  API call failed (attempt {attempt + 1}/{max_retries}): {e}[/yellow]")
                console.print(f"[dim]Retrying in {delay:.1f}s...[/dim]")
                time.sleep(delay)

    def validate_collections_health(self, conversation_ids: list[str]) -> bool:
        """Validate that conversation collections are healthy and accessible."""
        try:
            total_docs = 0
            healthy_collections = 0
            
            # Sample a few conversations for health check
            sample_ids = conversation_ids[:min(5, len(conversation_ids))]
            
            for conv_id in sample_ids:
                try:
                    collection = self._get_or_create_collection(conv_id)
                    count = collection.count()
                    total_docs += count
                    healthy_collections += 1
                except Exception as e:
                    console.print(f"[yellow]⚠️  Collection {conv_id} health issue: {e}[/yellow]")
                    
            console.print(f"[green]✓ Collection health check passed ({healthy_collections}/{len(sample_ids)} collections)[/green]")
            console.print(f"[dim]Sample documents: {total_docs}[/dim]")
            
            return healthy_collections == len(sample_ids)

        except Exception as e:
            console.print(f"[red]❌ Collection health check failed: {e}[/red]")
            console.print("[yellow]Consider using --force-reingest to recreate the collections[/yellow]")
            return False

    def save_ingestion_checkpoint(
        self,
        checkpoint_path: str,
        batch_num: int,
        total_batches: int,
        processed_conversations: list,
        remaining_conversations: list,
        total_chunks_ingested: int,
    ) -> None:
        """Save ingestion checkpoint to allow resuming from failures."""
        checkpoint_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "collection_name": self.collection_name,
            "chroma_path": self.chroma_path,
            "current_batch": batch_num,
            "total_batches": total_batches,
            "processed_conversation_ids": [conv["custom_id"] for conv in processed_conversations],
            "remaining_conversation_ids": [conv["custom_id"] for conv in remaining_conversations],
            "total_chunks_ingested": total_chunks_ingested,
            "collection_count": self.collection.count(),
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
        }

        # Save checkpoint
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_file = checkpoint_file.with_suffix(checkpoint_file.suffix + ".tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_file.rename(checkpoint_file)
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

        console.print(
            f"[dim]💾 Checkpoint saved: batch {batch_num}/{total_batches} ({total_chunks_ingested} chunks)[/dim]"
        )

    def load_ingestion_checkpoint(self, checkpoint_path: str) -> dict | None:
        """Load ingestion checkpoint if it exists and is valid."""
        checkpoint_file = Path(checkpoint_path)

        if not checkpoint_file.exists():
            return None

        try:
            with checkpoint_file.open("r") as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint compatibility
            if (
                checkpoint_data.get("collection_name") != self.collection_name
                or checkpoint_data.get("chroma_path") != self.chroma_path
            ):
                console.print("[yellow]⚠️  Checkpoint is for different collection, ignoring[/yellow]")
                return None

            # Check if collection count matches checkpoint
            current_count = self.collection.count()
            expected_count = checkpoint_data.get("collection_count", 0)

            if current_count != expected_count:
                console.print(
                    f"[yellow]⚠️  Collection count mismatch: {current_count} vs {expected_count}, starting fresh[/yellow]"
                )
                return None

            console.print(
                f"[green]📂 Found valid checkpoint: batch {checkpoint_data['current_batch']}/{checkpoint_data['total_batches']}[/green]"
            )
            console.print(f"[dim]Processed: {len(checkpoint_data['processed_conversation_ids'])} conversations[/dim]")

            return checkpoint_data

        except Exception as e:
            console.print(f"[yellow]⚠️  Error loading checkpoint: {e}, starting fresh[/yellow]")
            return None

    def cleanup_checkpoint(self, checkpoint_path: str) -> None:
        """Remove checkpoint file after successful completion."""
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            console.print("[dim]🗑️  Checkpoint file cleaned up[/dim]")

    def ingest_longmemeval_conversations(
        self,
        full_conversations_df: pd.DataFrame,
        test_mode: bool = False,
        num_docs: int | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        """
        Ingest LongMemEval full conversations into ChromaDB.

        This method expects the LongMemEval CSV format with:
        - custom_id: Unique conversation identifier
        - full_prompt: Complete conversation with distractors
        - Additional columns are preserved in metadata

        Args:
            full_conversations_df: DataFrame with LongMemEval full conversation data
            test_mode: If True, process only 1 document for testing
            num_docs: Custom number of documents to process (overrides test_mode)
            checkpoint_path: Path to checkpoint file for resuming interrupted ingestion
        """

        # Handle test mode and document limits
        original_count = len(full_conversations_df)
        if test_mode and num_docs is None:
            full_conversations_df = full_conversations_df.head(1)
            console.print(f"[yellow]🧪 Test mode: Processing only 1 document (out of {original_count})[/yellow]")
        elif num_docs is not None:
            full_conversations_df = full_conversations_df.head(num_docs)
            console.print(
                f"[yellow]📊 Limited mode: Processing {len(full_conversations_df)} documents (out of {original_count})[/yellow]"
            )

        # Check for existing checkpoint
        checkpoint_data = None
        if checkpoint_path:
            checkpoint_data = self.load_ingestion_checkpoint(checkpoint_path)

        if checkpoint_data:
            # Resume from checkpoint
            processed_ids = set(checkpoint_data["processed_conversation_ids"])
            remaining_conversations = full_conversations_df[~full_conversations_df["custom_id"].isin(processed_ids)]
            console.print(
                f"[green]🔄 Resuming from checkpoint: {len(remaining_conversations)} conversations remaining[/green]"
            )
            full_conversations_df = remaining_conversations

        console.print(f"[blue]Ingesting {len(full_conversations_df)} LongMemEval conversations...[/blue]")

        # Validate expected LongMemEval columns
        required_columns = ["custom_id", "full_prompt"]
        missing_columns = [col for col in required_columns if col not in full_conversations_df.columns]
        if missing_columns:
            raise ValueError(f"LongMemEval ingestion requires columns: {missing_columns}")

        all_chunks = []
        all_ids = []
        all_metadatas = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console
        ) as progress:
            task = progress.add_task("Processing LongMemEval conversations...", total=len(full_conversations_df))

            for _, row in full_conversations_df.iterrows():
                custom_id = str(row["custom_id"])
                full_prompt = str(row["full_prompt"])

                # Clean the prompt to extract only conversation data
                conversation_content = self.prompt_cleaner.clean_prompt(full_prompt)

                # Validate cleaning (sample first few for debugging)
                if len(all_chunks) < 3:  # Only validate first few samples to avoid spam
                    is_valid = self.prompt_cleaner.validate_cleaning(full_prompt, conversation_content, custom_id)
                    if is_valid:
                        console.print(
                            f"[dim]✓ Cleaned {custom_id}: {len(full_prompt)} → {len(conversation_content)} chars[/dim]"
                        )

                # Chunk the cleaned conversation
                chunks = self.chunker.chunk_conversation(conversation_content, custom_id)

                # Prepare for batch insertion with LongMemEval-specific metadata
                for chunk in chunks:
                    all_chunks.append(chunk["content"])
                    all_ids.append(chunk["id"])

                    metadata = {
                        "source_conversation": custom_id,
                        "token_count": chunk["token_count"],
                        "start_token": chunk["start_token"],
                        "end_token": chunk["end_token"],
                        "chunk_position": chunk["chunk_position"],
                        "dataset": "LongMemEval",
                    }

                    # Add original row data to metadata (excluding large text fields)
                    for col in full_conversations_df.columns:
                        if col not in ["full_prompt", "focused_prompt"] and col in row:
                            metadata[f"original_{col}"] = str(row[col])

                    all_metadatas.append(metadata)

                progress.advance(task)

        # Group chunks by conversation for separate collections
        console.print(f"[green]📤 Creating separate collections and embedding {len(all_chunks)} total chunks...[/green]")
        
        # Group chunks by source conversation
        chunks_by_conversation: dict[str, dict] = {}
        for chunk_text, chunk_id, metadata in zip(all_chunks, all_ids, all_metadatas, strict=True):
            conv_id = metadata["source_conversation"]
            if conv_id not in chunks_by_conversation:
                chunks_by_conversation[conv_id] = {
                    "texts": [],
                    "ids": [],
                    "metadatas": []
                }
            chunks_by_conversation[conv_id]["texts"].append(chunk_text)
            chunks_by_conversation[conv_id]["ids"].append(chunk_id)
            chunks_by_conversation[conv_id]["metadatas"].append(metadata)
        
        total_conversations = len(chunks_by_conversation)
        console.print(f"[dim]💡 Will create {total_conversations} separate collections, one per conversation[/dim]")

        # Provide model-specific information
        if self.embedding_model == "openai":
            console.print("[dim]💡 Using OpenAI API embeddings - optimized for batching and parallelism[/dim]")
        elif self.embedding_model == "sentence-transformer":
            gpu_info = "MPS GPU" if self._has_mps() else ("CUDA GPU" if self._has_cuda() else "CPU")
            console.print(
                f"[dim]💡 Using local sentence-transformer on {gpu_info} - may take time for large datasets[/dim]"
            )
        else:
            console.print("[dim]💡 Using HuggingFace API embeddings - optimized for batching[/dim]")

        # Process each conversation separately
        processed_conversations = 0
        total_chunks_ingested = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            conversation_task = progress.add_task(f"Processing {total_conversations} conversations...", total=total_conversations)

            for conv_id, conv_data in chunks_by_conversation.items():
                try:
                    # Get or create collection for this conversation
                    collection = self._get_or_create_collection(conv_id)
                    
                    # Check if conversation already has chunks
                    existing_count = collection.count()
                    if existing_count > 0:
                        console.print(f"[dim]⏩ Conversation {conv_id} already has {existing_count} chunks, skipping[/dim]")
                        processed_conversations += 1
                        progress.advance(conversation_task)
                        continue
                    
                    conv_chunks = len(conv_data["texts"])
                    progress.update(
                        conversation_task,
                        description=f"Processing conversation {conv_id} ({conv_chunks} chunks)...",
                    )

                    # Embed and insert chunks for this conversation
                    embeddings = collection._embedding_function(conv_data["texts"])

                    collection.add(
                        documents=conv_data["texts"],
                        ids=conv_data["ids"],
                        metadatas=conv_data["metadatas"],
                        embeddings=embeddings,
                    )

                    total_chunks_ingested += conv_chunks
                    processed_conversations += 1
                    
                    # Explicit memory cleanup
                    del embeddings
                    
                    progress.advance(conversation_task)

                except Exception as e:
                    console.print(f"[red]❌ Error processing conversation {conv_id}: {e}[/red]")
                    raise RuntimeError(f"Conversation ingestion failed for {conv_id}: {e}") from e

        console.print(
            f"[green]✓ Ingested {total_chunks_ingested} chunks from {processed_conversations} LongMemEval conversations into separate collections[/green]"
        )

        # Clean up checkpoint file after successful completion
        if checkpoint_path:
            self.cleanup_checkpoint(checkpoint_path)

    def extract_longmemeval_ground_truth(
        self,
        focused_df: pd.DataFrame,
        similarity_threshold: float = 0.35,  # Lowered for better recall
        max_chunks_per_question: int = 10,
        primary_metric: str = "rouge_l",  # 'jaccard', 'rouge_l', or 'token_containment' (rouge_l preserves sequence order)
        query_strategy: str = "combined",  # 'focused', 'question', 'answer', 'combined'
        debug_mode: bool = False,
        limit_questions: int | None = None,  # Limit number of questions to process
    ) -> dict[str, Any]:
        """
        Extract ground truth chunk relevance for LongMemEval questions.

        Uses various query strategies to identify relevant chunks
        in the full conversation via similarity search and overlap metrics.

        Args:
            focused_df: DataFrame with LongMemEval focused conversation data
            similarity_threshold: Minimum similarity for relevance (0.35 for better recall)
            max_chunks_per_question: Maximum relevant chunks per question
            primary_metric: Primary overlap metric ('rouge_l' recommended for sequence order, 'jaccard', 'token_containment')
            query_strategy: How to construct queries:
                - 'focused': Use entire focused conversation content
                - 'question': Use only the question text
                - 'answer': Use question + answer
                - 'combined': Use question for query, focused for validation
            debug_mode: If True, log similarity scores for analysis

        Returns:
            Ground truth data structure with relevant chunks per question.
            Each question includes:
                - focused_content: The target content we're looking for (ground truth)
                - focused_content_length: Character count of focused content
                - focused_content_tokens: Token count of focused content
                - relevant_chunks: List of chunks found via similarity search
                
            Each relevant chunk includes:
                - chunk_id: ChromaDB chunk identifier
                - content: Full chunk text for validation
                - content_preview: First 150 characters  
                - content_length: Character count
                - content_tokens: Token count
                - chromadb_similarity_score: Embedding similarity (0-1)
                - overlap_metrics: Jaccard/ROUGE scores
                - primary_score: Score used for ranking
        """

        # Apply limit if specified with bounds checking
        if limit_questions is not None:
            if limit_questions <= 0:
                raise ValueError(f"limit_questions must be positive, got {limit_questions}")
            if limit_questions > len(focused_df):
                console.print(
                    f"[yellow]⚠️  Requested {limit_questions} questions but only {len(focused_df)} available[/yellow]"
                )
            focused_df = focused_df.head(limit_questions)
            console.print(f"[yellow]📊 Processing limited subset: {len(focused_df)} of original questions[/yellow]")

        # Validate ROUGE-L availability if selected
        if primary_metric == "rouge_l" and not ROUGE_AVAILABLE:
            console.print("[red]❌ ROUGE-L selected but rouge_score package not available[/red]")
            console.print("[yellow]💡 Install with: pip install rouge_score[/yellow]")
            console.print("[yellow]💡 Or use alternative metrics: jaccard, token_containment[/yellow]")
            raise ImportError("rouge_score package required for ROUGE-L metric")

        console.print(f"[blue]Extracting LongMemEval ground truth for {len(focused_df)} questions...[/blue]")
        console.print(f"[dim]Using primary metric: {primary_metric} (threshold: {similarity_threshold})[/dim]")
        console.print(f"[dim]Query strategy: {query_strategy}[/dim]")

        # Validate LongMemEval format
        required_columns = ["custom_id", "focused_prompt", "question"]
        missing_columns = [col for col in required_columns if col not in focused_df.columns]
        if missing_columns:
            raise ValueError(f"LongMemEval focused data requires columns: {missing_columns}")

        ground_truth_data = {
            "metadata": {
                "dataset": "LongMemEval",
                "similarity_threshold": similarity_threshold,
                "max_chunks_per_question": max_chunks_per_question,
                "total_questions": len(focused_df),
                "primary_metric": primary_metric,
                "query_strategy": query_strategy,
                "extraction_method": "chromadb_similarity_longmemeval",
                "rouge_available": ROUGE_AVAILABLE,
                "debug_mode": debug_mode,
                "includes_full_content": True,  # Full chunk content included for validation
                "includes_focused_content": True,  # Focused conversation content included as ground truth target
                "content_fields": ["content", "content_preview", "content_length", "content_tokens"],
                "focused_fields": ["focused_content", "focused_content_length", "focused_content_tokens"],
            },
            "questions": [],
        }

        # Initialize statistics for progress tracking
        stats = {"processed": 0, "found_chunks": 0, "no_chunks": 0, "total_relevant_chunks": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[stats]}[/cyan]"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Extracting LongMemEval ground truth...", total=len(focused_df), stats="Starting..."
            )

            for _, row in focused_df.iterrows():
                custom_id = str(row["custom_id"])
                focused_prompt = str(row["focused_prompt"])
                question = str(row["question"])
                answer = str(row.get("answer", ""))

                # Clean the focused prompt to extract only conversation data
                focused_content = self.prompt_cleaner.clean_prompt(focused_prompt)

                # Construct query based on strategy
                if query_strategy == "question":
                    query_text = question
                elif query_strategy == "answer":
                    query_text = f"{question} {answer}"
                elif query_strategy == "combined":
                    # Use question for retrieval, focused content for validation
                    query_text = question
                else:  # 'focused' or default
                    query_text = focused_content

                if debug_mode:
                    console.print(f"[dim]Query strategy: {query_strategy}, Query length: {len(query_text)} chars[/dim]")

                # Get collection for this specific conversation
                try:
                    collection = self._get_or_create_collection(custom_id)
                except Exception as e:
                    console.print(f"[yellow]⚠ Could not access collection for conversation {custom_id}: {e}[/yellow]")
                    stats["no_chunks"] += 1
                    progress.update(
                        task,
                        stats=f"❌ {stats['no_chunks']} no chunks | ✓ {stats['found_chunks']} found | 📊 {stats['total_relevant_chunks']} relevant",
                    )
                    progress.advance(task)
                    continue
                    
                # Query ChromaDB using the selected strategy (no where clause needed since each collection is per conversation)
                results = collection.query(
                    query_texts=[query_text],
                    n_results=max_chunks_per_question * 5,  # Get more candidates for filtering
                )

                if not results["documents"] or not results["documents"][0]:
                    console.print(f"[yellow]⚠ No chunks found for LongMemEval conversation {custom_id}[/yellow]")
                    stats["no_chunks"] += 1
                    progress.update(
                        task,
                        stats=f"❌ {stats['no_chunks']} no chunks | ✓ {stats['found_chunks']} found | 📊 {stats['total_relevant_chunks']} relevant",
                    )
                    progress.advance(task)
                    continue

                stats["found_chunks"] += 1

                # Process retrieved chunks with LongMemEval-specific logic
                relevant_chunks = []
                all_scores = []  # For debug analysis

                for i, (chunk_text, chunk_id, distance) in enumerate(
                    zip(results["documents"][0], results["ids"][0], results["distances"][0], strict=False)
                ):
                    # Calculate similarity score (ChromaDB uses L2 distance)
                    similarity_score = 1 / (1 + distance)

                    # For combined strategy, use focused content for validation
                    validation_text = focused_content if query_strategy == "combined" else query_text

                    # Calculate overlap metrics
                    overlap_metrics = self.overlap_metrics.calculate_primary_metrics(validation_text, chunk_text)

                    # Use primary metric for relevance determination
                    primary_score = overlap_metrics.get(f"{primary_metric}_similarity", similarity_score)

                    if debug_mode:
                        all_scores.append(
                            {
                                "chunk_id": chunk_id,
                                "embedding_sim": similarity_score,
                                "jaccard": overlap_metrics.get("jaccard_similarity", 0),
                                "token_containment": overlap_metrics.get("token_containment", 0),
                                "primary_score": primary_score,
                            }
                        )

                    if primary_score >= similarity_threshold:
                        relevant_chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "chromadb_similarity_score": similarity_score,
                                "chromadb_distance": distance,
                                "overlap_metrics": overlap_metrics,
                                "primary_score": primary_score,
                                "content_preview": chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text,
                                "content": chunk_text,  # Full chunk content for validation
                                "content_length": len(chunk_text),
                                "content_tokens": len(self.chunker.encoding.encode(chunk_text, disallowed_special=())),
                                "is_relevant": True,
                            }
                        )

                # Debug: Show score distribution
                if debug_mode and all_scores:
                    top_scores = sorted(all_scores, key=lambda x: x["primary_score"], reverse=True)[:5]
                    console.print(f"[dim]Top 5 scores for {custom_id}:[/dim]")
                    for score in top_scores:
                        console.print(
                            f"[dim]  - {score['chunk_id']}: {primary_metric}={score['primary_score']:.3f}, "
                            f"jaccard={score['jaccard']:.3f}, token_cont={score['token_containment']:.3f}[/dim]"
                        )

                # Sort by primary metric and take top chunks
                relevant_chunks.sort(key=lambda x: x["primary_score"], reverse=True)
                relevant_chunks = relevant_chunks[:max_chunks_per_question]

                # Update statistics
                stats["total_relevant_chunks"] += len(relevant_chunks)
                stats["processed"] += 1

                ground_truth_data["questions"].append(
                    {
                        "custom_id": custom_id,
                        "question": question,
                        "expected_answer": answer,
                        "focused_content": focused_content,  # The actual target content we're looking for
                        "focused_content_tokens": len(
                            self.chunker.encoding.encode(focused_content, disallowed_special=())
                        ),
                        "focused_content_length": len(focused_content),
                        "relevant_chunks": relevant_chunks,
                        "total_chunks_found": len(results["ids"][0]) if results["ids"] else 0,
                        "dataset_source": "LongMemEval",
                    }
                )

                # Update progress with current statistics
                progress.update(
                    task,
                    stats=f"✓ {stats['found_chunks']} found | 📊 {stats['total_relevant_chunks']} relevant | ❌ {stats['no_chunks']} no chunks",
                )
                progress.advance(task)

        # Display completion summary
        console.print("[green]🎉 Ground truth extraction completed![/green]")
        console.print("[dim]📊 Summary:[/dim]")
        console.print(f"[dim]  • Total questions processed: {stats['processed']}[/dim]")
        console.print(f"[dim]  • Questions with chunks found: {stats['found_chunks']}[/dim]")
        console.print(f"[dim]  • Questions without chunks: {stats['no_chunks']}[/dim]")
        console.print(f"[dim]  • Total relevant chunks identified: {stats['total_relevant_chunks']}[/dim]")
        if stats["found_chunks"] > 0:
            avg_relevant = stats["total_relevant_chunks"] / stats["found_chunks"]
            console.print(f"[dim]  • Average relevant chunks per question: {avg_relevant:.2f}[/dim]")

        return ground_truth_data

    def save_longmemeval_ground_truth(self, ground_truth_data: dict[str, Any], output_path: str):
        """
        Save LongMemEval ground truth data with dataset-specific summary.

        Args:
            ground_truth_data: Ground truth data structure
            output_path: Output JSON file path
        """

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)

        # Generate LongMemEval-specific summary
        questions_with_chunks = sum(1 for q in ground_truth_data["questions"] if q["relevant_chunks"])
        total_relevant_chunks = sum(len(q["relevant_chunks"]) for q in ground_truth_data["questions"])
        avg_chunks_per_question = total_relevant_chunks / questions_with_chunks if questions_with_chunks > 0 else 0

        summary_panel = Panel.fit(
            f"[bold green]✓ LongMemEval Ground Truth Extraction Complete[/bold green]\n"
            f"[dim]Dataset:[/dim] [cyan]LongMemEval[/cyan]\n"
            f"[dim]Questions processed:[/dim] [yellow]{ground_truth_data['metadata']['total_questions']}[/yellow]\n"
            f"[dim]Questions with relevant chunks:[/dim] [magenta]{questions_with_chunks}[/magenta]\n"
            f"[dim]Total relevant chunks identified:[/dim] [blue]{total_relevant_chunks}[/blue]\n"
            f"[dim]Average chunks per question:[/dim] [green]{avg_chunks_per_question:.1f}[/green]\n"
            f"[dim]Primary similarity metric:[/dim] [cyan]{ground_truth_data['metadata']['primary_metric']}[/cyan]\n"
            f"[dim]Similarity threshold:[/dim] [yellow]{ground_truth_data['metadata']['similarity_threshold']}[/yellow]\n"
            f"[dim]Full content included:[/dim] [green]✓ Yes (for validation)[/green]\n"
            f"[dim]Focused content included:[/dim] [green]✓ Yes (ground truth target)[/green]\n"
            f"[dim]Output saved to:[/dim] [blue]{output_path}[/blue]",
            title="📊 LongMemEval Ground Truth Summary",
            border_style="green",
        )
        console.print(summary_panel)

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the LongMemEval ChromaDB collections."""
        
        total_chunks = 0
        total_collections = len(self.collections)
        
        # Count chunks across all loaded collections
        for collection in self.collections.values():
            total_chunks += collection.count()
        
        return {
            "collection_strategy": "per-conversation",
            "total_collections": total_collections,
            "total_chunks": total_chunks,
            "avg_chunks_per_collection": total_chunks / total_collections if total_collections > 0 else 0,
            "chunk_size": self.chunker.chunk_size,
            "overlap": self.chunker.overlap,
            "dataset": "LongMemEval",
        }
