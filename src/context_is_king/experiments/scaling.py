"""
Context Window Scaling Experiment Module

This module provides tools to measure how LLM call duration scales with context window size.
Tests multiple models across different token sizes to understand performance characteristics.
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tiktoken
from openai import OpenAI
from loguru import logger


@dataclass
class ExperimentResult:
    """Results from a single LLM timing experiment."""

    model_name: str
    context_size: int
    duration_seconds: float
    tokens_per_second: float
    success: bool
    error_message: str = None
    timestamp: str = None
    iteration: int = None
    response_text: str = None
    response_tokens: int = None
    response_length: int = None
    # New streaming metrics
    time_to_first_token: float = None  # Most important metric for user experience
    total_generation_time: float = None  # Time from first token to completion
    tokens_per_second_generation: float = None  # Generation speed (tokens/sec after TTFT)
    streaming_chunks: int = None  # Number of streaming chunks received


@dataclass
class ExperimentCheckpoint:
    """Checkpoint data for resuming experiments."""

    experiment_id: str
    completed_results: list
    remaining_tasks: list  # List of (model_name, model_id, token_size, iteration) tuples
    experiment_config: dict
    timestamp: str


class ContextWindowExperiment:
    """Main class for running context window scaling experiments."""

    # Model configurations with exact ORQ proxy names and token limits
    MODELS = {  # noqa: RUF012
        "gpt-5": "azure/gpt-5-chat",
        "gpt-5-mini": "azure/gpt-5-mini",
        "gpt-5-nano": "azure/gpt-5-nano",
        # "gpt-4.1-mini": "azure/gpt-4.1-mini",
        # "gpt-4.1-nano": "azure/gpt-4.1-nano",
        "gemini-2.5-pro": "google-ai/gemini-2.5-pro",
        "gemini-2.5-flash": "google-ai/gemini-2.5-flash",
        "gemini-2.5-flash-lite": "google-ai/gemini-2.5-flash-lite",
        # "claude-sonnet-4": "google/claude-sonnet-4@20250514",
        "claude-sonnet-4": "google/claude-sonnet-4@20250514",
        "claude-opus-4.1": "anthropic/claude-opus-4-1-20250805",
        # "claude-sonnet-3.7": "google/claude-3-7-sonnet@20250219",
        # "claude-sonnet-3.5": "google/claude-3-5-sonnet-v2@20241022",
        "claude-haiku-3.5": "google/claude-3-5-haiku@20241022",
    }

    # Model token limits (context window size limits)
    # Sources: OpenAI docs, Anthropic docs, Google docs as of 2024-2025
    MODEL_TOKEN_LIMITS = {  # noqa: RUF012
        "gpt-5": 272_000,
        "gpt-5-mini": 272000,  # GPT-5 mini supports 1M tokens
        "gpt-5-nano": 272000,  # GPT-5 nano has a max of 272k tokens
        "gpt-4.1-mini": 1000000,  # GPT-4.1 mini supports 1M tokens (confirmed)
        "gpt-4.1-nano": 1000000,  # GPT-4.1 nano supports 1M tokens (confirmed)
        "gemini-2.5-pro": 1000000,  # Gemini 2.5 Flash supports ~1M tokens
        "gemini-2.5-flash": 1000000,  # Gemini 2.5 Flash supports ~1M tokens
        "gemini-2.5-flash-lite": 1000000,  # Gemini 2.0 Flash Lite supports ~1M tokens
        "claude-sonnet-4": 200_000,  # Claude Sonnet 4 supports 1M tokens (beta)
        "claude-sonnet-3.7": 200_000,  # Claude Sonnet 3.7 has 200k token limit
        "claude-sonnet-3.5": 200_000,  # Claude Sonnet 3.5 has 200k token limit
        "claude-haiku-3.5": 200_000,  # Claude Haiku 3.5 has 200k limit
        "claude-opus-4.1": 200_000,  # Claude Haiku 3.5 has 200k limit
    }
    DEFAULT_TOKEN_SIZES = [10, 100, 1000, 10000, 100000, 1000000]

    def __init__(self, data_dir: Path = None, orq_api_key: str = None):
        """Initialize the experiment with data directory and API credentials."""

        # Set up data directory
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data"
        self.data_dir = Path(data_dir)

        # Set up API client
        self.orq_api_key = orq_api_key or os.getenv("ORQ_API_KEY")
        if not self.orq_api_key:
            raise ValueError("ORQ_API_KEY must be provided or set as environment variable")

        self.orq_client = OpenAI(api_key=self.orq_api_key, base_url="https://api.orq.ai/v2/proxy")

        # Set up tiktoken encoding
        self.encoding = tiktoken.encoding_for_model("gpt-4o")

        # Cache for text data
        self._sentences = None

        # Set up checkpoint directory
        self.checkpoint_dir = self.data_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def load_experiment(self, experiment_id: str, with_analyzer: bool = False):
        """Load existing experiment results from saved data.

        Args:
            experiment_id: The experiment ID or directory name (e.g., "full_experiment")
            with_analyzer: If True, return (results, analyzer), else just results

        Returns:
            list[ExperimentResult] or tuple[list[ExperimentResult], ExperimentAnalyzer]
        """
        project_root = Path(__file__).parent.parent.parent.parent
        results_dir = project_root / "results" / "scaling" / experiment_id

        if not results_dir.exists():
            # Fallback to configured data directory
            results_dir = self.data_dir / "results" / experiment_id

        if not results_dir.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found in {results_dir}")

        csv_path = results_dir / "results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found at {csv_path}")

        # Load CSV data
        df = pd.read_csv(csv_path)

        # Convert to ExperimentResult objects
        results = []
        for _, row in df.iterrows():
            result = ExperimentResult(
                model_name=row["model_name"],
                context_size=int(row["context_size"]),
                duration_seconds=float(row["duration_seconds"]) if pd.notna(row["duration_seconds"]) else 0.0,
                tokens_per_second=float(row["tokens_per_second"]) if pd.notna(row["tokens_per_second"]) else 0.0,
                success=bool(row["success"]),
                error_message=row.get("error_message", None) if pd.notna(row.get("error_message", None)) else None,
                timestamp=row.get("timestamp", None) if "timestamp" in df.columns else None,
                iteration=int(row["iteration"])
                if "iteration" in df.columns and pd.notna(row.get("iteration", None))
                else None,
            )
            results.append(result)

        print(f"✅ Loaded {len(results)} results from {experiment_id}")

        if with_analyzer:
            analyzer = ExperimentAnalyzer(results)
            return results, analyzer
        else:
            return results

    def prepare_text_data(self) -> list[str]:
        """Load and prepare text data from various sources for context window experiments."""

        if self._sentences is not None:
            return self._sentences

        # Load Paul Graham essays
        pg_essays = []
        pg_path = self.data_dir / "paul_graham_essays"
        if pg_path.exists():
            for txt_file in pg_path.glob("*.txt"):
                try:
                    with open(txt_file, encoding="utf-8") as f:
                        pg_essays.append(f.read())
                except Exception as e:
                    print(f"Warning: Could not read {txt_file}: {e}")

        # Load arxiv papers
        arxiv_papers = []
        arxiv_path = self.data_dir / "arxiv_papers"
        if arxiv_path.exists():
            for txt_file in list(arxiv_path.glob("*.txt"))[:20]:  # Limit to 20 papers
                try:
                    with open(txt_file, encoding="utf-8") as f:
                        arxiv_papers.append(f.read())
                except Exception as e:
                    print(f"Warning: Could not read {txt_file}: {e}")

        # Combine all text sources
        all_text = " ".join(pg_essays + arxiv_papers)

        # Split into sentences for better mixing
        sentences = all_text.replace("\n", " ").split(". ")
        sentences = [s.strip() + "." for s in sentences if len(s.strip()) > 10]

        print(
            f"Prepared {len(sentences)} sentences from {len(pg_essays)} PG essays and {len(arxiv_papers)} arxiv papers"
        )

        self._sentences = sentences
        return sentences

    def create_text_with_exact_tokens(self, target_tokens: int, sentences: list[str] | None = None) -> str:
        """Create text with exactly the specified number of tokens using tiktoken validation."""

        if sentences is None:
            sentences = self.prepare_text_data()

        current_text = ""
        current_tokens = 0

        # Randomly shuffle sentences to create variety
        shuffled_sentences = sentences.copy()
        random.shuffle(shuffled_sentences)

        # Add sentences until we're close to target
        sentence_idx = 0
        while current_tokens < target_tokens - 100:  # Leave some buffer
            if sentence_idx >= len(shuffled_sentences):
                sentence_idx = 0
                random.shuffle(shuffled_sentences)

            sentence = shuffled_sentences[sentence_idx]
            sentence_tokens = len(self.encoding.encode(sentence))

            if current_tokens + sentence_tokens <= target_tokens:
                current_text += " " + sentence
                current_tokens += sentence_tokens

            sentence_idx += 1

        # Fine-tune to exact token count
        while current_tokens < target_tokens:
            current_text += " word"
            current_tokens = len(self.encoding.encode(current_text))

        while current_tokens > target_tokens:
            current_text = current_text[:-1]
            current_tokens = len(self.encoding.encode(current_text))

        # Final validation
        final_tokens = len(self.encoding.encode(current_text))
        if abs(final_tokens - target_tokens) > 1:
            print(f"Warning: Token count mismatch. Target: {target_tokens}, Actual: {final_tokens}")

        return current_text.strip()

    def _extract_retry_delay(self, error_message: str) -> int:
        """Extract retry delay in seconds from 429 error message."""
        # Look for patterns like "retry after 44 seconds"
        patterns = [
            r"retry after (\d+) seconds",
            r"Please retry after (\d+) seconds",
            r"try again in (\d+) seconds",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # Default retry delay if no specific time found
        return 60

    def _filter_token_sizes_for_model(self, model_name: str, token_sizes: list[int]) -> list[int]:
        """Filter token sizes based on model limits."""
        if model_name not in self.MODEL_TOKEN_LIMITS:
            return token_sizes

        max_tokens = self.MODEL_TOKEN_LIMITS[model_name]
        filtered_sizes = [size for size in token_sizes if size <= max_tokens]

        if len(filtered_sizes) != len(token_sizes):
            skipped = [size for size in token_sizes if size > max_tokens]
            print(f"⚠️  Skipping token sizes {skipped} for {model_name} (max: {max_tokens:,} tokens)")

        return filtered_sizes

    def display_model_info(self):
        """Display information about all available models and their limits."""
        print("🤖 Available Models and Token Limits:")
        print("=" * 50)

        for model_name, model_id in self.MODELS.items():
            limit = self.MODEL_TOKEN_LIMITS.get(model_name, "Unknown")
            print(
                f"{model_name:20} | {model_id:40} | {limit:>10,} tokens"
                if isinstance(limit, int)
                else f"{model_name:20} | {model_id:40} | {limit:>10}"
            )

    def _generate_experiment_id(self, models: dict, token_sizes: list, iterations: int) -> str:
        """Generate a unique experiment ID based on configuration."""
        import hashlib

        config_str = f"{sorted(models.keys())}_{sorted(token_sizes)}_{iterations}"
        hash_obj = hashlib.md5(config_str.encode())
        short_hash = hash_obj.hexdigest()[:8]

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}_{short_hash}"

    def _save_checkpoint(self, experiment_id: str, completed_results: list, remaining_tasks: list, config: dict):
        """Save current experiment state to checkpoint file (keeps only latest version)."""
        checkpoint = ExperimentCheckpoint(
            experiment_id=experiment_id,
            completed_results=completed_results,
            remaining_tasks=remaining_tasks,
            experiment_config=config,
            timestamp=pd.Timestamp.now().isoformat(),
        )

        # Clean up any old checkpoints for this experiment first
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}.json"

        # Also clean up any old checkpoint versions (if we decide to version them later)
        # For now, we just overwrite the existing file

        # Convert dataclasses to dict for JSON serialization
        checkpoint_data = {
            "experiment_id": checkpoint.experiment_id,
            "completed_results": [
                {
                    "model_name": r.model_name,
                    "context_size": r.context_size,
                    "duration_seconds": r.duration_seconds,
                    "tokens_per_second": r.tokens_per_second,
                    "success": r.success,
                    "error_message": r.error_message,
                    "timestamp": r.timestamp,
                    "iteration": r.iteration,
                }
                for r in checkpoint.completed_results
            ],
            "remaining_tasks": checkpoint.remaining_tasks,
            "experiment_config": checkpoint.experiment_config,
            "timestamp": checkpoint.timestamp,
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_file}")

    def _load_checkpoint(self, experiment_id: str) -> ExperimentCheckpoint:
        """Load experiment state from checkpoint file."""
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}.json"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)

        # Convert dict back to dataclasses
        completed_results = [ExperimentResult(**r) for r in checkpoint_data["completed_results"]]

        return ExperimentCheckpoint(
            experiment_id=checkpoint_data["experiment_id"],
            completed_results=completed_results,
            remaining_tasks=checkpoint_data["remaining_tasks"],
            experiment_config=checkpoint_data["experiment_config"],
            timestamp=checkpoint_data["timestamp"],
        )

    def list_checkpoints(self) -> list:
        """List all available experiment checkpoints."""
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                checkpoints.append(
                    {
                        "experiment_id": data["experiment_id"],
                        "timestamp": data["timestamp"],
                        "completed_tasks": len(data["completed_results"]),
                        "remaining_tasks": len(data["remaining_tasks"]),
                        "file": checkpoint_file,
                    }
                )
            except Exception as e:
                print(f"⚠️  Error reading checkpoint {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

    def time_llm_call(
        self, model_name: str, model_id: str, context_text: str, iteration: int = 1, max_retries: int = 3
    ) -> ExperimentResult:
        """Time a single LLM call and return results."""

        # Simple prompt that incorporates the context
        prompt = f"""Based on the following context, determine if this text is about rabbits. Answer with just Yes or No. Don't say anything else.

Context:
{context_text}

Answer:"""

        context_tokens = len(self.encoding.encode(context_text))

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                logger.info(f"Starting streaming call for {model_id}")

                # Use streaming to capture TTFT
                stream = self.orq_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,  # Short response to focus on input processing time
                    temperature=0.0,
                    stream=True,  # Enable streaming
                )

                # Track streaming metrics
                first_token_time = None
                generation_start_time = None
                response_chunks = []
                chunk_count = 0
                response_text = ""

                for chunk in stream:
                    chunk_time = time.time()

                    # Capture time to first token (TTFT)
                    if first_token_time is None:
                        first_token_time = chunk_time
                        generation_start_time = chunk_time

                    # Extract content from chunk
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            response_text += delta.content
                            chunk_count += 1

                end_time = time.time()

                # Calculate timing metrics
                total_duration = end_time - start_time
                time_to_first_token = (first_token_time - start_time) if first_token_time else None
                generation_time = (end_time - generation_start_time) if generation_start_time else None

                # Calculate throughput metrics
                response_tokens = len(self.encoding.encode(response_text)) if response_text else 0
                response_length = len(response_text) if response_text else 0

                # Input processing throughput (context tokens / total time)
                tokens_per_sec = context_tokens / total_duration if total_duration > 0 else 0

                # Generation throughput (response tokens / generation time)
                tokens_per_sec_generation = (
                    (response_tokens / generation_time) if generation_time and generation_time > 0 else None
                )

                return ExperimentResult(
                    model_name=model_name,
                    context_size=context_tokens,
                    duration_seconds=total_duration,
                    tokens_per_second=tokens_per_sec,
                    success=True,
                    timestamp=pd.Timestamp.now().isoformat(),
                    iteration=iteration,
                    response_text=response_text,
                    response_tokens=response_tokens,
                    response_length=response_length,
                    time_to_first_token=time_to_first_token,
                    total_generation_time=generation_time,
                    tokens_per_second_generation=tokens_per_sec_generation,
                    streaming_chunks=chunk_count,
                )

            except Exception as e:  # noqa: PERF203
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed for {model_name} with {context_tokens} tokens: {error_msg}")

                # Check if this is a 429 rate limit error
                is_rate_limit_error = "429" in error_msg and "rate limit" in error_msg.lower()

                if attempt == max_retries - 1:
                    return ExperimentResult(
                        model_name=model_name,
                        context_size=context_tokens,
                        duration_seconds=0.0,
                        tokens_per_second=0.0,
                        success=False,
                        error_message=error_msg,
                        timestamp=pd.Timestamp.now().isoformat(),
                        iteration=iteration,
                        response_text=None,
                        response_tokens=None,
                        response_length=None,
                        time_to_first_token=None,
                        total_generation_time=None,
                        tokens_per_second_generation=None,
                        streaming_chunks=None,
                    )

                # Wait before retry - use extracted delay for 429 errors
                if is_rate_limit_error:
                    retry_delay = self._extract_retry_delay(error_msg)
                    print(f"⏳ Rate limit hit. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    # Exponential backoff for other errors
                    time.sleep(2**attempt)
        return None

    def run_single_experiment(
        self, model_name: str, model_id: str, token_count: int, iteration: int = 1, sentences: list[str] | None = None
    ) -> ExperimentResult:
        """Run a single experiment with specified model and token count."""

        if sentences is None:
            sentences = self.prepare_text_data()

        print(f"Testing {model_name} with {token_count:,} tokens...")

        # Create text with exact token count
        context_text = self.create_text_with_exact_tokens(token_count, sentences)

        # Time the LLM call
        result = self.time_llm_call(model_name, model_id, context_text, iteration)

        if result.success:
            print(f"✓ {model_name}: {result.duration_seconds:.2f}s ({result.tokens_per_second:.1f} tokens/sec)")
        else:
            print(f"✗ {model_name}: Failed - {result.error_message}")

        return result

    def run_experiment(
        self,
        models: dict[str, str] | None = None,
        token_sizes: list[int] | None = None,
        iterations_per_test: int = 3,
        sentences: list[str] | None = None,
        resume_from: str | None = None,
        save_checkpoints: bool = True,
    ) -> list[ExperimentResult]:
        """Run the full context window scaling experiment with checkpoint support."""

        # Handle resuming from checkpoint
        if resume_from:
            print(f"📂 Resuming experiment from checkpoint: {resume_from}")
            checkpoint = self._load_checkpoint(resume_from)

            all_results = checkpoint.completed_results.copy()
            remaining_tasks = checkpoint.remaining_tasks
            config = checkpoint.experiment_config
            experiment_id = checkpoint.experiment_id

            print(f"✅ Loaded {len(all_results)} completed results")
            print(f"📋 {len(remaining_tasks)} tasks remaining")

        else:
            # Set up new experiment
            if models is None:
                models = self.MODELS
            if token_sizes is None:
                token_sizes = DEFAULT_TOKEN_SIZES
            if sentences is None:
                sentences = self.prepare_text_data()

            config = {"models": models, "token_sizes": token_sizes, "iterations_per_test": iterations_per_test}

            experiment_id = self._generate_experiment_id(models, token_sizes, iterations_per_test)
            all_results = []

            # Generate all tasks
            remaining_tasks = []
            for model_name, model_id in models.items():
                # Filter token sizes for this model
                model_token_sizes = self._filter_token_sizes_for_model(model_name, token_sizes)
                for token_count in model_token_sizes:
                    for iteration in range(1, iterations_per_test + 1):
                        remaining_tasks.append((model_name, model_id, token_count, iteration))

        total_tests = len(all_results) + len(remaining_tasks)

        print("\n🚀 Context Window Scaling Experiment")
        print(f"📊 Experiment ID: {experiment_id}")
        print(f"📈 Progress: {len(all_results)}/{total_tests} completed ({len(all_results) / total_tests * 100:.1f}%)")
        print(f"⏱️  Remaining: {len(remaining_tasks)} tests")
        print("=" * 60)

        # Process remaining tasks
        test_count = len(all_results)

        try:
            while remaining_tasks:
                model_name, model_id, token_count, iteration = remaining_tasks.pop(0)
                test_count += 1

                print(f"\n🧪 Testing {model_name} | {token_count:,} tokens | Iteration {iteration}")
                print(f"📊 Progress: {test_count}/{total_tests} ({test_count / total_tests * 100:.1f}%)")

                try:
                    result = self.run_single_experiment(model_name, model_id, token_count, iteration, sentences)
                    all_results.append(result)

                    if result.success:
                        print(f"✅ Success: {result.duration_seconds:.2f}s ({result.tokens_per_second:.1f} tokens/sec)")
                    else:
                        print(f"❌ Failed: {result.error_message}")

                except Exception as e:
                    print(f"⚠️  Unexpected error: {e}")
                    error_result = ExperimentResult(
                        model_name=model_name,
                        context_size=token_count,
                        duration_seconds=0.0,
                        tokens_per_second=0.0,
                        success=False,
                        error_message=str(e),
                        timestamp=pd.Timestamp.now().isoformat(),
                        iteration=iteration,
                    )
                    all_results.append(error_result)

                # Save checkpoint after each test
                if save_checkpoints:
                    self._save_checkpoint(experiment_id, all_results, remaining_tasks, config)

                # Small delay between tests
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n⚠️  Experiment interrupted by user!")
            print(f"📊 Completed: {len(all_results)} tests")
            if save_checkpoints:
                self._save_checkpoint(experiment_id, all_results, remaining_tasks, config)
                print(f"💾 Progress saved. Resume with: resume_from='{experiment_id}'")

            return all_results

        print("\n" + "=" * 60)
        print("🎉 Experiment complete!")
        print(f"📊 Total tests: {len(all_results)}")

        # Quick summary
        successful_tests = [r for r in all_results if r.success]
        success_rate = len(successful_tests) / len(all_results) * 100 if all_results else 0
        print(f"✅ Success rate: {len(successful_tests)}/{len(all_results)} ({success_rate:.1f}%)")

        # Mark experiment as completed but don't clean up checkpoint yet
        # Cleanup will happen in cleanup_checkpoint() method after results are saved
        self._last_experiment_id = experiment_id if save_checkpoints and not remaining_tasks else None

        return all_results

    def cleanup_checkpoint(self, experiment_id: str = None):
        """Clean up checkpoint file after successful results saving."""
        if experiment_id is None:
            experiment_id = getattr(self, "_last_experiment_id", None)

        if experiment_id:
            checkpoint_file = self.checkpoint_dir / f"{experiment_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                print("🗑️  Cleaned up checkpoint file")
                # Clear the stored experiment ID
                if hasattr(self, "_last_experiment_id"):
                    self._last_experiment_id = None

    def recover_results_from_checkpoint(self, experiment_id: str) -> list[ExperimentResult]:
        """Recover completed results from a checkpoint file."""
        try:
            checkpoint = self._load_checkpoint(experiment_id)
            print(f"🔄 Recovered {len(checkpoint.completed_results)} results from checkpoint {experiment_id}")
            return checkpoint.completed_results
        except FileNotFoundError:
            print(f"❌ Checkpoint {experiment_id} not found")
            return []
        except Exception as e:
            print(f"❌ Error recovering from checkpoint: {e}")
            return []


class ExperimentAnalyzer:
    """Analysis and visualization tools for experiment results."""

    def __init__(self, results: list[ExperimentResult]):
        self.results = results
        self.df = self._to_dataframe()

    @staticmethod
    def _calculate_ci(values, confidence=0.95):
        """Calculate confidence interval for the mean using t-distribution."""
        from scipy import stats
        import numpy as np

        n = len(values)
        if n < 2:
            return np.mean(values), 0  # No CI for single points

        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of mean
        ci_range = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean, ci_range

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert experiment results to pandas DataFrame."""
        data = []
        for result in self.results:
            data.append(
                {
                    "model_name": result.model_name,
                    "context_size": result.context_size,
                    "duration_seconds": result.duration_seconds if result.success else None,
                    "tokens_per_second": result.tokens_per_second if result.success else None,
                    "success": result.success,
                    "error_message": result.error_message,
                    "response_text": getattr(result, "response_text", None),
                    "response_tokens": getattr(result, "response_tokens", None),
                    "response_length": getattr(result, "response_length", None),
                    # New streaming metrics
                    "time_to_first_token": getattr(result, "time_to_first_token", None),
                    "total_generation_time": getattr(result, "total_generation_time", None),
                    "tokens_per_second_generation": getattr(result, "tokens_per_second_generation", None),
                    "streaming_chunks": getattr(result, "streaming_chunks", None),
                }
            )
        return pd.DataFrame(data)

    def create_visualizations_new(self, save_dir: Path = None, show_plots: bool = False):
        """Create comprehensive visualizations using the new plotting module."""
        successful_df = self.df[self.df["success"]].copy()
        if len(successful_df) == 0:
            print("❌ No successful results to visualize")
            return

        # Use the new plotting module
        try:
            from context_is_king.plotting import ScalingPlotter

            print("📊 Creating visualizations using refactored plotting module...")

            # Create ScalingPlotter instance
            plotter = ScalingPlotter(successful_df)

            # Generate all plots
            plots = plotter.generate_all_plots(output_dir=save_dir)

            if show_plots:
                for plot_name, fig in plots.items():
                    print(f"📈 Displaying {plot_name}")
                    plt.figure(fig.number)
                    plt.show()

            # Print scaling statistics
            print("\n📊 Scaling Analysis:")
            scaling_stats = plotter.get_scaling_statistics()
            print(scaling_stats)

            print(f"✅ Generated {len(plots)} visualizations successfully")
            return plots

        except ImportError as e:
            print(f"⚠️  New plotting module not available: {e}")
            print("Please use create_visualizations() for legacy plotting")
            return None

    def print_summary(self):
        """Print comprehensive summary of experiment results."""
        df = self.df

        print("📊 EXPERIMENT RESULTS SUMMARY")
        print("=" * 50)

        print(f"\nTotal tests: {len(df)}")
        print(f"Successful tests: {df['success'].sum()}")
        print(f"Failed tests: {(~df['success']).sum()}")
        print(f"Success rate: {df['success'].mean():.1%}")

        # Success rate by model
        print("\n🏆 Success Rate by Model:")
        success_by_model = df.groupby("model_name")["success"].agg(["count", "sum", "mean"])
        success_by_model.columns = ["total_tests", "successful_tests", "success_rate"]
        success_by_model["success_rate"] = success_by_model["success_rate"].apply(lambda x: f"{x:.1%}")
        print(success_by_model)

        # Performance analysis for successful tests only
        successful_df = df[df["success"]].copy()

        if len(successful_df) > 0:
            print("\n⚡ Performance Analysis (Successful Tests Only):")
            print(f"Average duration: {successful_df['duration_seconds'].mean():.2f}s")
            print(f"Average throughput: {successful_df['tokens_per_second'].mean():.1f} tokens/sec")

            print("\n📈 Performance by Context Size:")
            perf_by_size = (
                successful_df.groupby("context_size")
                .agg({"duration_seconds": ["mean", "std", "count"], "tokens_per_second": ["mean", "std"]})
                .round(2)
            )
            print(perf_by_size)

            print("\n🚀 Performance by Model:")
            perf_by_model = (
                successful_df.groupby("model_name")
                .agg({"duration_seconds": ["mean", "std", "count"], "tokens_per_second": ["mean", "std"]})
                .round(2)
            )
            print(perf_by_model)

        # Error analysis
        failed_df = df[~df["success"]].copy()
        if len(failed_df) > 0:
            print(f"\n❌ Error Analysis ({len(failed_df)} failures):")
            error_counts = failed_df["error_message"].value_counts()
            for error, count in error_counts.head(5).items():
                print(f"  - {error}: {count} occurrences")

    def create_visualizations(self, save_dir: Path = None, show_plots: bool = False, figsize=(8, 6)):
        """Create comprehensive visualizations of the experiment results."""

        successful_df = self.df[self.df["success"]].copy()

        if len(successful_df) == 0:
            print("❌ No successful results to visualize")
            return

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create individual plots first and save them separately
        individual_plots = []

        # Import for scaling analysis
        import numpy as np
        from scipy import stats

        # 1. Duration vs Context Size (line plot with error bars)
        fig1, ax1 = plt.subplots(figsize=(10, 8))

        # Calculate scaling exponents for each model
        model_slopes = {}
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            if len(model_data) >= 3:
                log_size = np.log10(model_data["context_size"])
                log_duration = np.log10(model_data["duration_seconds"])
                slope, _, r_value, _, _ = stats.linregress(log_size, log_duration)
                if abs(float(r_value)) >= 0.7:  # Only show if correlation is strong
                    model_slopes[model] = float(slope)

        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]

            # Calculate mean and 95% CI for each context size
            context_sizes = []
            means = []
            ci_ranges = []

            for context_size in sorted(model_data["context_size"].unique()):
                subset = model_data[model_data["context_size"] == context_size]["duration_seconds"]
                mean, ci_range = self._calculate_ci(subset)
                context_sizes.append(context_size)
                means.append(mean)
                ci_ranges.append(ci_range)

            # Create label with scaling exponent if available
            if model in model_slopes:
                slope = model_slopes[model]
                label = f"{model} (slope: {slope:.2f})"
            else:
                label = model

            # Plot line with 95% confidence interval error bars
            ax1.errorbar(
                context_sizes,
                means,
                yerr=ci_ranges,
                label=label,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                capthick=1.5,
            )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Context Size (tokens)")
        ax1.set_ylabel("Duration (seconds)")
        ax1.set_title("Response Time vs Context Size (Mean ± 95% CI)")

        # Set custom y-axis ticks with actual numbers
        y_ticks = [1, 2, 5, 10, 20]
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([str(tick) for tick in y_ticks])

        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        individual_plots.append((fig1, "response_time_vs_context_size.png"))

        # 1b. Duration vs Context Size (linear y-axis version)
        fig1b, ax1b = plt.subplots(figsize=(figsize))
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]

            # Calculate mean and 95% CI for each context size
            context_sizes = []
            means = []
            ci_ranges = []

            for context_size in sorted(model_data["context_size"].unique()):
                subset = model_data[model_data["context_size"] == context_size]["duration_seconds"]
                mean, ci_range = self._calculate_ci(subset)
                context_sizes.append(context_size)
                means.append(mean)
                ci_ranges.append(ci_range)

            # Create label with scaling exponent if available (same as log-log version)
            if model in model_slopes:
                slope = model_slopes[model]
                label = f"{model} (slope: {slope:.2f})"
            else:
                label = model

            # Plot line with 95% confidence interval error bars
            ax1b.errorbar(
                context_sizes,
                means,
                yerr=ci_ranges,
                label=label,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                capthick=1.5,
            )

        ax1b.set_xscale("log")  # Keep x-axis log for context sizes
        # y-axis stays linear to show actual duration values
        ax1b.set_xlabel("Context Size (tokens)")
        ax1b.set_ylabel("Duration (seconds)")
        ax1b.set_title("Response Time vs Context Size (Linear Scale, Mean ± 95% CI)")
        ax1b.legend()
        ax1b.grid(True, alpha=0.3)
        plt.tight_layout()
        individual_plots.append((fig1b, "response_time_vs_context_size_linear.png"))

        # 2. Tokens per Second vs Context Size (line plot with error bars)
        fig2, ax2 = plt.subplots(figsize=(figsize))
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]

            # Calculate mean and 95% CI for each context size
            context_sizes = []
            means = []
            ci_ranges = []

            for context_size in sorted(model_data["context_size"].unique()):
                subset = model_data[model_data["context_size"] == context_size]["tokens_per_second"]
                mean, ci_range = self._calculate_ci(subset)
                context_sizes.append(context_size)
                means.append(mean)
                ci_ranges.append(ci_range)

            # Plot line with 95% confidence interval error bars
            ax2.errorbar(
                context_sizes,
                means,
                yerr=ci_ranges,
                label=model,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                capthick=1.5,
            )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Context Size (tokens)")
        ax2.set_ylabel("Throughput (tokens/sec)")
        ax2.set_title("Processing Throughput vs Context Size (Mean ± 95% CI)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        individual_plots.append((fig2, "throughput_vs_context_size.png"))

        # 3. Box plot of Duration by Model
        fig3, ax3 = plt.subplots(figsize=(figsize))
        if len(successful_df["model_name"].unique()) > 1:
            sns.boxplot(data=successful_df, x="model_name", y="duration_seconds", ax=ax3)
            ax3.set_title("Duration Distribution by Model")
            ax3.set_xlabel("Model")
            ax3.set_ylabel("Duration (seconds)")
            ax3.tick_params(axis="x", rotation=45)
        else:
            ax3.text(
                0.5, 0.5, "Multiple models needed\nfor comparison", ha="center", va="center", transform=ax3.transAxes
            )
            ax3.set_title("Duration Distribution by Model")
        plt.tight_layout()
        individual_plots.append((fig3, "duration_distribution_by_model.png"))

        # 4. Efficiency scatter: Duration vs Tokens per Second (aggregated by model+context_size)
        fig4, ax4 = plt.subplots(figsize=(figsize))
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            # Group by context size and calculate averages
            grouped = (
                model_data.groupby("context_size")
                .agg({"duration_seconds": "mean", "tokens_per_second": "mean"})
                .reset_index()
                .sort_values("context_size")
            )

            # Plot line connecting points for the same model
            ax4.plot(
                grouped["duration_seconds"],
                grouped["tokens_per_second"],
                marker="o",
                label=model,
                alpha=0.8,
                linewidth=2,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_xlabel("Duration (seconds)")
        ax4.set_ylabel("Throughput (tokens/sec)")
        ax4.set_title("Speed vs Throughput Efficiency (Averaged by Context Size)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        individual_plots.append((fig4, "speed_vs_throughput_efficiency.png"))

        # Save individual plots if save_dir is provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save individual plots
            for fig, filename in individual_plots:
                individual_plot_path = save_dir / filename
                fig.savefig(individual_plot_path, dpi=150, bbox_inches="tight")
                print(f"📊 Individual plot saved: {individual_plot_path}")

        # Now create the combined plot
        fig_combined, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig_combined.suptitle("Context Window Size vs LLM Performance Analysis", fontsize=16, fontweight="bold")

        # Copy the content from individual plots to the combined plot
        # 1. Duration vs Context Size (reuse the model_slopes calculation)
        ax1_combined = axes[0, 0]
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            grouped = model_data.groupby("context_size")["duration_seconds"].agg(["mean", "std", "count"])
            grouped = grouped.reset_index()

            # Use the same label logic as the individual plot
            if model in model_slopes:
                slope = model_slopes[model]
                label = f"{model} (slope: {slope:.2f})"
            else:
                label = model

            ax1_combined.errorbar(
                grouped["context_size"],
                grouped["mean"],
                yerr=grouped["std"],
                label=label,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                capthick=1.5,
            )

        ax1_combined.set_xscale("log")
        ax1_combined.set_yscale("log")
        ax1_combined.set_xlabel("Context Size (tokens)")
        ax1_combined.set_ylabel("Duration (seconds)")
        ax1_combined.set_title("Response Time vs Context Size (Average ± Std)")
        y_ticks = [1, 2, 5, 10, 20]
        ax1_combined.set_yticks(y_ticks)
        ax1_combined.set_yticklabels([str(tick) for tick in y_ticks])
        ax1_combined.legend()
        ax1_combined.grid(True, alpha=0.3)

        # 2. Tokens per Second vs Context Size
        ax2_combined = axes[0, 1]
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            grouped = model_data.groupby("context_size")["tokens_per_second"].agg(["mean", "std", "count"])
            grouped = grouped.reset_index()

            ax2_combined.errorbar(
                grouped["context_size"],
                grouped["mean"],
                yerr=grouped["std"],
                label=model,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                capthick=1.5,
            )

        ax2_combined.set_xscale("log")
        ax2_combined.set_yscale("log")
        ax2_combined.set_xlabel("Context Size (tokens)")
        ax2_combined.set_ylabel("Throughput (tokens/sec)")
        ax2_combined.set_title("Processing Throughput vs Context Size (Average ± Std)")
        ax2_combined.legend()
        ax2_combined.grid(True, alpha=0.3)

        # 3. Box plot of Duration by Model
        ax3_combined = axes[1, 0]
        if len(successful_df["model_name"].unique()) > 1:
            sns.boxplot(data=successful_df, x="model_name", y="duration_seconds", ax=ax3_combined)
            ax3_combined.set_title("Duration Distribution by Model")
            ax3_combined.set_xlabel("Model")
            ax3_combined.set_ylabel("Duration (seconds)")
            ax3_combined.tick_params(axis="x", rotation=45)
        else:
            ax3_combined.text(
                0.5,
                0.5,
                "Multiple models needed\nfor comparison",
                ha="center",
                va="center",
                transform=ax3_combined.transAxes,
            )
            ax3_combined.set_title("Duration Distribution by Model")

        # 4. Efficiency scatter
        ax4_combined = axes[1, 1]
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            grouped = (
                model_data.groupby("context_size")
                .agg({"duration_seconds": "mean", "tokens_per_second": "mean"})
                .reset_index()
                .sort_values("context_size")
            )

            ax4_combined.plot(
                grouped["duration_seconds"],
                grouped["tokens_per_second"],
                marker="o",
                label=model,
                alpha=0.8,
                linewidth=2,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        ax4_combined.set_xscale("log")
        ax4_combined.set_yscale("log")
        ax4_combined.set_xlabel("Duration (seconds)")
        ax4_combined.set_ylabel("Throughput (tokens/sec)")
        ax4_combined.set_title("Speed vs Throughput Efficiency (Averaged by Context Size)")
        ax4_combined.legend()
        ax4_combined.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the combined plot
        if save_dir:
            combined_plot_path = save_dir / "performance_analysis.png"
            fig_combined.savefig(combined_plot_path, dpi=150, bbox_inches="tight")
            print(f"📊 Combined plot saved: {combined_plot_path}")

        if show_plots:
            plt.show()
        else:
            # Close all figures to free memory
            for fig, _ in individual_plots:
                plt.close(fig)
            plt.close(fig_combined)

        # Additional scaling analysis
        self._analyze_scaling_patterns(successful_df)

    def _analyze_scaling_patterns(self, successful_df: pd.DataFrame):
        """Analyze scaling patterns in the data with intuitive interpretations."""
        if len(successful_df) <= 3:
            return

        print("\n📊 Scaling Pattern Analysis:")
        print("-" * 60)

        import numpy as np
        from scipy import stats

        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model].sort_values("context_size")
            if len(model_data) >= 3:
                # Log-log analysis for power law fitting
                log_size = np.log10(model_data["context_size"])
                log_duration = np.log10(model_data["duration_seconds"])

                # Linear regression in log-log space
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_size, log_duration)
                correlation = float(r_value)
                slope = float(slope)

                print(f"\n🤖 {model}:")
                print(f"  📈 Log-log correlation: {correlation:.3f}")

                # Power law interpretation
                if abs(correlation) >= 0.7:  # Strong correlation
                    print(f"  ⚡ Scaling exponent: {slope:.2f}")
                    if slope < 0.2:
                        scaling_desc = "Nearly constant (excellent scaling)"
                        emoji = "🚀"
                    elif slope < 0.5:
                        scaling_desc = "Sub-linear scaling (very good)"
                        emoji = "✅"
                    elif slope < 1.0:
                        scaling_desc = "Moderate scaling (acceptable)"
                        emoji = "⚠️"
                    elif slope < 2.0:
                        scaling_desc = "Linear to quadratic scaling (concerning)"
                        emoji = "🔶"
                    else:
                        scaling_desc = "Super-quadratic scaling (problematic)"
                        emoji = "❌"

                    print(f"  {emoji} Pattern: {scaling_desc}")

                    # Practical interpretation
                    if len(model_data) >= 2:
                        sizes = model_data["context_size"].values
                        durations = model_data["duration_seconds"].values

                        # Find 10x size increase example
                        size_ratios = []
                        duration_ratios = []
                        for i in range(len(sizes) - 1):
                            for j in range(i + 1, len(sizes)):
                                if sizes[j] / sizes[i] >= 5:  # At least 5x increase
                                    size_ratio = sizes[j] / sizes[i]
                                    duration_ratio = durations[j] / durations[i]
                                    size_ratios.append(size_ratio)
                                    duration_ratios.append(duration_ratio)

                        if size_ratios:
                            avg_size_ratio = np.mean(size_ratios)
                            avg_duration_ratio = np.mean(duration_ratios)
                            print(f"  📏 Example: {avg_size_ratio:.1f}x context → {avg_duration_ratio:.1f}x slower")

                else:  # Weak correlation
                    print(f"  🔄 Pattern: Inconsistent scaling (r={correlation:.3f})")
                    print(f"  💭 Interpretation: Performance may depend on factors beyond context size")

                # Performance classification
                avg_duration = model_data["duration_seconds"].mean()
                avg_throughput = model_data["tokens_per_second"].mean()

                if avg_duration < 2.0:
                    speed_class = "🚀 Fast"
                elif avg_duration < 5.0:
                    speed_class = "⚡ Moderate"
                else:
                    speed_class = "🐌 Slow"

                print(f"  {speed_class} (avg: {avg_duration:.1f}s, {avg_throughput:.0f} tok/s)")

        # Overall insights
        print(f"\n💡 Key Insights:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Find best scaling model
        scaling_scores = {}
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model].sort_values("context_size")
            if len(model_data) >= 3:
                log_size = np.log10(model_data["context_size"])
                log_duration = np.log10(model_data["duration_seconds"])
                slope, _, r_value, _, _ = stats.linregress(log_size, log_duration)

                # Score: lower slope is better, higher correlation is better
                score = -slope + abs(r_value)  # Favor low slope and high correlation
                scaling_scores[model] = score

        if scaling_scores:
            best_scaling = max(scaling_scores.keys(), key=lambda k: scaling_scores[k])
            print(f"🏆 Best scaling behavior: {best_scaling}")

        # Find fastest model
        avg_speeds = successful_df.groupby("model_name")["duration_seconds"].mean()
        fastest_model = avg_speeds.idxmin()
        print(f"🚀 Fastest model overall: {fastest_model} ({avg_speeds[fastest_model]:.1f}s avg)")

        # Find highest throughput
        avg_throughput = successful_df.groupby("model_name")["tokens_per_second"].mean()
        highest_throughput = avg_throughput.idxmax()
        print(f"⚡ Highest throughput: {highest_throughput} ({avg_throughput[highest_throughput]:.0f} tok/s avg)")

    def analyze_streaming_performance(self):
        """Comprehensive analysis of streaming performance metrics focusing on TTFT."""
        successful_df = self.df[self.df["success"] & self.df["time_to_first_token"].notna()].copy()

        if len(successful_df) == 0:
            print("❌ No streaming data available for analysis")
            return

        print("\n⚡ STREAMING PERFORMANCE ANALYSIS")
        print("=" * 60)

        # TTFT Analysis - The Most Important Metric
        print("🚀 TIME TO FIRST TOKEN (TTFT) - Key User Experience Metric:")
        print(f"   • Average TTFT: {successful_df['time_to_first_token'].mean():.3f} seconds")
        print(f"   • Median TTFT: {successful_df['time_to_first_token'].median():.3f} seconds")
        print(
            f"   • TTFT range: {successful_df['time_to_first_token'].min():.3f}s - {successful_df['time_to_first_token'].max():.3f}s"
        )

        # TTFT by Model
        print("\n📊 TTFT Performance by Model:")
        ttft_stats = (
            successful_df.groupby("model_name")["time_to_first_token"]
            .agg(["mean", "median", "std", "min", "max"])
            .round(4)
        )
        ttft_stats.columns = ["Mean_TTFT", "Median_TTFT", "Std_TTFT", "Min_TTFT", "Max_TTFT"]
        print(ttft_stats)

        # TTFT vs Context Size Analysis
        print("\n🔗 TTFT vs Context Size Correlation:")
        ttft_correlations = (
            successful_df.groupby("model_name")
            .apply(
                lambda x: pd.Series(
                    {
                        "ttft_context_corr": x["time_to_first_token"].corr(x["context_size"]),
                        "ttft_scaling_slope": np.polyfit(np.log10(x["context_size"]), x["time_to_first_token"], 1)[0]
                        if len(x) > 1
                        else None,
                    }
                )
            )
            .round(4)
        )
        print(ttft_correlations)

        # Generation Speed Analysis
        if "tokens_per_second_generation" in successful_df.columns:
            generation_df = successful_df[successful_df["tokens_per_second_generation"].notna()]
            if len(generation_df) > 0:
                print("\n📈 Generation Speed Analysis:")
                print(
                    f"   • Average generation speed: {generation_df['tokens_per_second_generation'].mean():.1f} tokens/sec"
                )
                print(
                    f"   • Generation speed range: {generation_df['tokens_per_second_generation'].min():.1f} - {generation_df['tokens_per_second_generation'].max():.1f} tokens/sec"
                )

                gen_stats = (
                    generation_df.groupby("model_name")["tokens_per_second_generation"].agg(["mean", "std"]).round(2)
                )
                print("\n📊 Generation Speed by Model:")
                print(gen_stats)

        # Context Processing vs Generation Time Breakdown
        print("\n⏱️  Timing Breakdown (Context Processing vs Generation):")
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            avg_ttft = model_data["time_to_first_token"].mean()
            avg_generation = (
                model_data["total_generation_time"].mean() if "total_generation_time" in model_data.columns else None
            )
            avg_total = model_data["duration_seconds"].mean()

            print(f"\n{model}:")
            print(f"   • TTFT (context processing): {avg_ttft:.3f}s ({avg_ttft / avg_total * 100:.1f}% of total)")
            if avg_generation is not None:
                print(f"   • Generation time: {avg_generation:.3f}s ({avg_generation / avg_total * 100:.1f}% of total)")
            print(f"   • Total time: {avg_total:.3f}s")

        # Performance Rankings
        print("\n🏆 PERFORMANCE RANKINGS:")

        # Fastest TTFT
        fastest_ttft = successful_df.groupby("model_name")["time_to_first_token"].mean().idxmin()
        fastest_ttft_time = successful_df.groupby("model_name")["time_to_first_token"].mean().min()
        print(f"   🥇 Fastest TTFT: {fastest_ttft} ({fastest_ttft_time:.3f}s)")

        # Most consistent TTFT
        most_consistent = successful_df.groupby("model_name")["time_to_first_token"].std().idxmin()
        consistency_score = successful_df.groupby("model_name")["time_to_first_token"].std().min()
        print(f"   🎯 Most consistent TTFT: {most_consistent} (σ={consistency_score:.4f}s)")

        # Best scaling (lowest correlation with context size)
        if len(ttft_correlations) > 0:
            best_scaling = ttft_correlations["ttft_context_corr"].idxmin()
            scaling_score = ttft_correlations.loc[best_scaling, "ttft_context_corr"]
            print(f"   📈 Best TTFT scaling: {best_scaling} (correlation={scaling_score:.3f})")

    def analyze_response_sizes(self):
        """Analyze response text sizes and patterns (legacy function for compatibility)."""
        successful_df = self.df[self.df["success"] & self.df["response_text"].notna()].copy()

        if len(successful_df) == 0:
            print("❌ No response data available for analysis")
            return

        print("\n📝 Response Size Analysis:")
        print("=" * 50)

        # Overall statistics
        print(f"Total responses analyzed: {len(successful_df)}")
        if "response_length" in successful_df.columns:
            print(f"Average response length: {successful_df['response_length'].mean():.1f} characters")
        if "response_tokens" in successful_df.columns:
            print(f"Average response tokens: {successful_df['response_tokens'].mean():.1f} tokens")

        # Response size by model
        if "response_length" in successful_df.columns and "response_tokens" in successful_df.columns:
            print("\n📊 Response Size by Model:")
            model_stats = (
                successful_df.groupby("model_name")
                .agg({"response_length": ["mean", "std"], "response_tokens": ["mean", "std"]})
                .round(2)
            )
            print(model_stats)

    def create_ttft_visualizations(self, save_dir: Path = None, show_plots: bool = True):
        """Create comprehensive TTFT and streaming performance visualizations."""
        successful_df = self.df[self.df["success"] & self.df["time_to_first_token"].notna()].copy()

        if len(successful_df) == 0:
            print("❌ No TTFT data available for visualization")
            return

        # Create individual TTFT plots with confidence intervals
        individual_plots = []

        # 1. TTFT vs Context Size with Mean ± 95% CI (PRIMARY CHART)
        fig_ttft, ax_ttft = plt.subplots(figsize=(12, 8))

        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]

            # Calculate mean and 95% CI for each context size
            context_sizes = []
            means = []
            ci_ranges = []

            for context_size in sorted(model_data["context_size"].unique()):
                subset = model_data[model_data["context_size"] == context_size]["time_to_first_token"]
                mean, ci_range = self._calculate_ci(subset)
                context_sizes.append(context_size)
                means.append(mean)
                ci_ranges.append(ci_range)

            # Plot line with 95% confidence interval error bars
            ax_ttft.errorbar(
                context_sizes,
                means,
                yerr=ci_ranges,
                label=model,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=5,
                capthick=2,
            )

        ax_ttft.set_xscale("log")
        ax_ttft.set_xlabel("Context Size (tokens)", fontsize=12)
        ax_ttft.set_ylabel("Time to First Token (seconds)", fontsize=12)
        ax_ttft.set_title("🚀 Time to First Token vs Context Size (Mean ± 95% CI)", fontsize=14, fontweight="bold")
        ax_ttft.legend(fontsize=10)
        ax_ttft.grid(True, alpha=0.3)
        plt.tight_layout()
        individual_plots.append((fig_ttft, "ttft_vs_context_size.png"))

        # 2. TTFT vs Context Size (Linear Y-axis)
        fig_ttft_linear, ax_ttft_linear = plt.subplots(figsize=(12, 8))

        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]

            # Calculate mean and 95% CI for each context size
            context_sizes = []
            means = []
            ci_ranges = []

            for context_size in sorted(model_data["context_size"].unique()):
                subset = model_data[model_data["context_size"] == context_size]["time_to_first_token"]
                mean, ci_range = self._calculate_ci(subset)
                context_sizes.append(context_size)
                means.append(mean)
                ci_ranges.append(ci_range)

            # Plot line with 95% confidence interval error bars
            ax_ttft_linear.errorbar(
                context_sizes,
                means,
                yerr=ci_ranges,
                label=model,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=5,
                capthick=2,
            )

        ax_ttft_linear.set_xscale("log")  # Keep x-axis log for context sizes
        ax_ttft_linear.set_xlabel("Context Size (tokens)", fontsize=12)
        ax_ttft_linear.set_ylabel("Time to First Token (seconds)", fontsize=12)
        ax_ttft_linear.set_title(
            "🚀 Time to First Token vs Context Size (Linear Scale, Mean ± 95% CI)", fontsize=14, fontweight="bold"
        )
        ax_ttft_linear.legend(fontsize=10)
        ax_ttft_linear.grid(True, alpha=0.3)
        plt.tight_layout()
        individual_plots.append((fig_ttft_linear, "ttft_vs_context_size_linear.png"))

        # Create a 2x3 subplot layout for additional TTFT analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("🚀 Additional TTFT Analysis", fontsize=16, fontweight="bold")

        # 1. TTFT Scatter Plot (for detailed view)
        ax1 = axes[0, 0]
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            ax1.scatter(model_data["context_size"], model_data["time_to_first_token"], label=model, alpha=0.7, s=60)
        ax1.set_xscale("log")
        ax1.set_xlabel("Context Size (tokens)")
        ax1.set_ylabel("Time to First Token (seconds)")
        ax1.set_title("TTFT Scatter Plot (All Data Points)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. TTFT Distribution by Model (Box Plot)
        ax2 = axes[0, 1]
        successful_df.boxplot(column="time_to_first_token", by="model_name", ax=ax2)
        ax2.set_title("TTFT Distribution by Model")
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Time to First Token (seconds)")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. TTFT vs Total Duration (Processing vs Generation Time)
        ax3 = axes[0, 2]
        ax3.scatter(
            successful_df["time_to_first_token"], successful_df["duration_seconds"], alpha=0.6, s=40, c="orange"
        )
        ax3.set_xlabel("Time to First Token (seconds)")
        ax3.set_ylabel("Total Duration (seconds)")
        ax3.set_title("TTFT vs Total Duration")
        ax3.grid(True, alpha=0.3)

        # Add diagonal line for reference
        max_val = max(successful_df["time_to_first_token"].max(), successful_df["duration_seconds"].max())
        ax3.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Equal time")
        ax3.legend()

        # 4. Average TTFT by Context Size
        ax4 = axes[1, 0]
        avg_ttft = successful_df.groupby(["model_name", "context_size"])["time_to_first_token"].mean().reset_index()
        for model in avg_ttft["model_name"].unique():
            model_data = avg_ttft[avg_ttft["model_name"] == model]
            ax4.plot(
                model_data["context_size"], model_data["time_to_first_token"], marker="o", label=model, linewidth=2
            )
        ax4.set_xscale("log")
        ax4.set_xlabel("Context Size (tokens)")
        ax4.set_ylabel("Average TTFT (seconds)")
        ax4.set_title("Average TTFT Scaling by Model")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Generation Speed (if available)
        ax5 = axes[1, 1]
        if "tokens_per_second_generation" in successful_df.columns:
            generation_df = successful_df[successful_df["tokens_per_second_generation"].notna()]
            if len(generation_df) > 0:
                generation_df.boxplot(column="tokens_per_second_generation", by="model_name", ax=ax5)
                ax5.set_title("Generation Speed Distribution")
                ax5.set_xlabel("Model")
                ax5.set_ylabel("Generation Speed (tokens/sec)")
                plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax5.text(0.5, 0.5, "No generation speed data", ha="center", va="center", transform=ax5.transAxes)
                ax5.set_title("Generation Speed Distribution")
        else:
            ax5.text(0.5, 0.5, "Generation speed not available", ha="center", va="center", transform=ax5.transAxes)
            ax5.set_title("Generation Speed Distribution")

        # 6. TTFT Scaling Analysis (Linear vs Context)
        ax6 = axes[1, 2]
        for model in successful_df["model_name"].unique():
            model_data = successful_df[successful_df["model_name"] == model]
            if len(model_data) > 1:
                # Plot linear scale to see scaling behavior
                ax6.plot(model_data["context_size"], model_data["time_to_first_token"], "o-", label=model, alpha=0.7)
        ax6.set_xlabel("Context Size (tokens)")
        ax6.set_ylabel("TTFT (seconds)")
        ax6.set_title("TTFT Scaling (Linear Scale)")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save individual plots and combined plot
        if save_dir:
            # Save individual TTFT plots (most important ones first)
            for fig, filename in individual_plots:
                save_path = save_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"📊 TTFT plot saved: {save_path}")
                if not show_plots:
                    plt.close(fig)

            # Save combined analysis plot
            combined_path = save_dir / "ttft_combined_analysis.png"
            fig.savefig(combined_path, dpi=300, bbox_inches="tight")
            print(f"📊 Combined TTFT analysis saved: {combined_path}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def create_response_size_visualizations(self, save_dir: Path = None, show_plots: bool = True):
        """Create visualizations for response size analysis (legacy compatibility)."""
        # Call the new TTFT visualizations instead
        self.create_ttft_visualizations(save_dir, show_plots)

    def save_results(self, experiment_id: str = None, results_dir: Path = None) -> Path:
        """Save experiment results to a dedicated directory with CSV, JSON, and plots."""

        if results_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            results_dir = project_root / "results" / "scaling"
        results_dir = Path(results_dir)

        # Create experiment-specific directory
        if experiment_id is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"experiment_{timestamp}"

        # Create experiment directory
        experiment_dir = results_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = experiment_dir / "results.csv"
        self.df.to_csv(csv_path, index=False)

        # Save JSON with full details
        json_data = []
        for result in self.results:
            json_data.append(
                {
                    "model_name": result.model_name,
                    "context_size": result.context_size,
                    "duration_seconds": result.duration_seconds,
                    "tokens_per_second": result.tokens_per_second,
                    "success": result.success,
                    "error_message": result.error_message,
                    "timestamp": getattr(result, "timestamp", pd.Timestamp.now().isoformat()),
                    "iteration": getattr(result, "iteration", None),
                    "response_text": getattr(result, "response_text", None),
                    "response_tokens": getattr(result, "response_tokens", None),
                    "response_length": getattr(result, "response_length", None),
                    # New streaming metrics
                    "time_to_first_token": getattr(result, "time_to_first_token", None),
                    "total_generation_time": getattr(result, "total_generation_time", None),
                    "tokens_per_second_generation": getattr(result, "tokens_per_second_generation", None),
                    "streaming_chunks": getattr(result, "streaming_chunks", None),
                }
            )

        json_path = experiment_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Create visualizations and save them
        self.create_visualizations(save_dir=experiment_dir, show_plots=False)

        # Create TTFT analysis and visualizations
        self.create_ttft_visualizations(save_dir=experiment_dir, show_plots=False)

        # Save summary report
        summary_path = experiment_dir / "summary.txt"
        with open(summary_path, "w") as f:
            # Redirect print output to file
            import sys

            old_stdout = sys.stdout
            sys.stdout = f
            self.print_summary()
            self.analyze_streaming_performance()  # Primary analysis
            self.analyze_response_sizes()  # Legacy compatibility
            sys.stdout = old_stdout

        print(f"📁 Results saved to: {experiment_dir}")
        print("   ├── results.csv (now includes TTFT + streaming metrics)")
        print("   ├── results.json (now includes TTFT + streaming metrics)")
        print("   ├── summary.txt (now includes TTFT analysis)")
        print("   ├── 🎯 ttft_vs_context_size.png (KEY CHART: TTFT Mean ± 95% CI)")
        print("   ├── 🎯 ttft_vs_context_size_linear.png (TTFT Linear Scale)")
        print("   ├── ttft_combined_analysis.png (comprehensive TTFT analysis)")
        print("   ├── performance_analysis.png (combined legacy charts)")
        print("   ├── response_time_vs_context_size.png")
        print("   ├── response_time_vs_context_size_linear.png")
        print("   ├── throughput_vs_context_size.png")
        print("   ├── duration_distribution_by_model.png")
        print("   └── speed_vs_throughput_efficiency.png")

        return experiment_dir


def quick_test(data_dir: Path = None) -> list[ExperimentResult]:
    """Run a quick test with one model and small context sizes."""
    experiment = ContextWindowExperiment(data_dir=data_dir)

    return experiment.run_experiment(
        models={"gpt-5-nano": experiment.MODELS["gpt-5-nano"]}, token_sizes=[10, 100, 1000], iterations_per_test=2
    )


def full_experiment(data_dir: Path = None) -> list[ExperimentResult]:
    """Run the full experiment with all models and context sizes."""
    experiment = ContextWindowExperiment(data_dir=data_dir)

    return experiment.run_experiment()


if __name__ == "__main__":
    # Quick test when run directly
    print("Running quick test...")
    results = quick_test()

    analyzer = ExperimentAnalyzer(results)
    analyzer.print_summary()
    analyzer.create_visualizations()
