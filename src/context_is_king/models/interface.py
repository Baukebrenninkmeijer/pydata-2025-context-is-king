#!/usr/bin/env python3
"""
Unified Model Interface for Context Window Advantage Experiments

This module provides a consistent interface for testing multiple models through the ORQ API.
Supports both high-capacity models (1M+ tokens) and medium-capacity models (200K tokens)
for context window comparison experiments.

Usage:
    from model_interface import ModelInterface, ModelConfig

    interface = ModelInterface()
    result = interface.query_model("gpt-4o", "What is 2+2?", max_tokens=100)
"""

import asyncio
import random
import re
import time
from dataclasses import dataclass
import os
import openai
import tiktoken
from openai import AsyncOpenAI
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for a model."""

    name: str
    api_name: str  # Name used in API calls
    max_context_tokens: int
    cost_per_1k_tokens: float  # Input cost
    provider: str  # 'openai', 'anthropic', 'google', etc.
    supports_system_prompt: bool = True


@dataclass
class QueryResult:
    """Result from a model query."""

    model_name: str
    prompt: str
    response: str
    success: bool
    error_message: str | None

    # Timing and token information
    response_time: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Cost information
    estimated_cost: float

    # Metadata
    temperature: float
    max_tokens: int
    context_size: int
    timestamp: float


class ModelInterface:
    """Unified interface for querying multiple models via ORQ API."""

    # Model configurations for context window advantage experiments
    # Focus on same-tier models with different context window training
    MODELS = {  # noqa: RUF012
        # GPT-4 Family: Full Models (Same architecture, different context windows)
        "gpt-4": ModelConfig(
            name="GPT-4",
            api_name="openai/gpt-4",
            max_context_tokens=8192,
            cost_per_1k_tokens=0.03,
            provider="openai",
        ),
        "gpt-4-turbo": ModelConfig(
            name="GPT-4 Turbo",
            api_name="openai/gpt-4-turbo",
            max_context_tokens=128000,
            cost_per_1k_tokens=0.01,
            provider="openai",
        ),
        "gpt-4o": ModelConfig(
            name="GPT-4o",
            api_name="azure/gpt-4o",
            max_context_tokens=128000,
            cost_per_1k_tokens=0.005,
            provider="openai",
        ),
        "gpt-4.1": ModelConfig(
            name="GPT-4.1",
            api_name="azure/gpt-4.1",  # Hypothetical - adjust when available
            max_context_tokens=1000000,  # Assumed larger context
            cost_per_1k_tokens=0.007,
            provider="openai",
        ),
        # GPT-4 Family: Mini Models (Same architecture tier, different context windows)
        "gpt-4o-mini": ModelConfig(
            name="GPT-4o Mini",
            api_name="azure/gpt-4o-mini",
            max_context_tokens=128000,
            cost_per_1k_tokens=0.00015,
            provider="openai",
        ),
        "gpt-4.1-mini": ModelConfig(
            name="GPT-4.1 Mini",
            api_name="azure/gpt-4.1-mini",  # Hypothetical - adjust when available
            max_context_tokens=1000000,  # Assumed larger context
            cost_per_1k_tokens=0.0003,
            provider="openai",
        ),
        # "gpt-4.1-mini": ModelConfig(
        #     name="GPT-4.1 Mini",
        #     api_name="pydata-talk@pydata-talk-gpt-4.1-mini-2025-04-14",  # Hypothetical - adjust when available
        #     max_context_tokens=1000000,  # Assumed larger context
        #     cost_per_1k_tokens=0.0003,
        #     provider="openai",
        # ),
        "gpt-4.1-mini-oa": ModelConfig(
            name="GPT-4.1 Mini",
            api_name="openai/gpt-4.1-mini",  # Hypothetical - adjust when available
            max_context_tokens=1000000,  # Assumed larger context
            cost_per_1k_tokens=0.0003,
            provider="openai",
        ),
        # Claude Family: Sonnet Models (Same tier, different context windows)
        "claude-sonnet-3.5": ModelConfig(
            name="Claude Sonnet 3.5",
            api_name="google/claude-3-5-sonnet-v2@20241022",
            max_context_tokens=200_000,
            cost_per_1k_tokens=0.003,
            provider="anthropic",
        ),
        "claude-sonnet-3.7": ModelConfig(
            name="Claude Sonnet 3.7",
            api_name="google/claude-3-7-sonnet@20250219",  # Hypothetical - adjust when available
            max_context_tokens=200_000,  # Assumed intermediate context
            cost_per_1k_tokens=0.004,
            provider="anthropic",
        ),
        "claude-sonnet-4": ModelConfig(
            name="Claude Sonnet 4",
            api_name="google/claude-sonnet-4@20250514",
            max_context_tokens=1000000,
            cost_per_1k_tokens=0.005,
            provider="anthropic",
        ),
        # Claude Family: Haiku Models (Same tier, different context windows)
        "claude-haiku-3.5": ModelConfig(
            name="Claude Haiku 3.5",
            api_name="google/claude-3-5-haiku@20241022",
            max_context_tokens=200_000,
            cost_per_1k_tokens=0.001,
            provider="anthropic",
        ),
        "gemini-2.5-flash-lite": ModelConfig(
            name="Gemini 2.5 Flash Lite",
            api_name="google/gemini-2.5-flash-lite",
            max_context_tokens=1000000,
            cost_per_1k_tokens=0.005,
            provider="google",
        ),
        "gemini-2.5-flash": ModelConfig(
            name="Gemini 2.5 Flash",
            api_name="google/gemini-2.5-flash",
            max_context_tokens=1000000,
            cost_per_1k_tokens=0.005,
            provider="google",
        ),
        "gemini-2.5-pro": ModelConfig(
            name="Gemini 2.5 Flash",
            api_name="google/gemini-2.5-flash",
            max_context_tokens=1000000,
            cost_per_1k_tokens=0.005,
            provider="google",
        ),
    }

    def __init__(
        self,
        api_key: str | None = os.environ.get("ORQ_API_KEY"),
        base_url: str = os.environ["ORQ_BASE_URL"],
        default_max_retries: int = 5,
        default_base_delay: float = 1.0,
    ):
        """Initialize the model interface with rate limiting configuration."""
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # Use GPT-4 encoding as standard

        # Rate limiting configuration
        self.default_max_retries = default_max_retries
        self.default_base_delay = default_base_delay

        print("🔗 Initialized Model Interface")
        print(f"📡 API Base URL: {base_url}")
        print(f"🤖 Available Models: {len(self.MODELS)}")
        print(f"🔄 Rate Limiting: {default_max_retries} retries, {default_base_delay}s base delay")

    def _extract_wait_time_from_error(self, error_message: str) -> float:
        """Extract wait time from rate limit error messages."""
        error_str = str(error_message).lower()

        # Common patterns for wait times in rate limit errors
        patterns = [
            r"after (\d+) seconds?",
            r"retry after (\d+) seconds?",
            r"please wait (\d+) seconds?",
            r"rate limit.*?(\d+) seconds?",
            r"try again in (\d+) seconds?",
            r"try again in (\d+)s",
            r"wait (\d+)s",
            r"(\d+)\s*seconds? and try again",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str)
            if match:
                seconds = int(match.group(1))
                # Add small buffer to be safe
                return float(seconds + 1)

        # Check for minute patterns and convert to seconds
        minute_patterns = [
            r"after (\d+) minutes?",
            r"retry after (\d+) minutes?",
            r"wait (\d+) minutes?",
        ]

        for pattern in minute_patterns:
            match = re.search(pattern, error_str)
            if match:
                minutes = int(match.group(1))
                return float(minutes * 60 + 5)  # Convert to seconds + buffer

        # Default fallback based on attempt number
        return None

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        return self.MODELS[model_name]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def validate_context_size(self, model_name: str, context_size: int) -> bool:
        """Check if context size is within model limits."""
        config = self.get_model_config(model_name)
        return context_size <= config.max_context_tokens

    def estimate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
        """Estimate cost for a query."""
        config = self.get_model_config(model_name)
        # Simplified cost calculation (input tokens only)
        return (prompt_tokens + completion_tokens) / 1000 * config.cost_per_1k_tokens

    async def query_model_async(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float = 300.0,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> QueryResult:
        """Query a model asynchronously with retry logic and rate limit handling."""

        config = self.get_model_config(model_name)
        start_time = time.time()

        # Count prompt tokens
        full_prompt = prompt
        if system_prompt and config.supports_system_prompt:
            full_prompt = system_prompt + "\n\n" + prompt

        prompt_tokens = self.count_tokens(full_prompt)
        context_size = prompt_tokens

        # Validate context size
        if not self.validate_context_size(model_name, context_size):
            return QueryResult(
                model_name=model_name,
                prompt=prompt,
                response="",
                success=False,
                error_message=f"Context size {context_size} exceeds model limit {config.max_context_tokens}",
                response_time=time.time() - start_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
                estimated_cost=0.0,
                temperature=temperature,
                max_tokens=max_tokens,
                context_size=context_size,
                timestamp=time.time(),
            )

        # Prepare messages
        messages = []
        if system_prompt and config.supports_system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Retry loop with exponential backoff
        for attempt in range(max_retries + 1):
            try:
                # Make API call
                response = await self.async_client.chat.completions.create(
                    model=config.api_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

                # Extract response
                completion_text = response.choices[0].message.content
                completion_tokens = self.count_tokens(completion_text) if completion_text else 0
                total_tokens = prompt_tokens + completion_tokens

                # Calculate cost
                estimated_cost = self.estimate_cost(model_name, prompt_tokens, completion_tokens)

                return QueryResult(
                    model_name=model_name,
                    prompt=prompt,
                    response=completion_text or "",
                    success=True,
                    error_message=None,
                    response_time=time.time() - start_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost=estimated_cost,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context_size=context_size,
                    timestamp=time.time(),
                )

            except asyncio.TimeoutError:
                if attempt == max_retries:
                    return QueryResult(
                        model_name=model_name,
                        prompt=prompt,
                        response="",
                        success=False,
                        error_message=f"Request timed out after {timeout} seconds (final attempt)",
                        response_time=time.time() - start_time,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_tokens,
                        estimated_cost=0.0,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context_size=context_size,
                        timestamp=time.time(),
                    )
                # Short delay before timeout retry
                await asyncio.sleep(base_delay)
                continue

            except openai.RateLimitError as e:
                if attempt == max_retries:
                    return QueryResult(
                        model_name=model_name,
                        prompt=prompt,
                        response="",
                        success=False,
                        error_message=f"Rate limit exceeded after {max_retries} retries: {e!s}",
                        response_time=time.time() - start_time,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_tokens,
                        estimated_cost=0.0,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context_size=context_size,
                        timestamp=time.time(),
                    )

                # Try to extract wait time from error message
                extracted_wait = self._extract_wait_time_from_error(str(e))
                if extracted_wait:
                    delay = extracted_wait
                    logger.error(
                        f"⚠️  Rate limit hit for {model_name}, API requested {delay:.1f}s wait (attempt {attempt + 1}/{max_retries + 1})"
                    )
                else:
                    # Exponential backoff with jitter as fallback
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.error(
                        f"⚠️  Rate limit hit for {model_name}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                    )
                await asyncio.sleep(delay)
                continue

            except openai.APIError as e:
                # Check if it's a 429 specifically
                if hasattr(e, "status_code") and e.status_code == 429:
                    if attempt == max_retries:
                        return QueryResult(
                            model_name=model_name,
                            prompt=prompt,
                            response="",
                            success=False,
                            error_message=f"API Error 429 after {max_retries} retries: {e!s}",
                            response_time=time.time() - start_time,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            total_tokens=prompt_tokens,
                            estimated_cost=0.0,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            context_size=context_size,
                            timestamp=time.time(),
                        )

                    # Try to extract wait time from 429 error message
                    extracted_wait = self._extract_wait_time_from_error(str(e))
                    if extracted_wait:
                        delay = extracted_wait
                        print(
                            f"⚠️  API Error 429 for {model_name}, API requested {delay:.1f}s wait (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    else:
                        # Handle 429 with exponential backoff as fallback
                        delay = base_delay * (2**attempt) + random.uniform(0, 2)
                        print(
                            f"⚠️  API Error 429 for {model_name}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    await asyncio.sleep(delay)
                    continue
                # Non-retryable API error
                return QueryResult(
                    model_name=model_name,
                    prompt=prompt,
                    response="",
                    success=False,
                    error_message=f"API Error: {e!s}",
                    response_time=time.time() - start_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    total_tokens=prompt_tokens,
                    estimated_cost=0.0,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context_size=context_size,
                    timestamp=time.time(),
                )

            except Exception as e:
                # Check if it's a rate limit error in the message
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                    if attempt == max_retries:
                        return QueryResult(
                            model_name=model_name,
                            prompt=prompt,
                            response="",
                            success=False,
                            error_message=f"Rate limit error after {max_retries} retries: {e!s}",
                            response_time=time.time() - start_time,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            total_tokens=prompt_tokens,
                            estimated_cost=0.0,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            context_size=context_size,
                            timestamp=time.time(),
                        )

                    # Try to extract wait time from rate limit error message
                    extracted_wait = self._extract_wait_time_from_error(str(e))
                    if extracted_wait:
                        delay = extracted_wait
                        print(
                            f"⚠️  Rate limit detected for {model_name}, API requested {delay:.1f}s wait (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    else:
                        # Retry with exponential backoff as fallback
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        print(
                            f"⚠️  Rate limit detected for {model_name}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    await asyncio.sleep(delay)
                    continue
                # Non-retryable error
                return QueryResult(
                    model_name=model_name,
                    prompt=prompt,
                    response="",
                    success=False,
                    error_message=str(e),
                    response_time=time.time() - start_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    total_tokens=prompt_tokens,
                    estimated_cost=0.0,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context_size=context_size,
                    timestamp=time.time(),
                )

        # Should never reach here, but just in case
        return QueryResult(
            model_name=model_name,
            prompt=prompt,
            response="",
            success=False,
            error_message="Unexpected error: exceeded retry loop",
            response_time=time.time() - start_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
            estimated_cost=0.0,
            temperature=temperature,
            max_tokens=max_tokens,
            context_size=context_size,
            timestamp=time.time(),
        )

    def query_model(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float = 300.0,
        *,
        use_async: bool = True,
    ) -> QueryResult:
        """Query a model synchronously with default retry settings."""
        if use_async:
            return asyncio.run(
                self.query_model_async(
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    max_retries=self.default_max_retries,
                    base_delay=self.default_base_delay,
                )
            )
        return self.query_model_sync(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=self.default_max_retries,
        )

    def query_model_sync(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        timeout: float = 30,
        max_tokens: int = 1000,
    ) -> QueryResult:
        config = self.get_model_config(model_name)
        start_time = time.time()

        messages = []
        if system_prompt and config.supports_system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        prompt_tokens = self.count_tokens(prompt)
        context_size = prompt_tokens

        # Retry loop with exponential backoff
        for attempt in range(max_retries + 1):
            try:
                # Make API call
                response = self.client.chat.completions.create(
                    model=config.api_name,
                    messages=messages,
                    temperature=temperature,
                )

                # Extract response
                completion_text = response.choices[0].message.content
                completion_tokens = self.count_tokens(completion_text) if completion_text else 0
                total_tokens = prompt_tokens + completion_tokens

                # Calculate cost
                estimated_cost = self.estimate_cost(model_name, prompt_tokens, completion_tokens)

                return QueryResult(
                    model_name=model_name,
                    prompt=prompt,
                    response=completion_text or "",
                    success=True,
                    error_message=None,
                    response_time=time.time() - start_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost=estimated_cost,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context_size=context_size,
                    timestamp=time.time(),
                )
            except Exception as e:
                if attempt == max_retries:
                    return QueryResult(
                        model_name=model_name,
                        prompt=prompt,
                        response="",
                        success=False,
                        error_message=f"Error after {max_retries} retries: {e!s}",
                        response_time=time.time() - start_time,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_tokens,
                        estimated_cost=0.0,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context_size=context_size,
                        timestamp=time.time(),
                    )
                else:
                    sleep_duration = 2**attempt
                    time.sleep(sleep_duration)
                    continue
        return None

    async def batch_query_models(
        self,
        queries: list[tuple[str, str]],  # [(model_name, prompt), ...]
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_concurrent: int = 5,
    ) -> list[QueryResult]:
        """Query multiple models concurrently."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_query(model_name: str, prompt: str) -> QueryResult:
            async with semaphore:
                return await self.query_model_async(model_name, prompt, system_prompt, temperature, max_tokens)

        tasks = [bounded_query(model_name, prompt) for model_name, prompt in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model_name, prompt = queries[i]
                processed_results.append(
                    QueryResult(
                        model_name=model_name,
                        prompt=prompt,
                        response="",
                        success=False,
                        error_message=str(result),
                        response_time=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_cost=0.0,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context_size=0,
                        timestamp=time.time(),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_model_comparison_pairs(self) -> list[tuple[str, str]]:
        """Get model pairs for context window advantage comparison."""
        return [
            # GPT-4 Family: Full Models
            ("gpt-4", "gpt-4-turbo"),  # 8K vs 128K
            ("gpt-4o", "gpt-4.1"),  # 128K vs 1M (if available)
            # GPT-4 Family: Mini Models
            ("gpt-4o-mini", "gpt-4.1-mini"),  # 128K vs 1M (if available)
            # Claude Family: Sonnet Models
            # ("claude-sonnet-3.5", "claude-sonnet-3.7"),  # 200K vs 500K (if available)
            ("claude-sonnet-3.5", "claude-sonnet-4"),  # 200K vs 1M
            ("claude-sonnet-3.7", "claude-sonnet-4"),  # 500K vs 1M (if available)
            # Claude Family: Haiku Models
            # ("claude-haiku-3.5", "claude-haiku-4"),  # 200K vs 1M (if available)
        ]

    def get_gpt4_family_models(self) -> dict[str, ModelConfig]:
        """Get GPT-4 family models for comparison."""
        return {name: config for name, config in self.MODELS.items() if name.startswith("gpt-4")}

    def get_claude_sonnet_models(self) -> dict[str, ModelConfig]:
        """Get Claude Sonnet models for comparison."""
        return {name: config for name, config in self.MODELS.items() if "sonnet" in name}

    def get_claude_haiku_models(self) -> dict[str, ModelConfig]:
        """Get Claude Haiku models for comparison."""
        return {name: config for name, config in self.MODELS.items() if "haiku" in name}

    def get_context_sizes_for_pair(self, smaller_model: str, larger_model: str) -> list[int]:
        """Get appropriate context sizes for testing a model pair."""
        smaller_config = self.get_model_config(smaller_model)
        max_context = smaller_config.max_context_tokens

        # Test at various points within the smaller model's capacity
        if max_context <= 8192:  # GPT-4 original
            return [2000, 4000, 6000, 8000]
        if max_context <= 128000:  # GPT-4 Turbo, GPT-4o
            return [25000, 50000, 100000, 128000]
        if max_context <= 200000:  # Claude 3.5
            return [50000, 100000, 150000, 200000]
        if max_context <= 500000:  # Hypothetical intermediate
            return [100000, 200000, 400000, 500000]
        return [100000, 250000, 500000, 750000]

    def print_available_models(self) -> None:
        """Print information about available models organized by comparison pairs."""
        print("\n🤖 Context Window Advantage Experiment Models:")
        print("=" * 70)

        print("Model Comparison Pairs:")
        print("-" * 30)

        pairs = self.get_model_comparison_pairs()
        for smaller_model, larger_model in pairs:
            try:
                smaller_config = self.get_model_config(smaller_model)
                larger_config = self.get_model_config(larger_model)
                context_sizes = self.get_context_sizes_for_pair(smaller_model, larger_model)

                print(f"📊 {smaller_config.name} vs {larger_config.name}")
                print(
                    f"   Context: {smaller_config.max_context_tokens:,} → {larger_config.max_context_tokens:,} tokens"
                )
                print(f"   Test Sizes: {[f'{size:,}' for size in context_sizes]}")
                print(
                    f"   Cost: ${smaller_config.cost_per_1k_tokens:.4f} vs ${larger_config.cost_per_1k_tokens:.4f} per 1K tokens"
                )
                print()
            except ValueError:
                print(f"⚠️  {smaller_model} vs {larger_model} (not yet available)")
                print()

        print(f"Total Models: {len(self.MODELS)}")
        print(f"Available Pairs: {len([p for p in pairs if all(m in self.MODELS for m in p)])}")

    def test_connection(self, model_name: str = None) -> bool:
        """Test connection to ORQ API."""
        if model_name is None:
            # Use the cheapest model for testing
            cheapest = min(self.MODELS.items(), key=lambda x: x[1].cost_per_1k_tokens)
            model_name = cheapest[0]

        try:
            result = self.query_model(
                model_name=model_name, prompt="Say 'Hello' if you can respond.", max_tokens=10, timeout=30.0
            )

            if result.success:
                print(f"✅ Connection test successful with {model_name}")
                print(f"   Response: {result.response}")
                return True
            print(f"❌ Connection test failed with {model_name}")
            print(f"   Error: {result.error_message}")
            return False

        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


def main():
    """Test the model interface."""
    import os

    # Initialize interface
    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        print("⚠️  ORQ_API_KEY environment variable not set")
        return

    interface = ModelInterface(api_key=api_key)
    interface.print_available_models()

    # Test wait time extraction
    print("\n🔧 Testing wait time extraction:")
    test_messages = [
        "Rate limit exceeded. Please retry after 30 seconds.",
        "HTTP 429: Too many requests. Wait 45 seconds and try again.",
        "Rate limit hit. Try again in 60s.",
        "Please wait 2 minutes before retrying.",
        "Rate limit error after 120 seconds",
    ]

    for msg in test_messages:
        wait_time = interface._extract_wait_time_from_error(msg)
        print(f"  '{msg[:50]}...' → {wait_time}s")

    # Test connection
    if interface.test_connection():
        print("\n🎉 Model interface is ready for experiments!")
    else:
        print("\n❌ Model interface test failed. Check your API key and connection.")


if __name__ == "__main__":
    main()
