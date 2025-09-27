"""
Generation interface with dual API support (NVIDIA + OpenAI) and rate limiting
"""

import time
import asyncio
from collections import deque
from typing import Optional, Any

import httpx
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.progress import track

from ..types import GenerationResult, JudgeResult, PipelineConfig

console = Console()


class RateLimitedNvidiaClient:
    """NVIDIA API client with rate limiting"""
    
    def __init__(self, api_key: str, model: str, rate_limit: int = 40):
        self.api_key = api_key
        self.model = model
        self.rate_limit = rate_limit  # requests per minute
        self.request_times = deque()
        
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
        
        console.print(f"[green]NVIDIA client initialized with {model}, rate limit: {rate_limit}/min[/green]")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def make_request(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Make a single request with rate limiting and retry logic"""
        # Rate limiting
        self._enforce_rate_limit()
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Record successful request time
            self.request_times.append(time.time())
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "response": result["choices"][0]["message"]["content"],
                "model": self.model,
                "token_usage": result.get("usage", {}),
                "latency_ms": latency_ms,
                "success": True
            }
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Extract retry time from response
                retry_delay = 60  # default fallback
                
                # Try to get retry time from headers
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_delay = int(retry_after_header)
                        console.print(f"[yellow]Rate limit hit - retry after {retry_delay}s (from header)[/yellow]")
                    except ValueError:
                        console.print(f"[yellow]Rate limit hit - invalid retry header, using {retry_delay}s default[/yellow]")
                else:
                    # Try to extract from response body
                    try:
                        error_text = e.response.text
                        import re
                        patterns = [
                            r"retry after (\d+) seconds?",
                            r"wait (\d+) seconds?",
                            r"retry in (\d+) seconds?", 
                            r"after (\d+)s",
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, error_text, re.IGNORECASE)
                            if match:
                                retry_delay = int(match.group(1))
                                console.print(f"[yellow]Rate limit hit - retry after {retry_delay}s (from message)[/yellow]")
                                break
                        else:
                            console.print(f"[yellow]Rate limit hit - no retry time found, using {retry_delay}s default[/yellow]")
                    except:
                        console.print(f"[yellow]Rate limit hit - using {retry_delay}s default[/yellow]")
                time.sleep(retry_delay)
                raise
            else:
                console.print(f"[red]HTTP error: {e}[/red]")
                return {
                    "response": f"ERROR: HTTP {e.response.status_code}",
                    "model": self.model,
                    "success": False,
                    "error": str(e)
                }
        except Exception as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return {
                "response": f"ERROR: {str(e)}",
                "model": self.model, 
                "success": False,
                "error": str(e)
            }
    
    def batch_requests(self, requests: list[dict], batch_size: int = 10) -> list[dict]:
        """Process multiple requests with rate limiting"""
        results = []
        
        for i in track(range(0, len(requests), batch_size), description="Processing NVIDIA requests..."):
            batch = requests[i:i + batch_size]
            batch_results = []
            
            for request in batch:
                result = self.make_request(**request)
                batch_results.append(result)
                
                # Add small delay between requests in batch
                if len(batch) > 1:
                    time.sleep(1)
            
            results.extend(batch_results)
            
        return results
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting by tracking request times"""
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # If we're at the rate limit, wait
        if len(self.request_times) >= self.rate_limit:
            oldest_request = self.request_times[0]
            wait_time = 60 - (now - oldest_request) + 1  # +1 for buffer
            
            if wait_time > 0:
                console.print(f"[yellow]Rate limit reached, waiting {wait_time:.1f}s[/yellow]")
                time.sleep(wait_time)


class OpenAIJudgeClient:
    """OpenAI API client for GPT-4.1 judge evaluation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-2025-04-14", base_url: str = None):
        import os
        self.api_key = api_key
        self.model = model
        
        if base_url is None:
            base_url = os.getenv("ORQ_BASE_URL", "https://api.orq.ai/v2/proxy")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # Judge prompts from Chroma
        self.judge_prompts = {
            "niah": """Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
You must only respond with "true" or "false".
If the response is partially incorrect, such as a typo, respond with "false".
If the response contains a snippet of text or additional supporting information, while still maintaining the correct answer without changing the meaning, respond with "true".
If the response starts with anything like "here is the most relevant information in the documents: ", respond with "true". This is fine as long as the following content aligns with the correct answer.

Question: {question}

CORRECT answer: {correct_answer}

Response to judge: {output}

Instructions: Respond with only "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false".""",

            "longmemeval": """Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
In some cases, 0 and "I do not have an answer" are considered to be both correct. 
If both responses say that there is no answer, this should be judged as true.
If the correct answer contains an answer, but the response abstains from answering, this should be judged as false.

Question: {question}

CORRECT answer: {correct_answer}

Response to judge: {output}

Instructions: Respond with only "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false"."""
        }
        
        console.print(f"[green]OpenAI judge client initialized with {model}[/green]")
    
    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    def evaluate_response(self, response: str, question: str, correct_answer: str, 
                         prompt_type: str = "niah") -> JudgeResult:
        """Evaluate a response using the appropriate judge prompt"""
        if prompt_type not in self.judge_prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        prompt = self.judge_prompts[prompt_type].format(
            question=question,
            correct_answer=correct_answer,
            output=response
        )
        
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert judge evaluating the correctness of responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            judge_response = completion.choices[0].message.content.strip().lower()
            is_correct = judge_response == "true"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return JudgeResult(
                is_correct=is_correct,
                judge_response=judge_response,
                question=question,
                correct_answer=correct_answer,
                model_response=response,
                judge_model=self.model,
                metadata={
                    "prompt_type": prompt_type,
                    "latency_ms": latency_ms,
                    "token_usage": dict(completion.usage) if completion.usage else {}
                }
            )
            
        except Exception as e:
            console.print(f"[red]Judge evaluation failed: {e}[/red]")
            return JudgeResult(
                is_correct=False,
                judge_response=f"ERROR: {str(e)}",
                question=question,
                correct_answer=correct_answer,
                model_response=response,
                judge_model=self.model,
                metadata={"error": str(e), "prompt_type": prompt_type}
            )
    
    def batch_evaluate(self, evaluations: list[dict], batch_size: int = 20) -> list[JudgeResult]:
        """Batch evaluate multiple responses"""
        results = []
        
        for i in track(range(0, len(evaluations), batch_size), description="Judge evaluation..."):
            batch = evaluations[i:i + batch_size]
            batch_results = []
            
            for eval_request in batch:
                result = self.evaluate_response(**eval_request)
                batch_results.append(result)
                
                # Small delay to be respectful to API
                time.sleep(0.1)
            
            results.extend(batch_results)
            
            # Longer delay between batches
            if i + batch_size < len(evaluations):
                time.sleep(2)
        
        return results


class GenerationInterface:
    """Interface to LLM APIs with rate limiting and error handling"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize clients
        if not config.nvidia_api_key:
            raise ValueError("NVIDIA API key is required")
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.nvidia_client = RateLimitedNvidiaClient(
            api_key=config.nvidia_api_key,
            model=config.nvidia_model,
            rate_limit=config.nvidia_rate_limit
        )
        
        self.judge_client = OpenAIJudgeClient(
            api_key=config.openai_api_key,
            model=config.judge_model
        )
        
        console.print("[green]GenerationInterface initialized with dual API support[/green]")
    
    def generate_response(self, context: str, query: str, **kwargs) -> GenerationResult:
        """Generate response using NVIDIA API"""
        messages = [
            {"role": "user", "content": context}
        ]
        
        result = self.nvidia_client.make_request(messages, **kwargs)
        
        return GenerationResult(
            response=result["response"],
            context=context,
            query=query,
            model=result["model"],
            metadata=result,
            token_usage=result.get("token_usage"),
            latency_ms=result.get("latency_ms")
        )
    
    def batch_generate(self, contexts: list[str], queries: list[str], 
                      batch_size: int = 10, **kwargs) -> list[GenerationResult]:
        """Batch generation with rate limiting"""
        if len(contexts) != len(queries):
            raise ValueError("Contexts and queries must have the same length")
        
        # Prepare requests
        requests = []
        for context, query in zip(contexts, queries):
            messages = [{"role": "user", "content": context}]
            requests.append({
                "messages": messages,
                "context": context,
                "query": query,
                **kwargs
            })
        
        # Execute batch requests
        results = self.nvidia_client.batch_requests(requests, batch_size)
        
        # Convert to GenerationResult objects
        generation_results = []
        for (context, query), result in zip(zip(contexts, queries), results):
            gen_result = GenerationResult(
                response=result["response"],
                context=context,
                query=query,
                model=result["model"],
                metadata=result,
                token_usage=result.get("token_usage"),
                latency_ms=result.get("latency_ms")
            )
            generation_results.append(gen_result)
        
        return generation_results
    
    def evaluate_response(self, response: str, question: str, correct_answer: str, 
                         experiment_type: str = "niah") -> JudgeResult:
        """Evaluate response using appropriate judge prompt"""
        # Map experiment types to judge prompt types
        prompt_type_mapping = {
            "needle_similarity": "niah",
            "distractor_impact": "niah", 
            "cross_domain": "niah",
            "structure_impact": "niah",
            "longmemeval": "longmemeval"
        }
        
        prompt_type = prompt_type_mapping.get(experiment_type, "niah")
        
        return self.judge_client.evaluate_response(
            response=response,
            question=question,
            correct_answer=correct_answer,
            prompt_type=prompt_type
        )
    
    def batch_evaluate(self, evaluations: list[dict], batch_size: int = 20) -> list[JudgeResult]:
        """Batch evaluate multiple responses"""
        return self.judge_client.batch_evaluate(evaluations, batch_size)
    
    def get_generation_stats(self) -> dict[str, Any]:
        """Get generation client statistics"""
        return {
            "nvidia_model": self.config.nvidia_model,
            "nvidia_rate_limit": self.config.nvidia_rate_limit,
            "nvidia_requests_in_window": len(self.nvidia_client.request_times),
            "judge_model": self.config.judge_model,
            "available_judge_prompts": list(self.judge_client.judge_prompts.keys())
        }
    
    def test_connections(self) -> dict[str, bool]:
        """Test both API connections"""
        results = {}
        
        # Test NVIDIA API
        try:
            test_result = self.nvidia_client.make_request([
                {"role": "user", "content": "Say 'connection test successful'"}
            ], max_tokens=10)
            results["nvidia_connection"] = test_result.get("success", False)
        except Exception as e:
            console.print(f"[red]NVIDIA connection test failed: {e}[/red]")
            results["nvidia_connection"] = False
        
        # Test OpenAI API
        try:
            judge_result = self.judge_client.evaluate_response(
                response="Paris",
                question="What is the capital of France?",
                correct_answer="Paris",
                prompt_type="niah"
            )
            results["openai_connection"] = judge_result.is_correct
        except Exception as e:
            console.print(f"[red]OpenAI connection test failed: {e}[/red]")
            results["openai_connection"] = False
        
        return results