#!/usr/bin/env python3
"""
LongMemEval Context Grouper for Context Window Advantage Experiments

This module analyzes LongMemEval questions and groups them by optimal context length
requirements. It provides insights into how question complexity interacts with
context size requirements for real-world validation of context window experiments.

Context Length Bins:
- Short (0-25K tokens)
- Medium (25K-75K tokens)
- Long (75K-150K tokens)
- Extended (150K+ tokens)

Usage:
    python context_grouper.py --input-path ../data/longmemeval/
    python context_grouper.py --analyze-complexity --output-dir data/longmemeval_grouped/
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import tiktoken


@dataclass
class ContextRequirement:
    """Represents context requirements for a question."""

    question_id: str
    estimated_tokens: int
    actual_tokens: int | None
    context_bin: str  # 'short', 'medium', 'long', 'extended'
    required_documents: list[str]
    spans_multiple_docs: bool


@dataclass
class QuestionComplexity:
    """Analysis of question complexity factors."""

    question_id: str
    question_type: str  # 'factual', 'reasoning', 'synthesis', 'multi_hop'
    reasoning_steps: int
    entities_mentioned: int
    temporal_reasoning: bool
    numerical_reasoning: bool
    multi_document: bool
    complexity_score: float  # 0.0 to 1.0


@dataclass
class LongMemEvalQuestion:
    """Enhanced LongMemEval question with context analysis."""

    id: str
    question: str
    answer: str
    context_docs: list[str]
    gold_spans: list[str]  # If available
    domain: str
    difficulty: str | None

    # Context analysis
    context_requirement: ContextRequirement
    complexity: QuestionComplexity

    # Token counts
    question_tokens: int
    context_tokens: int
    total_tokens: int


@dataclass
class ContextGroup:
    """A group of questions with similar context requirements."""

    bin_name: str
    token_range: tuple[int, int]  # (min_tokens, max_tokens)
    questions: list[LongMemEvalQuestion]

    # Statistics
    avg_context_tokens: float
    median_context_tokens: float
    avg_complexity_score: float
    question_types: dict[str, int]
    domains: dict[str, int]


class LongMemEvalContextGrouper:
    """Groups LongMemEval questions by context length requirements."""

    # Context length bins (in tokens)
    CONTEXT_BINS = {
        "short": (0, 25000),
        "medium": (25001, 75000),
        "long": (75001, 150000),
        "extended": (150001, float("inf")),
    }

    def __init__(self, data_dir: Path = None):
        """Initialize the context grouper."""
        if data_dir is None:
            # Look for LongMemEval data in the standard location
            data_dir = Path(__file__).parent.parent.parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.encoding = tiktoken.encoding_for_model("gpt-4o")

        print("📊 Initialized LongMemEval Context Grouper")
        print(f"📁 Data directory: {self.data_dir}")

        # Try to locate LongMemEval data
        self.longmemeval_path = self._find_longmemeval_data()
        self._loaded_documents = {}  # Cache for loaded document content
        
        if self.longmemeval_path:
            print(f"🔍 Found LongMemEval data: {self.longmemeval_path}")
            # Try to preload document content if available
            self._preload_document_content()

    def _find_longmemeval_data(self) -> Path | None:
        """Find LongMemEval dataset in the data directory."""
        possible_paths = [
            self.data_dir / "longmemeval",
            self.data_dir / "LongMemEval",
            self.data_dir / "long_mem_eval",
            self.data_dir / "datasets" / "longmemeval",
            self.data_dir / "raw" / "longmemeval",
        ]

        for path in possible_paths:
            if path.exists():
                # Look for JSON or CSV files
                data_files = (
                    list(path.glob("*.json"))
                    + list(path.glob("**/*.json"))
                    + list(path.glob("*.csv"))
                    + list(path.glob("**/*.csv"))
                )
                if data_files:
                    return path

        print(f"⚠️  LongMemEval data not found in {self.data_dir}")
        print(f"Expected locations: {[str(p) for p in possible_paths]}")
        return None
    
    def _preload_document_content(self):
        """Try to preload document content from LongMemEval data files."""
        if not self.longmemeval_path:
            return
            
        # Look for document content in JSON files
        json_files = list(self.longmemeval_path.glob("*.json")) + list(self.longmemeval_path.glob("**/*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Look for embedded documents in various formats
                if isinstance(data, dict):
                    # Check for documents field
                    if 'documents' in data:
                        docs = data['documents']
                        if isinstance(docs, dict):
                            self._loaded_documents.update(docs)
                        elif isinstance(docs, list):
                            for i, doc in enumerate(docs):
                                if isinstance(doc, dict) and 'id' in doc and 'content' in doc:
                                    self._loaded_documents[doc['id']] = doc['content']
                                elif isinstance(doc, str):
                                    self._loaded_documents[f"doc_{i}"] = doc
                    
                    # Check for context field in questions
                    if 'questions' in data or isinstance(data, list):
                        questions = data.get('questions', data if isinstance(data, list) else [])
                        for q in questions:
                            if isinstance(q, dict) and 'context' in q:
                                context = q['context']
                                if isinstance(context, str):
                                    # Use question id as document identifier
                                    doc_id = q.get('id', q.get('question_id', f"context_{len(self._loaded_documents)}"))
                                    self._loaded_documents[doc_id] = context
                                    
            except Exception as e:
                print(f"⚠️  Could not preload documents from {json_file}: {e}")
                continue
        
        if self._loaded_documents:
            print(f"📄 Preloaded {len(self._loaded_documents)} document contents")

    def load_longmemeval_data(self, file_path: Path | None = None) -> list[dict]:
        """Load LongMemEval dataset from JSON or CSV files."""
        if file_path:
            data_path = file_path
        elif self.longmemeval_path:
            # Find the main dataset file (prefer CSV, fallback to JSON)
            csv_files = list(self.longmemeval_path.glob("*.csv")) + list(self.longmemeval_path.glob("**/*.csv"))
            json_files = list(self.longmemeval_path.glob("*.json")) + list(self.longmemeval_path.glob("**/*.json"))

            if csv_files:
                # Take the largest CSV file (likely the main dataset)
                data_path = max(csv_files, key=lambda p: p.stat().st_size)
            elif json_files:
                # Fallback to JSON
                data_path = max(json_files, key=lambda p: p.stat().st_size)
            else:
                raise FileNotFoundError(f"No JSON or CSV files found in {self.longmemeval_path}")
        else:
            raise FileNotFoundError("No LongMemEval data path available")

        print(f"📖 Loading LongMemEval data from {data_path}")

        if data_path.suffix.lower() == ".csv":
            # Try different CSV loading approaches
            questions = None

            # First try: Python csv module with increased field size limit
            try:
                # Increase field size limit for large CSV fields
                csv.field_size_limit(2**31 - 1)  # Set to maximum possible value
                with open(data_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f, quoting=csv.QUOTE_MINIMAL)
                    questions = [row for row in reader]
                print(f"📋 Loaded {len(questions)} questions from CSV using Python csv module")
            except Exception as e:
                print(f"⚠️  Python csv failed ({e}), trying pandas...")

                # Second try: pandas with different options
                try:
                    df_pandas = pd.read_csv(data_path, quoting=csv.QUOTE_MINIMAL, engine="python")
                    questions = df_pandas.to_dict("records")
                    print(f"📋 Loaded {len(questions)} questions from CSV using pandas")
                except Exception as e2:
                    print(f"⚠️  Pandas also failed ({e2}), trying Polars...")

                    # Third try: Polars with different options
                    try:
                        df = pl.read_csv(data_path, quote_char='"', separator=",", ignore_errors=True)
                        questions = df.to_dicts()
                        print(f"📋 Loaded {len(questions)} questions from CSV using Polars")
                    except Exception as e3:
                        raise ValueError(f"All CSV loading methods failed: csv={e}, pandas={e2}, polars={e3}")

            if not questions:
                raise ValueError("No questions loaded from CSV")
        else:
            # Load JSON
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):
                if "questions" in data:
                    questions = data["questions"]
                elif "data" in data:
                    questions = data["data"]
                else:
                    # Assume the dict values are questions
                    questions = list(data.values())
            elif isinstance(data, list):
                questions = data
            else:
                raise ValueError(f"Unexpected data format in {data_path}")

            print(f"📋 Loaded {len(questions)} questions from JSON")

        return questions

    def calculate_context_tokens(self, context_docs: list[str]) -> int:
        """Calculate actual token count for context documents."""
        total_content = ""
        
        for doc_identifier in context_docs:
            # Load the actual document content
            doc_content = self._load_document_content(doc_identifier)
            if doc_content:
                total_content += doc_content + "\n\n"
            else:
                # If we can't find the document, this is a problem - log it
                print(f"⚠️  Warning: Could not load document content for '{doc_identifier}'")
                print(f"   This will result in inaccurate token counts.")
        
        # Calculate actual tokens from the real content
        if total_content:
            return len(self.encoding.encode(total_content))
        else:
            print(f"❌ Error: No document content found for {context_docs}")
            return 0

    def _load_document_content(self, doc_identifier: str) -> str | None:
        """Load content for a document identifier."""
        # Approach 1: Check preloaded documents cache first
        if hasattr(self, '_loaded_documents') and doc_identifier in self._loaded_documents:
            return self._loaded_documents[doc_identifier]
        
        # Approach 2: If doc_identifier looks like actual content (long text), use it directly
        if len(doc_identifier) > 500:  # Likely actual content, not just an identifier
            return doc_identifier
            
        # Approach 3: Try file system paths
        if self.longmemeval_path:
            potential_paths = [
                self.longmemeval_path / doc_identifier,
                self.longmemeval_path / "documents" / doc_identifier,
                self.longmemeval_path / "docs" / doc_identifier,
                self.longmemeval_path / "context" / doc_identifier,
            ]
            
            # Try with common extensions
            for base_path in potential_paths:
                for ext in ['', '.txt', '.md', '.json']:
                    path = Path(str(base_path) + ext)
                    if path.exists() and path.is_file():
                        try:
                            with open(path, encoding="utf-8") as f:
                                content = f.read()
                                if content.strip():  # Only return non-empty content
                                    return content
                        except Exception as e:
                            print(f"⚠️  Error reading {path}: {e}")
            
        # If no content found, return None
        return None


    def analyze_question_complexity(self, question: str, answer: str, context_docs: list[str]) -> QuestionComplexity:
        """Analyze the complexity of a question."""
        question_lower = question.lower()

        # Determine question type
        question_type = self._classify_question_type(question, answer)

        # Count reasoning steps (heuristic)
        reasoning_steps = self._estimate_reasoning_steps(question, answer)

        # Count entities mentioned
        entities_mentioned = len(re.findall(r"\b[A-Z][a-zA-Z]*\b", question))

        # Check for temporal reasoning
        temporal_keywords = ["when", "before", "after", "during", "since", "until", "year", "date", "time"]
        temporal_reasoning = any(word in question_lower for word in temporal_keywords)

        # Check for numerical reasoning
        numerical_reasoning = bool(re.search(r"\d+|number|count|amount|percentage", question_lower))

        # Multi-document reasoning
        multi_document = len(context_docs) > 1

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            question_type, reasoning_steps, entities_mentioned, temporal_reasoning, numerical_reasoning, multi_document
        )

        return QuestionComplexity(
            question_id="",  # Will be set later
            question_type=question_type,
            reasoning_steps=reasoning_steps,
            entities_mentioned=entities_mentioned,
            temporal_reasoning=temporal_reasoning,
            numerical_reasoning=numerical_reasoning,
            multi_document=multi_document,
            complexity_score=complexity_score,
        )

    def _classify_question_type(self, question: str, answer: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()

        # Keywords for different question types
        if any(word in question_lower for word in ["what", "who", "where", "when"]):
            if any(word in question_lower for word in ["why", "how", "explain", "analyze"]):
                return "reasoning"
            return "factual"
        if any(word in question_lower for word in ["compare", "contrast", "relationship", "connect"]):
            return "synthesis"
        if any(word in question_lower for word in ["based on", "according to", "from the"]):
            return "multi_hop"
        return "reasoning"

    def _estimate_reasoning_steps(self, question: str, answer: str) -> int:
        """Estimate the number of reasoning steps required."""
        question_lower = question.lower()

        # Count complexity indicators
        steps = 1  # Base step

        if any(word in question_lower for word in ["why", "how", "explain"]):
            steps += 1
        if any(word in question_lower for word in ["compare", "contrast", "relationship"]):
            steps += 1
        if any(word in question_lower for word in ["and", "also", "furthermore"]):
            steps += question_lower.count(" and ")
        if len(answer.split(".")) > 2:  # Multi-sentence answer
            steps += 1

        return min(steps, 5)  # Cap at 5 steps

    def _calculate_complexity_score(
        self,
        question_type: str,
        reasoning_steps: int,
        entities_mentioned: int,
        temporal_reasoning: bool,
        numerical_reasoning: bool,
        multi_document: bool,
    ) -> float:
        """Calculate overall complexity score (0.0 to 1.0)."""
        score = 0.0

        # Question type contribution
        type_scores = {"factual": 0.2, "reasoning": 0.4, "synthesis": 0.6, "multi_hop": 0.8}
        score += type_scores.get(question_type, 0.4)

        # Reasoning steps (0.0 to 0.3)
        score += min(reasoning_steps / 5.0, 1.0) * 0.3

        # Entity complexity (0.0 to 0.1)
        score += min(entities_mentioned / 10.0, 1.0) * 0.1

        # Boolean factors (0.05 each)
        if temporal_reasoning:
            score += 0.05
        if numerical_reasoning:
            score += 0.05
        if multi_document:
            score += 0.1

        return min(score, 1.0)

    def assign_context_bin(self, token_count: int) -> str:
        """Assign a context bin based on token count."""
        for bin_name, (min_tokens, max_tokens) in self.CONTEXT_BINS.items():
            if min_tokens <= token_count <= max_tokens:
                return bin_name

        # Fallback to extended if somehow out of range
        return "extended"

    def process_questions(self, questions: list[dict]) -> list[LongMemEvalQuestion]:
        """Process raw questions into analyzed LongMemEval questions."""
        processed_questions = []

        print(f"🔄 Processing {len(questions)} questions...")

        for i, q_data in enumerate(questions):
            try:
                # Extract basic question data (adapt keys for CSV format)
                question_id = q_data.get("custom_id", q_data.get("id", f"q_{i:04d}"))
                question = q_data.get("question", q_data.get("input", ""))
                answer = q_data.get("answer", q_data.get("output", ""))

                # For CSV format, we need to extract context from full_prompt
                if "full_prompt" in q_data:
                    # Extract context documents from full_prompt if available
                    full_prompt = q_data["full_prompt"]
                    context_docs = [full_prompt]  # Use the full prompt as context
                else:
                    context_docs = q_data.get("context", q_data.get("documents", []))

                gold_spans = q_data.get("gold_spans", [])
                domain = q_data.get("domain", q_data.get("category", "longmemeval"))
                difficulty = q_data.get("difficulty", q_data.get("level"))

                # Use token_count from CSV if available
                if q_data.get("token_count"):
                    estimated_tokens = int(q_data["token_count"])
                    use_estimated = True
                else:
                    estimated_tokens = None
                    use_estimated = False

                # Ensure context_docs is a list
                if isinstance(context_docs, str):
                    context_docs = [context_docs]

                # Calculate actual context requirements
                if not use_estimated:
                    actual_tokens = self.calculate_context_tokens(context_docs)
                else:
                    actual_tokens = estimated_tokens
                context_bin = self.assign_context_bin(actual_tokens)

                context_requirement = ContextRequirement(
                    question_id=question_id,
                    estimated_tokens=actual_tokens if use_estimated else None,
                    actual_tokens=actual_tokens,
                    context_bin=context_bin,
                    required_documents=context_docs,
                    spans_multiple_docs=len(context_docs) > 1,
                )

                # Analyze complexity
                complexity = self.analyze_question_complexity(question, answer, context_docs)
                complexity.question_id = question_id

                # Calculate token counts
                question_tokens = len(self.encoding.encode(question))
                context_tokens = actual_tokens
                total_tokens = question_tokens + context_tokens

                # Create processed question
                processed_question = LongMemEvalQuestion(
                    id=question_id,
                    question=question,
                    answer=answer,
                    context_docs=context_docs,
                    gold_spans=gold_spans,
                    domain=domain,
                    difficulty=difficulty,
                    context_requirement=context_requirement,
                    complexity=complexity,
                    question_tokens=question_tokens,
                    context_tokens=context_tokens,
                    total_tokens=total_tokens,
                )

                processed_questions.append(processed_question)

                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(questions)} questions...")

            except Exception as e:
                print(f"⚠️  Error processing question {i}: {e}")
                continue

        print(f"✅ Successfully processed {len(processed_questions)} questions")
        return processed_questions

    def create_context_groups(self, questions: list[LongMemEvalQuestion]) -> dict[str, ContextGroup]:
        """Group questions by context length bins."""
        groups = {}

        # Group questions by bin
        binned_questions = defaultdict(list)
        for question in questions:
            bin_name = question.context_requirement.context_bin
            binned_questions[bin_name].append(question)

        # Create ContextGroup objects
        for bin_name in self.CONTEXT_BINS.keys():
            questions_in_bin = binned_questions[bin_name]

            if questions_in_bin:
                # Calculate statistics
                context_tokens = [q.context_tokens for q in questions_in_bin]
                avg_context_tokens = statistics.mean(context_tokens)
                median_context_tokens = statistics.median(context_tokens)
                avg_complexity_score = statistics.mean([q.complexity.complexity_score for q in questions_in_bin])

                # Count question types and domains
                question_types = defaultdict(int)
                domains = defaultdict(int)

                for q in questions_in_bin:
                    question_types[q.complexity.question_type] += 1
                    domains[q.domain] += 1

                groups[bin_name] = ContextGroup(
                    bin_name=bin_name,
                    token_range=self.CONTEXT_BINS[bin_name],
                    questions=questions_in_bin,
                    avg_context_tokens=avg_context_tokens,
                    median_context_tokens=median_context_tokens,
                    avg_complexity_score=avg_complexity_score,
                    question_types=dict(question_types),
                    domains=dict(domains),
                )
            else:
                # Create empty group
                groups[bin_name] = ContextGroup(
                    bin_name=bin_name,
                    token_range=self.CONTEXT_BINS[bin_name],
                    questions=[],
                    avg_context_tokens=0.0,
                    median_context_tokens=0.0,
                    avg_complexity_score=0.0,
                    question_types={},
                    domains={},
                )

        return groups

    def print_analysis_summary(self, groups: dict[str, ContextGroup]) -> None:
        """Print summary of context grouping analysis."""
        print("\n" + "=" * 60)
        print("📊 LONGMEMEVAL CONTEXT ANALYSIS SUMMARY")
        print("=" * 60)

        total_questions = sum(len(group.questions) for group in groups.values())
        print(f"Total Questions Analyzed: {total_questions}")
        print()

        for bin_name, group in groups.items():
            token_range = group.token_range
            range_str = (
                f"{token_range[0]:,}-{token_range[1]:,}" if token_range[1] != float("inf") else f"{token_range[0]:,}+"
            )

            print(f"📋 {bin_name.upper()} Context ({range_str} tokens):")
            print(f"   Questions: {len(group.questions)}")

            if group.questions:
                print(f"   Avg Context Tokens: {group.avg_context_tokens:,.0f}")
                print(f"   Median Context Tokens: {group.median_context_tokens:,.0f}")
                print(f"   Avg Complexity Score: {group.avg_complexity_score:.2f}")

                # Top question types
                if group.question_types:
                    top_types = sorted(group.question_types.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"   Top Question Types: {', '.join([f'{t}({c})' for t, c in top_types])}")

                # Top domains
                if group.domains:
                    top_domains = sorted(group.domains.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"   Top Domains: {', '.join([f'{d}({c})' for d, c in top_domains])}")

            print()

    def save_grouped_data(self, groups: dict[str, ContextGroup], output_dir: Path) -> None:
        """Save grouped data to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each group separately
        for bin_name, group in groups.items():
            if group.questions:
                # Save questions in this group
                group_path = output_dir / f"{bin_name}_context_questions.json"

                questions_data = []
                for question in group.questions:
                    questions_data.append(asdict(question))

                with open(group_path, "w") as f:
                    json.dump(questions_data, f, indent=2)

                print(f"💾 Saved {len(group.questions)} {bin_name} questions to {group_path}")

        # Save overall summary
        summary_path = output_dir / "context_grouping_summary.json"
        summary_data = {}

        for bin_name, group in groups.items():
            summary_data[bin_name] = {
                "token_range": group.token_range
                if group.token_range[1] != float("inf")
                else [group.token_range[0], -1],
                "question_count": len(group.questions),
                "avg_context_tokens": group.avg_context_tokens,
                "median_context_tokens": group.median_context_tokens,
                "avg_complexity_score": group.avg_complexity_score,
                "question_types": group.question_types,
                "domains": group.domains,
            }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"📊 Saved analysis summary to {summary_path}")


def main() -> None:
    """Command line interface for context grouping."""
    parser = argparse.ArgumentParser(description="Group LongMemEval questions by context length")
    parser.add_argument("--input-path", type=Path, help="Path to LongMemEval dataset")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: ../data/longmemeval_grouped/)")
    parser.add_argument("--data-dir", type=Path, help="Data directory containing LongMemEval")
    parser.add_argument("--analyze-complexity", action="store_true", help="Perform detailed complexity analysis")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).parent.parent / "data" / "longmemeval_grouped"

    try:
        # Initialize grouper
        grouper = LongMemEvalContextGrouper(data_dir=args.data_dir)

        # Load data
        if args.input_path:
            questions_data = grouper.load_longmemeval_data(args.input_path)
        else:
            questions_data = grouper.load_longmemeval_data()

        # Process questions
        processed_questions = grouper.process_questions(questions_data)

        # Create context groups
        groups = grouper.create_context_groups(processed_questions)

        # Print analysis
        grouper.print_analysis_summary(groups)

        # Save results
        grouper.save_grouped_data(groups, args.output_dir)

        print("\n✅ Context grouping complete!")
        print(f"📁 Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Error in context grouping: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
