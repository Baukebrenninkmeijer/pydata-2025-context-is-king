#!/usr/bin/env python3
"""
Human Validation Framework for Reranking Value Experiment

This module generates subsets for human validation to verify:
1. LLM Judge calibration and accuracy
2. Answer quality assessment beyond correctness
3. Approach bias detection
4. Edge case handling
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class ValidationCategory(Enum):
    """Categories for human validation."""
    JUDGE_CALIBRATION = "judge_calibration"
    ANSWER_QUALITY = "answer_quality"
    APPROACH_BIAS = "approach_bias"
    EDGE_CASES = "edge_cases"


@dataclass
class ValidationExample:
    """Single example for human validation."""
    example_id: str
    category: ValidationCategory
    question: str
    ground_truth: str
    model_response: str
    judge_evaluation: Dict[str, Any]
    approach_used: str
    context_preview: str
    metadata: Dict[str, Any]
    
    # To be filled by human validators
    human_correctness: Optional[bool] = None
    human_quality_score: Optional[int] = None  # 1-5 scale
    human_completeness: Optional[int] = None   # 1-5 scale
    human_clarity: Optional[int] = None        # 1-5 scale
    human_notes: Optional[str] = None


@dataclass
class ValidationSubset:
    """Collection of validation examples."""
    subset_id: str
    category: ValidationCategory
    examples: List[ValidationExample]
    instructions: str
    metadata: Dict[str, Any]


class HumanValidationGenerator:
    """Generates human validation subsets from experiment results."""
    
    def __init__(self, experiment_results_path: str):
        """Initialize with experiment results."""
        self.results_path = Path(experiment_results_path)
        self.results_data = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load experiment results from JSON file."""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
            
        with open(self.results_path, 'r') as f:
            return json.load(f)
    
    def generate_validation_subsets(self, 
                                   total_examples: int = 120,
                                   seed: int = 42) -> List[ValidationSubset]:
        """Generate all validation subsets."""
        random.seed(seed)
        
        subsets = []
        
        # Distribution of examples across categories
        distribution = {
            ValidationCategory.JUDGE_CALIBRATION: 40,  # 33%
            ValidationCategory.ANSWER_QUALITY: 30,     # 25% 
            ValidationCategory.APPROACH_BIAS: 30,      # 25%
            ValidationCategory.EDGE_CASES: 20          # 17%
        }
        
        console.print("[blue]Generating human validation subsets...[/blue]")
        
        for category, count in distribution.items():
            console.print(f"[dim]Generating {count} examples for {category.value}[/dim]")
            
            if category == ValidationCategory.JUDGE_CALIBRATION:
                subset = self._generate_judge_calibration_subset(count)
            elif category == ValidationCategory.ANSWER_QUALITY:
                subset = self._generate_answer_quality_subset(count)
            elif category == ValidationCategory.APPROACH_BIAS:
                subset = self._generate_approach_bias_subset(count)
            elif category == ValidationCategory.EDGE_CASES:
                subset = self._generate_edge_cases_subset(count)
            
            subsets.append(subset)
        
        return subsets
    
    def _generate_judge_calibration_subset(self, count: int) -> ValidationSubset:
        """Generate examples to validate judge accuracy."""
        
        examples = []
        results = self.results_data.get("results", [])
        
        # Stratified sampling across different judge confidence levels
        high_confidence = [r for r in results if r["judge_evaluation"]["confidence"] > 0.8]
        medium_confidence = [r for r in results if 0.4 <= r["judge_evaluation"]["confidence"] <= 0.8]
        low_confidence = [r for r in results if r["judge_evaluation"]["confidence"] < 0.4]
        
        # Sample proportionally
        high_samples = random.sample(high_confidence, min(count//2, len(high_confidence)))
        medium_samples = random.sample(medium_confidence, min(count//3, len(medium_confidence)))
        low_samples = random.sample(low_confidence, min(count//6, len(low_confidence)))
        
        all_samples = high_samples + medium_samples + low_samples
        
        for i, result in enumerate(all_samples[:count]):
            example = ValidationExample(
                example_id=f"judge_cal_{i+1:03d}",
                category=ValidationCategory.JUDGE_CALIBRATION,
                question=result["question"],
                ground_truth=result["expected_answer"],
                model_response=result["model_answer"],
                judge_evaluation=result["judge_evaluation"],
                approach_used=result["approach"],
                context_preview=result["approach_metadata"].get("context_preview", "")[:200],
                metadata={
                    "judge_confidence": result["judge_evaluation"]["confidence"],
                    "judge_correctness": result["judge_evaluation"]["is_correct"],
                    "context_group": result["context_group"],
                    "question_type": result["question_type"]
                }
            )
            examples.append(example)
        
        instructions = """
        **Judge Calibration Validation Instructions**
        
        For each example, evaluate:
        1. **Correctness**: Is the model's answer actually correct? (Yes/No)
        2. **Judge Agreement**: Do you agree with the LLM judge's assessment?
        
        Focus on:
        - Factual accuracy vs ground truth
        - Partial credit for incomplete but correct answers
        - Different valid phrasings of the same answer
        - Judge's reasoning quality
        
        Rate on a 5-point scale:
        - Quality: 1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent
        """
        
        return ValidationSubset(
            subset_id="judge_calibration",
            category=ValidationCategory.JUDGE_CALIBRATION,
            examples=examples,
            instructions=instructions,
            metadata={"total_examples": len(examples), "confidence_distribution": {
                "high": len(high_samples),
                "medium": len(medium_samples), 
                "low": len(low_samples)
            }}
        )
    
    def _generate_answer_quality_subset(self, count: int) -> ValidationSubset:
        """Generate examples for answer quality assessment."""
        
        examples = []
        results = self.results_data.get("results", [])
        
        # Sample across different approaches and question types
        stratified_samples = {}
        for result in results:
            key = (result["approach"], result["question_type"])
            if key not in stratified_samples:
                stratified_samples[key] = []
            stratified_samples[key].append(result)
        
        # Sample evenly across strata
        samples_per_stratum = max(1, count // len(stratified_samples))
        all_samples = []
        
        for stratum_results in stratified_samples.values():
            sampled = random.sample(stratum_results, min(samples_per_stratum, len(stratum_results)))
            all_samples.extend(sampled)
        
        for i, result in enumerate(all_samples[:count]):
            example = ValidationExample(
                example_id=f"quality_{i+1:03d}",
                category=ValidationCategory.ANSWER_QUALITY,
                question=result["question"],
                ground_truth=result["expected_answer"],
                model_response=result["model_answer"],
                judge_evaluation=result["judge_evaluation"],
                approach_used=result["approach"],
                context_preview=result["approach_metadata"].get("context_preview", "")[:200],
                metadata={
                    "approach": result["approach"],
                    "question_type": result["question_type"],
                    "context_tokens": result["context_measurement"]["total_tokens"]
                }
            )
            examples.append(example)
        
        instructions = """
        **Answer Quality Assessment Instructions**
        
        Rate each answer on three dimensions (1-5 scale):
        
        1. **Correctness**: Is the information accurate?
           - 5: Completely correct
           - 4: Mostly correct with minor issues
           - 3: Partially correct
           - 2: Mostly incorrect
           - 1: Completely incorrect
        
        2. **Completeness**: Does it fully answer the question?
           - 5: Comprehensive, addresses all parts
           - 4: Good coverage, minor gaps
           - 3: Adequate, some missing elements
           - 2: Incomplete, significant gaps
           - 1: Very incomplete or irrelevant
        
        3. **Clarity**: Is it well-written and clear?
           - 5: Excellent clarity and organization
           - 4: Good clarity, easy to follow
           - 3: Adequate clarity
           - 2: Somewhat unclear or confusing
           - 1: Very unclear or poorly written
        """
        
        return ValidationSubset(
            subset_id="answer_quality",
            category=ValidationCategory.ANSWER_QUALITY,
            examples=examples,
            instructions=instructions,
            metadata={"total_examples": len(examples)}
        )
    
    def _generate_approach_bias_subset(self, count: int) -> ValidationSubset:
        """Generate blind comparison examples to detect approach bias."""
        
        examples = []
        results = self.results_data.get("results", [])
        
        # Group by question to get different approaches for same question
        questions_by_id = {}
        for result in results:
            qid = result["question_id"]
            if qid not in questions_by_id:
                questions_by_id[qid] = []
            questions_by_id[qid].append(result)
        
        # Find questions answered by multiple approaches
        multi_approach_questions = {
            qid: results_list for qid, results_list in questions_by_id.items() 
            if len(results_list) >= 3  # At least 3 different approaches
        }
        
        sampled_questions = list(multi_approach_questions.keys())[:count//3]
        
        example_idx = 0
        for qid in sampled_questions:
            question_results = multi_approach_questions[qid]
            question_text = question_results[0]["question"]
            ground_truth = question_results[0]["expected_answer"]
            
            # Create blind comparison set
            responses = []
            for result in question_results:
                responses.append({
                    "response": result["model_answer"],
                    "true_approach": result["approach"],
                    "context_preview": result["approach_metadata"].get("context_preview", "")[:100]
                })
            
            # Shuffle responses to blind the approaches
            shuffled_responses = responses.copy()
            random.shuffle(shuffled_responses)
            
            example = ValidationExample(
                example_id=f"bias_{example_idx+1:03d}",
                category=ValidationCategory.APPROACH_BIAS,
                question=question_text,
                ground_truth=ground_truth,
                model_response=json.dumps([r["response"] for r in shuffled_responses]),
                judge_evaluation={"note": "Multiple responses for blind comparison"},
                approach_used="multiple_blind",
                context_preview="Multiple contexts compared",
                metadata={
                    "question_id": qid,
                    "num_approaches": len(responses),
                    "true_approach_order": [r["true_approach"] for r in shuffled_responses]
                }
            )
            examples.append(example)
            example_idx += 1
        
        instructions = """
        **Approach Bias Detection Instructions**
        
        For each question, you'll see multiple responses generated by different approaches.
        The approaches are NOT identified - evaluate each response independently.
        
        For each response (A, B, C, etc.), rate:
        1. **Overall Quality** (1-5): How good is this response overall?
        2. **Preference Ranking**: Rank responses from best to worst
        3. **Style Notes**: Any distinctive characteristics you notice
        
        This helps detect if the judge has systematic bias toward certain response styles
        or if certain approaches produce consistently better/worse responses.
        """
        
        return ValidationSubset(
            subset_id="approach_bias", 
            category=ValidationCategory.APPROACH_BIAS,
            examples=examples,
            instructions=instructions,
            metadata={"total_examples": len(examples), "questions_compared": len(sampled_questions)}
        )
    
    def _generate_edge_cases_subset(self, count: int) -> ValidationSubset:
        """Generate examples focusing on edge cases and difficult scenarios."""
        
        examples = []
        results = self.results_data.get("results", [])
        
        # Identify edge cases based on various criteria
        edge_cases = []
        
        for result in results:
            judge_eval = result["judge_evaluation"]
            
            # Low confidence cases
            if judge_eval["confidence"] < 0.3:
                edge_cases.append(("low_confidence", result))
            
            # Confidence-correctness mismatch
            if judge_eval["confidence"] > 0.8 and not judge_eval["is_correct"]:
                edge_cases.append(("overconfident_wrong", result))
            if judge_eval["confidence"] < 0.4 and judge_eval["is_correct"]:
                edge_cases.append(("underconfident_right", result))
            
            # Very short or very long responses
            response_len = len(result["model_answer"].split())
            if response_len < 5:
                edge_cases.append(("very_short", result))
            elif response_len > 100:
                edge_cases.append(("very_long", result))
            
            # Multi-hop questions (complex reasoning)
            if result["question_type"] == "multi_hop":
                edge_cases.append(("complex_reasoning", result))
        
        # Sample edge cases
        sampled_cases = random.sample(edge_cases, min(count, len(edge_cases)))
        
        for i, (case_type, result) in enumerate(sampled_cases):
            example = ValidationExample(
                example_id=f"edge_{i+1:03d}",
                category=ValidationCategory.EDGE_CASES,
                question=result["question"],
                ground_truth=result["expected_answer"],
                model_response=result["model_answer"],
                judge_evaluation=result["judge_evaluation"],
                approach_used=result["approach"],
                context_preview=result["approach_metadata"].get("context_preview", "")[:200],
                metadata={
                    "edge_case_type": case_type,
                    "judge_confidence": result["judge_evaluation"]["confidence"],
                    "response_length": len(result["model_answer"].split()),
                    "question_type": result["question_type"]
                }
            )
            examples.append(example)
        
        instructions = """
        **Edge Cases Validation Instructions**
        
        These are challenging cases where the LLM judge may struggle. Pay special attention to:
        
        1. **Confidence Calibration**: Does judge confidence match actual correctness?
        2. **Partial Credit**: Should incomplete but partially correct answers get credit?
        3. **Different Valid Answers**: Are there multiple ways to correctly answer?
        4. **Context Dependency**: Is the answer reasonable given the available context?
        
        Special focus areas:
        - Very short responses (may be correct but terse)
        - Very long responses (may contain correct info buried in text)
        - Complex multi-step reasoning questions
        - Cases where judge seems overconfident or underconfident
        """
        
        return ValidationSubset(
            subset_id="edge_cases",
            category=ValidationCategory.EDGE_CASES, 
            examples=examples,
            instructions=instructions,
            metadata={"total_examples": len(examples), "edge_case_types": [c[0] for c in sampled_cases]}
        )
    
    def save_validation_subsets(self, subsets: List[ValidationSubset], output_dir: str) -> Dict[str, str]:
        """Save validation subsets to files."""
        output_path = Path(output_dir) / "human_validation"
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for subset in subsets:
            # Save as JSON for programmatic access
            json_file = output_path / f"{subset.subset_id}.json"
            with open(json_file, 'w') as f:
                json.dump(asdict(subset), f, indent=2, default=str)
            saved_files[f"{subset.subset_id}_json"] = str(json_file)
            
            # Save as CSV for human annotation
            csv_file = output_path / f"{subset.subset_id}_annotation.csv"
            self._save_as_csv(subset, csv_file)
            saved_files[f"{subset.subset_id}_csv"] = str(csv_file)
            
            # Save instructions as markdown
            md_file = output_path / f"{subset.subset_id}_instructions.md"
            with open(md_file, 'w') as f:
                f.write(f"# {subset.subset_id.title()} Validation Instructions\n\n")
                f.write(subset.instructions)
                f.write(f"\n\n## Metadata\n\n```json\n{json.dumps(subset.metadata, indent=2)}\n```")
            saved_files[f"{subset.subset_id}_instructions"] = str(md_file)
        
        # Save summary
        summary_file = output_path / "validation_summary.json"
        summary = {
            "total_subsets": len(subsets),
            "total_examples": sum(len(s.examples) for s in subsets),
            "categories": {s.category.value: len(s.examples) for s in subsets},
            "files_created": saved_files,
            "instructions": "Use CSV files for human annotation, JSON files contain full data structure"
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files["summary"] = str(summary_file)
        
        return saved_files
    
    def _save_as_csv(self, subset: ValidationSubset, csv_file: Path):
        """Save validation subset as CSV for human annotation."""
        import csv
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if subset.category == ValidationCategory.APPROACH_BIAS:
                # Special format for blind comparison
                fieldnames = [
                    'example_id', 'question', 'ground_truth', 'response_a', 'response_b', 'response_c',
                    'quality_a', 'quality_b', 'quality_c', 'preference_ranking', 'notes'
                ]
            else:
                # Standard format
                fieldnames = [
                    'example_id', 'question', 'ground_truth', 'model_response', 'judge_correctness',
                    'judge_confidence', 'approach', 'human_correctness', 'human_quality', 
                    'human_completeness', 'human_clarity', 'notes'
                ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for example in subset.examples:
                if subset.category == ValidationCategory.APPROACH_BIAS:
                    responses = json.loads(example.model_response)
                    row = {
                        'example_id': example.example_id,
                        'question': example.question,
                        'ground_truth': example.ground_truth,
                        'response_a': responses[0] if len(responses) > 0 else "",
                        'response_b': responses[1] if len(responses) > 1 else "",
                        'response_c': responses[2] if len(responses) > 2 else "",
                        'quality_a': '', 'quality_b': '', 'quality_c': '',
                        'preference_ranking': '', 'notes': ''
                    }
                else:
                    row = {
                        'example_id': example.example_id,
                        'question': example.question,
                        'ground_truth': example.ground_truth,
                        'model_response': example.model_response,
                        'judge_correctness': example.judge_evaluation.get("is_correct", ""),
                        'judge_confidence': example.judge_evaluation.get("confidence", ""),
                        'approach': example.approach_used,
                        'human_correctness': '',
                        'human_quality': '',
                        'human_completeness': '',
                        'human_clarity': '',
                        'notes': ''
                    }
                writer.writerow(row)
    
    def display_subset_summary(self, subsets: List[ValidationSubset]):
        """Display a summary of generated validation subsets."""
        
        summary_table = Table(title="🧑‍⚖️ Human Validation Subsets Generated", show_header=True, header_style="bold magenta")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Examples", style="yellow", justify="right")
        summary_table.add_column("Purpose", style="green")
        
        for subset in subsets:
            purpose_map = {
                ValidationCategory.JUDGE_CALIBRATION: "Validate LLM judge accuracy",
                ValidationCategory.ANSWER_QUALITY: "Assess answer quality beyond correctness", 
                ValidationCategory.APPROACH_BIAS: "Detect systematic approach preferences",
                ValidationCategory.EDGE_CASES: "Handle difficult evaluation scenarios"
            }
            
            summary_table.add_row(
                subset.category.value.replace("_", " ").title(),
                str(len(subset.examples)),
                purpose_map.get(subset.category, "Unknown")
            )
        
        console.print(summary_table)
        
        total_examples = sum(len(s.examples) for s in subsets)
        info_panel = Panel.fit(
            f"[bold blue]📊 Total Validation Examples: {total_examples}[/bold blue]\n"
            f"[dim]Estimated annotation time:[/dim] [yellow]~{total_examples * 2} minutes[/yellow]\n"
            f"[dim]Recommended:[/dim] [green]2-3 human annotators for reliability[/green]\n"
            f"[dim]Output format:[/dim] [cyan]CSV files for annotation, JSON for analysis[/cyan]",
            title="📋 Validation Summary",
            border_style="blue"
        )
        console.print(info_panel)


def main():
    """Main function for testing human validation generation."""
    # This would be called with actual experiment results
    console.print("[bold blue]🧑‍⚖️ Human Validation Framework[/bold blue]")
    console.print("[dim]This module generates validation subsets from experiment results[/dim]")
    console.print("\n[yellow]Usage:[/yellow]")
    console.print("1. Run experiment to generate results")
    console.print("2. Initialize HumanValidationGenerator with results path")
    console.print("3. Call generate_validation_subsets() to create validation sets")
    console.print("4. Save subsets as CSV files for human annotation")
    console.print("5. Analyze annotation results to validate LLM judge performance")


if __name__ == "__main__":
    main()