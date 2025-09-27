"""
Cost Analyzer for Reranking Value Experiment

Analyzes token costs, computational overhead, and cost-benefit trade-offs
across different approaches in the reranking experiment.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class CostBreakdown:
    """Cost breakdown for an experimental approach."""
    approach: str
    total_tokens: int
    cost_per_million_tokens: float
    total_cost_usd: float
    cost_per_trial: float
    cost_per_correct_answer: float
    computational_overhead: Dict[str, float]


@dataclass
class ROIAnalysis:
    """Return on Investment analysis for reranking."""
    base_approach: str
    enhanced_approach: str
    accuracy_improvement: float
    cost_increase_usd: float
    roi_ratio: float  # accuracy improvement per dollar
    break_even_accuracy: float


class CostAnalyzer:
    """Comprehensive cost analysis for reranking experiment."""
    
    # Estimated token costs (USD per million tokens)
    TOKEN_COSTS = {
        "gpt-4.1-2025-04-14": 0.03,
        "claude-sonnet-3.7": 0.015,
        "gpt-3.5-turbo": 0.002,
        "default": 0.02
    }
    
    # Estimated computational costs for different operations
    COMPUTATIONAL_COSTS = {
        "embedding_generation": 0.0001,  # Per document
        "semantic_search": 0.0002,       # Per query
        "query_rewriting": 0.001,        # Per query rewrite
        "reranking": 0.002,              # Per reranking operation
        "context_assembly": 0.0001       # Per assembly operation
    }
    
    def __init__(self, results_file: Optional[Path] = None, results_data: Optional[Dict[str, Any]] = None):
        """
        Initialize cost analyzer with experiment results.
        
        Args:
            results_file: Path to results JSON file
            results_data: Results data dictionary (alternative to file)
        """
        if results_file:
            with open(results_file, 'r') as f:
                self.data = json.load(f)
        elif results_data:
            self.data = results_data
        else:
            raise ValueError("Either results_file or results_data must be provided")
        
        self.experiment_id = self.data.get("experiment_id", "unknown")
        self.results = self.data.get("results", [])
        
        console.print(f"[green]CostAnalyzer initialized with {len(self.results)} results[/green]")
    
    def analyze_token_costs(self, model_costs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze token usage and associated costs across approaches.
        
        Args:
            model_costs: Custom token costs per model (USD per million tokens)
            
        Returns:
            Dictionary with detailed cost analysis
        """
        console.print("[blue]Analyzing token costs...[/blue]")
        
        costs = model_costs or self.TOKEN_COSTS
        
        # Group results by approach
        by_approach = defaultdict(list)
        for result in self.results:
            approach = result["approach"]
            model = result["model"]
            by_approach[approach].append({
                "model": model,
                "total_tokens": result["context_measurement"]["total_tokens"],
                "query_tokens": result["context_measurement"]["query_tokens"],
                "context_tokens": result["context_measurement"]["retrieved_context_tokens"],
                "is_correct": result["judge_evaluation"]["is_correct"],
                "approach_metadata": result.get("approach_metadata", {})
            })
        
        cost_analysis = {}
        
        for approach, results in by_approach.items():
            # Calculate token statistics
            total_tokens_sum = sum(r["total_tokens"] for r in results)
            total_trials = len(results)
            correct_trials = sum(1 for r in results if r["is_correct"])
            
            # Calculate costs by model
            model_costs_breakdown = defaultdict(lambda: {"tokens": 0, "trials": 0})
            for result in results:
                model = result["model"]
                tokens = result["total_tokens"]
                model_costs_breakdown[model]["tokens"] += tokens
                model_costs_breakdown[model]["trials"] += 1
            
            # Calculate total cost
            total_cost_usd = 0
            for model, model_data in model_costs_breakdown.items():
                cost_per_token = costs.get(model, costs["default"]) / 1_000_000
                total_cost_usd += model_data["tokens"] * cost_per_token
            
            # Calculate efficiency metrics
            cost_per_trial = total_cost_usd / total_trials if total_trials > 0 else 0
            cost_per_correct = total_cost_usd / correct_trials if correct_trials > 0 else float('inf')
            avg_tokens_per_trial = total_tokens_sum / total_trials if total_trials > 0 else 0
            
            # Token composition analysis
            avg_query_tokens = np.mean([r["query_tokens"] for r in results])
            avg_context_tokens = np.mean([r["context_tokens"] for r in results])
            
            cost_breakdown = CostBreakdown(
                approach=approach,
                total_tokens=total_tokens_sum,
                cost_per_million_tokens=costs.get("default", 0.02),
                total_cost_usd=total_cost_usd,
                cost_per_trial=cost_per_trial,
                cost_per_correct_answer=cost_per_correct,
                computational_overhead=self._calculate_computational_costs(approach, total_trials)
            )
            
            cost_analysis[approach] = {
                "cost_breakdown": cost_breakdown,
                "token_statistics": {
                    "total_tokens": total_tokens_sum,
                    "avg_tokens_per_trial": avg_tokens_per_trial,
                    "avg_query_tokens": avg_query_tokens,
                    "avg_context_tokens": avg_context_tokens,
                    "token_composition": {
                        "query_ratio": avg_query_tokens / avg_tokens_per_trial if avg_tokens_per_trial > 0 else 0,
                        "context_ratio": avg_context_tokens / avg_tokens_per_trial if avg_tokens_per_trial > 0 else 0
                    }
                },
                "cost_statistics": {
                    "total_cost_usd": total_cost_usd,
                    "cost_per_trial": cost_per_trial,
                    "cost_per_correct_answer": cost_per_correct,
                    "cost_per_million_tokens": total_cost_usd / (total_tokens_sum / 1_000_000) if total_tokens_sum > 0 else 0
                },
                "model_breakdown": dict(model_costs_breakdown)
            }
        
        # Find most cost-efficient approach
        valid_approaches = {k: v for k, v in cost_analysis.items() 
                           if v["cost_statistics"]["cost_per_correct_answer"] != float('inf')}
        
        most_cost_efficient = None
        if valid_approaches:
            most_cost_efficient = min(valid_approaches.items(),
                                    key=lambda x: x[1]["cost_statistics"]["cost_per_correct_answer"])[0]
        
        return {
            "cost_analysis_by_approach": cost_analysis,
            "most_cost_efficient_approach": most_cost_efficient,
            "total_experiment_cost": sum(a["cost_statistics"]["total_cost_usd"] for a in cost_analysis.values())
        }
    
    def _calculate_computational_costs(self, approach: str, trial_count: int) -> Dict[str, float]:
        """
        Estimate computational overhead costs for an approach.
        
        Args:
            approach: Approach name
            trial_count: Number of trials for this approach
            
        Returns:
            Dictionary with computational cost breakdown
        """
        base_costs = {
            "context_assembly": trial_count * self.COMPUTATIONAL_COSTS["context_assembly"]
        }
        
        if approach == "full_context":
            # Full context has minimal computational overhead
            base_costs.update({
                "retrieval": 0,
                "query_rewriting": 0,
                "reranking": 0
            })
        
        elif approach in ["enhanced_rag_no_rerank", "enhanced_rag_rerank"]:
            # Enhanced RAG approaches have retrieval and query rewriting costs
            base_costs.update({
                "retrieval": trial_count * self.COMPUTATIONAL_COSTS["semantic_search"],
                "query_rewriting": trial_count * self.COMPUTATIONAL_COSTS["query_rewriting"]
            })
            
            if approach == "enhanced_rag_rerank":
                base_costs["reranking"] = trial_count * self.COMPUTATIONAL_COSTS["reranking"]
            else:
                base_costs["reranking"] = 0
        
        return base_costs
    
    def analyze_roi_of_reranking(self) -> ROIAnalysis:
        """
        Analyze the Return on Investment of adding reranking to RAG.
        
        Returns:
            ROIAnalysis comparing enhanced RAG with and without reranking
        """
        console.print("[blue]Analyzing ROI of reranking...[/blue]")
        
        # Get cost analysis
        cost_analysis = self.analyze_token_costs()
        
        # Find the relevant approaches
        no_rerank_key = "enhanced_rag_no_rerank"
        with_rerank_key = "enhanced_rag_rerank"
        
        cost_data = cost_analysis["cost_analysis_by_approach"]
        
        if no_rerank_key not in cost_data or with_rerank_key not in cost_data:
            console.print("[yellow]Cannot perform ROI analysis: missing approach data[/yellow]")
            return ROIAnalysis(
                base_approach=no_rerank_key,
                enhanced_approach=with_rerank_key,
                accuracy_improvement=0,
                cost_increase_usd=0,
                roi_ratio=0,
                break_even_accuracy=0
            )
        
        # Get accuracy for each approach
        no_rerank_results = [r for r in self.results if r["approach"] == no_rerank_key]
        with_rerank_results = [r for r in self.results if r["approach"] == with_rerank_key]
        
        no_rerank_accuracy = np.mean([r["judge_evaluation"]["is_correct"] for r in no_rerank_results]) if no_rerank_results else 0
        with_rerank_accuracy = np.mean([r["judge_evaluation"]["is_correct"] for r in with_rerank_results]) if with_rerank_results else 0
        
        accuracy_improvement = with_rerank_accuracy - no_rerank_accuracy
        
        # Get cost increase
        no_rerank_cost = cost_data[no_rerank_key]["cost_statistics"]["total_cost_usd"]
        with_rerank_cost = cost_data[with_rerank_key]["cost_statistics"]["total_cost_usd"]
        cost_increase = with_rerank_cost - no_rerank_cost
        
        # Calculate ROI ratio (accuracy improvement per dollar)
        roi_ratio = accuracy_improvement / cost_increase if cost_increase > 0 else float('inf')
        
        # Calculate break-even accuracy (accuracy needed to justify cost)
        # Assuming $1 per 1% accuracy improvement as baseline value
        break_even_accuracy = no_rerank_accuracy + (cost_increase * 0.01)
        
        return ROIAnalysis(
            base_approach=no_rerank_key,
            enhanced_approach=with_rerank_key,
            accuracy_improvement=accuracy_improvement,
            cost_increase_usd=cost_increase,
            roi_ratio=roi_ratio,
            break_even_accuracy=break_even_accuracy
        )
    
    def analyze_cost_scaling(self) -> Dict[str, Any]:
        """
        Analyze how costs scale with different experimental parameters.
        
        Returns:
            Dictionary with cost scaling analysis
        """
        console.print("[blue]Analyzing cost scaling patterns...[/blue]")
        
        # Group by context group and question type
        by_context_group = defaultdict(lambda: defaultdict(list))
        by_question_type = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            approach = result["approach"]
            context_group = result["context_group"]
            question_type = result["question_type"]
            tokens = result["context_measurement"]["total_tokens"]
            is_correct = result["judge_evaluation"]["is_correct"]
            
            by_context_group[context_group][approach].append({"tokens": tokens, "correct": is_correct})
            by_question_type[question_type][approach].append({"tokens": tokens, "correct": is_correct})
        
        # Analyze scaling by context group
        context_scaling = {}
        for context_group, approaches in by_context_group.items():
            approach_stats = {}
            for approach, results in approaches.items():
                avg_tokens = np.mean([r["tokens"] for r in results])
                correct_count = sum(1 for r in results if r["correct"])
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                approach_stats[approach] = {
                    "avg_tokens": avg_tokens,
                    "accuracy": accuracy,
                    "trials": total_count
                }
            
            context_scaling[context_group] = approach_stats
        
        # Analyze scaling by question type
        question_scaling = {}
        for question_type, approaches in by_question_type.items():
            approach_stats = {}
            for approach, results in approaches.items():
                avg_tokens = np.mean([r["tokens"] for r in results])
                correct_count = sum(1 for r in results if r["correct"])
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                approach_stats[approach] = {
                    "avg_tokens": avg_tokens,
                    "accuracy": accuracy,
                    "trials": total_count
                }
            
            question_scaling[question_type] = approach_stats
        
        return {
            "scaling_by_context_group": context_scaling,
            "scaling_by_question_type": question_scaling
        }
    
    def generate_cost_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate comprehensive cost analysis report.
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Cost report text
        """
        console.print("[blue]Generating cost analysis report...[/blue]")
        
        # Run analyses
        cost_analysis = self.analyze_token_costs()
        roi_analysis = self.analyze_roi_of_reranking()
        scaling_analysis = self.analyze_cost_scaling()
        
        # Build report
        report_lines = []
        report_lines.append("# Cost Analysis Report")
        report_lines.append("")
        report_lines.append(f"**Experiment ID**: {self.experiment_id}")
        report_lines.append(f"**Total Experiment Cost**: ${cost_analysis['total_experiment_cost']:.4f}")
        report_lines.append("")
        
        # Cost by approach
        report_lines.append("## Cost Breakdown by Approach")
        report_lines.append("")
        
        cost_data = cost_analysis["cost_analysis_by_approach"]
        for approach, data in cost_data.items():
            approach_name = approach.replace("_", " ").title()
            stats = data["cost_statistics"]
            
            report_lines.append(f"### {approach_name}")
            report_lines.append(f"- **Total Cost**: ${stats['total_cost_usd']:.4f}")
            report_lines.append(f"- **Cost per Trial**: ${stats['cost_per_trial']:.4f}")
            report_lines.append(f"- **Cost per Correct Answer**: ${stats['cost_per_correct_answer']:.4f}")
            report_lines.append(f"- **Average Tokens per Trial**: {data['token_statistics']['avg_tokens_per_trial']:.0f}")
            report_lines.append("")
        
        # ROI Analysis
        report_lines.append("## Reranking ROI Analysis")
        report_lines.append("")
        
        if roi_analysis.cost_increase_usd > 0:
            report_lines.append(f"- **Accuracy Improvement**: {roi_analysis.accuracy_improvement:+.2%}")
            report_lines.append(f"- **Additional Cost**: ${roi_analysis.cost_increase_usd:.4f}")
            report_lines.append(f"- **ROI Ratio**: {roi_analysis.roi_ratio:.2f} accuracy points per dollar")
            report_lines.append(f"- **Break-even Accuracy**: {roi_analysis.break_even_accuracy:.2%}")
            report_lines.append("")
            
            if roi_analysis.accuracy_improvement > 0:
                report_lines.append("**Conclusion**: Reranking provides positive ROI.")
            else:
                report_lines.append("**Conclusion**: Reranking does not provide positive ROI.")
        else:
            report_lines.append("ROI analysis unavailable - insufficient data.")
        
        report_lines.append("")
        
        # Most cost-efficient approach
        most_efficient = cost_analysis.get("most_cost_efficient_approach")
        if most_efficient:
            efficient_name = most_efficient.replace("_", " ").title()
            report_lines.append(f"## Most Cost-Efficient Approach")
            report_lines.append(f"**{efficient_name}** offers the best cost per correct answer.")
            report_lines.append("")
        
        # Cost scaling insights
        report_lines.append("## Cost Scaling Insights")
        report_lines.append("")
        
        context_scaling = scaling_analysis["scaling_by_context_group"]
        report_lines.append("### By Context Size:")
        for context_group in ["short", "medium", "long"]:
            if context_group in context_scaling:
                report_lines.append(f"- **{context_group.title()}**: Context requires higher token usage as expected")
        
        report_lines.append("")
        
        question_scaling = scaling_analysis["scaling_by_question_type"]
        report_lines.append("### By Question Complexity:")
        for question_type in ["factual", "reasoning", "synthesis", "multi_hop"]:
            if question_type in question_scaling:
                type_name = question_type.replace("_", " ").title()
                report_lines.append(f"- **{type_name}**: Question complexity affects processing requirements")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            console.print(f"[green]Cost report saved to: {output_file}[/green]")
        
        return report_text
    
    def display_cost_table(self):
        """Display cost analysis in a formatted table."""
        cost_analysis = self.analyze_token_costs()
        
        table = Table(title="Cost Analysis by Approach")
        table.add_column("Approach", style="cyan")
        table.add_column("Total Cost", style="red")
        table.add_column("Cost/Trial", style="yellow")
        table.add_column("Cost/Correct", style="green")
        table.add_column("Avg Tokens", justify="right", style="blue")
        
        cost_data = cost_analysis["cost_analysis_by_approach"]
        for approach, data in cost_data.items():
            approach_name = approach.replace("_", " ").title()
            stats = data["cost_statistics"]
            token_stats = data["token_statistics"]
            
            cost_per_correct = stats["cost_per_correct_answer"]
            cost_per_correct_str = f"${cost_per_correct:.4f}" if cost_per_correct != float('inf') else "N/A"
            
            table.add_row(
                approach_name,
                f"${stats['total_cost_usd']:.4f}",
                f"${stats['cost_per_trial']:.4f}",
                cost_per_correct_str,
                f"{token_stats['avg_tokens_per_trial']:.0f}"
            )
        
        console.print(table)
        
        # Display ROI analysis
        roi = self.analyze_roi_of_reranking()
        if roi.cost_increase_usd > 0:
            roi_table = Table(title="Reranking ROI Analysis")
            roi_table.add_column("Metric", style="cyan")
            roi_table.add_column("Value", style="green")
            
            roi_table.add_row("Accuracy Improvement", f"{roi.accuracy_improvement:+.2%}")
            roi_table.add_row("Additional Cost", f"${roi.cost_increase_usd:.4f}")
            roi_table.add_row("ROI Ratio", f"{roi.roi_ratio:.2f}")
            roi_table.add_row("Break-even Accuracy", f"{roi.break_even_accuracy:.2%}")
            
            console.print(roi_table)