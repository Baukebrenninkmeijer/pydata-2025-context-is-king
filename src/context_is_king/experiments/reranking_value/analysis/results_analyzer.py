"""
Results Analyzer for Reranking Value Experiment

Provides statistical analysis and comparison of experimental results
across different approaches and conditions.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
from scipy import stats
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class ComparisonResult:
    """Statistical comparison between two approaches."""

    approach_a: str
    approach_b: str
    accuracy_a: float
    accuracy_b: float
    accuracy_difference: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]


class ResultsAnalyzer:
    """Comprehensive analysis of reranking experiment results."""

    def __init__(self, results_file: Optional[Path] = None, results_data: Optional[Dict[str, Any]] = None):
        """
        Initialize analyzer with experiment results.

        Args:
            results_file: Path to results JSON file
            results_data: Results data dictionary (alternative to file)
        """
        if results_file:
            with open(results_file, "r") as f:
                self.data = json.load(f)
        elif results_data:
            self.data = results_data
        else:
            raise ValueError("Either results_file or results_data must be provided")

        self.experiment_id = self.data.get("experiment_id", "unknown")
        self.results = self.data.get("results", [])

        console.print(f"[green]ResultsAnalyzer initialized with {len(self.results)} results[/green]")

    def analyze_approach_performance(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of performance by approach.

        Returns:
            Dictionary with detailed performance statistics
        """
        console.print("[blue]Analyzing approach performance...[/blue]")

        # Group results by approach
        by_approach = defaultdict(list)
        for result in self.results:
            approach = result["approach"]
            by_approach[approach].append(result)

        approach_stats = {}

        for approach, results in by_approach.items():
            # Basic accuracy statistics
            correct_count = sum(1 for r in results if r["judge_evaluation"]["is_correct"])
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0

            # Confidence statistics
            confidences = [r["judge_evaluation"]["confidence"] for r in results]
            confidence_mean = np.mean(confidences) if confidences else 0
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0

            # Context size statistics
            context_sizes = [r["context_measurement"]["total_tokens"] for r in results]
            context_mean = np.mean(context_sizes) if context_sizes else 0
            context_std = np.std(context_sizes) if len(context_sizes) > 1 else 0

            # Performance by question type
            by_question_type = defaultdict(list)
            for result in results:
                qtype = result["question_type"]
                by_question_type[qtype].append(result)

            question_type_performance = {}
            for qtype, qtype_results in by_question_type.items():
                qtype_correct = sum(1 for r in qtype_results if r["judge_evaluation"]["is_correct"])
                qtype_total = len(qtype_results)
                question_type_performance[qtype] = {
                    "accuracy": qtype_correct / qtype_total if qtype_total > 0 else 0,
                    "count": qtype_total,
                }

            # Performance by context group
            by_context_group = defaultdict(list)
            for result in results:
                cgroup = result["context_group"]
                by_context_group[cgroup].append(result)

            context_group_performance = {}
            for cgroup, cgroup_results in by_context_group.items():
                cgroup_correct = sum(1 for r in cgroup_results if r["judge_evaluation"]["is_correct"])
                cgroup_total = len(cgroup_results)
                context_group_performance[cgroup] = {
                    "accuracy": cgroup_correct / cgroup_total if cgroup_total > 0 else 0,
                    "count": cgroup_total,
                }

            approach_stats[approach] = {
                "total_trials": total_count,
                "correct_trials": correct_count,
                "accuracy": accuracy,
                "accuracy_std_error": np.sqrt(accuracy * (1 - accuracy) / total_count) if total_count > 0 else 0,
                "confidence_stats": {
                    "mean": confidence_mean,
                    "std": confidence_std,
                    "min": min(confidences) if confidences else 0,
                    "max": max(confidences) if confidences else 0,
                },
                "context_stats": {
                    "mean_tokens": context_mean,
                    "std_tokens": context_std,
                    "min_tokens": min(context_sizes) if context_sizes else 0,
                    "max_tokens": max(context_sizes) if context_sizes else 0,
                },
                "question_type_performance": question_type_performance,
                "context_group_performance": context_group_performance,
            }

        return {
            "approaches": list(approach_stats.keys()),
            "approach_statistics": approach_stats,
            "best_overall": max(approach_stats.items(), key=lambda x: x[1]["accuracy"])[0] if approach_stats else None,
            "most_consistent": min(approach_stats.items(), key=lambda x: x[1]["accuracy_std_error"])[0]
            if approach_stats
            else None,
        }

    def perform_statistical_tests(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance tests between approaches.

        Args:
            alpha: Significance level for tests

        Returns:
            Dictionary with statistical test results
        """
        console.print("[blue]Performing statistical significance tests...[/blue]")

        # Group results by approach
        by_approach = defaultdict(list)
        for result in self.results:
            approach = result["approach"]
            is_correct = result["judge_evaluation"]["is_correct"]
            by_approach[approach].append(is_correct)

        approaches = list(by_approach.keys())
        comparisons = []

        # Pairwise comparisons
        for i, approach_a in enumerate(approaches):
            for j, approach_b in enumerate(approaches[i + 1 :], i + 1):
                results_a = by_approach[approach_a]
                results_b = by_approach[approach_b]

                # Chi-square test for proportions
                correct_a = sum(results_a)
                total_a = len(results_a)
                correct_b = sum(results_b)
                total_b = len(results_b)

                # Construct contingency table
                contingency_table = [[correct_a, total_a - correct_a], [correct_b, total_b - correct_b]]

                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                    # Calculate effect size (Cramer's V)
                    n = total_a + total_b
                    effect_size = np.sqrt(chi2 / (n * min(1, 1)))  # Simplified Cramer's V

                    # Calculate confidence interval for difference in proportions
                    p_a = correct_a / total_a if total_a > 0 else 0
                    p_b = correct_b / total_b if total_b > 0 else 0
                    diff = p_a - p_b

                    # Standard error for difference in proportions
                    se_diff = np.sqrt((p_a * (1 - p_a) / total_a) + (p_b * (1 - p_b) / total_b))
                    z_alpha = stats.norm.ppf(1 - alpha / 2)
                    ci_lower = diff - z_alpha * se_diff
                    ci_upper = diff + z_alpha * se_diff

                    comparison = ComparisonResult(
                        approach_a=approach_a,
                        approach_b=approach_b,
                        accuracy_a=p_a,
                        accuracy_b=p_b,
                        accuracy_difference=diff,
                        p_value=p_value,
                        significant=p_value < alpha,
                        effect_size=effect_size,
                        confidence_interval=(ci_lower, ci_upper),
                    )

                    comparisons.append(comparison)

                except Exception as e:
                    console.print(f"[yellow]Statistical test failed for {approach_a} vs {approach_b}: {e}[/yellow]")

        # Bonferroni correction for multiple comparisons
        bonferroni_alpha = alpha / len(comparisons) if comparisons else alpha
        significant_comparisons = [c for c in comparisons if c.p_value < bonferroni_alpha]

        return {
            "alpha": alpha,
            "bonferroni_corrected_alpha": bonferroni_alpha,
            "total_comparisons": len(comparisons),
            "significant_comparisons": len(significant_comparisons),
            "pairwise_comparisons": [
                {
                    "approach_a": c.approach_a,
                    "approach_b": c.approach_b,
                    "accuracy_a": c.accuracy_a,
                    "accuracy_b": c.accuracy_b,
                    "accuracy_difference": c.accuracy_difference,
                    "p_value": c.p_value,
                    "significant": c.significant,
                    "significant_bonferroni": c.p_value < bonferroni_alpha,
                    "effect_size": c.effect_size,
                    "confidence_interval": c.confidence_interval,
                }
                for c in comparisons
            ],
        }

    def analyze_context_efficiency(self) -> Dict[str, Any]:
        """
        Analyze context usage efficiency across approaches.

        Returns:
            Dictionary with context efficiency analysis
        """
        console.print("[blue]Analyzing context efficiency...[/blue]")

        by_approach = defaultdict(list)
        for result in self.results:
            approach = result["approach"]
            context_measurement = result["context_measurement"]
            is_correct = result["judge_evaluation"]["is_correct"]

            by_approach[approach].append(
                {
                    "total_tokens": context_measurement["total_tokens"],
                    "context_tokens": context_measurement["retrieved_context_tokens"],
                    "efficiency_metrics": context_measurement["efficiency_metrics"],
                    "is_correct": is_correct,
                }
            )

        efficiency_analysis = {}

        for approach, measurements in by_approach.items():
            # Token efficiency
            total_tokens = [m["total_tokens"] for m in measurements]
            context_tokens = [m["context_tokens"] for m in measurements]
            correct_answers = [m["is_correct"] for m in measurements]

            # Efficiency metrics
            utilization_ratios = []
            context_densities = []

            for m in measurements:
                metrics = m["efficiency_metrics"]
                utilization_ratios.append(metrics.get("context_utilization_ratio", 0))
                context_densities.append(metrics.get("context_density", 0))

            # Cost per correct answer
            correct_indices = [i for i, c in enumerate(correct_answers) if c]
            if correct_indices:
                tokens_per_correct = np.mean([total_tokens[i] for i in correct_indices])
            else:
                tokens_per_correct = float("inf")

            efficiency_analysis[approach] = {
                "avg_total_tokens": np.mean(total_tokens),
                "avg_context_tokens": np.mean(context_tokens),
                "tokens_per_correct_answer": tokens_per_correct,
                "avg_utilization_ratio": np.mean(utilization_ratios),
                "avg_context_density": np.mean(context_densities),
                "token_efficiency": np.mean(context_tokens) / np.mean(total_tokens) if total_tokens else 0,
            }

        # Find most efficient approach
        valid_approaches = {
            k: v for k, v in efficiency_analysis.items() if v["tokens_per_correct_answer"] != float("inf")
        }

        most_token_efficient = None
        best_utilization = None

        if valid_approaches:
            most_token_efficient = min(valid_approaches.items(), key=lambda x: x[1]["tokens_per_correct_answer"])[0]
            best_utilization = max(efficiency_analysis.items(), key=lambda x: x[1]["avg_utilization_ratio"])[0]

        return {
            "efficiency_by_approach": efficiency_analysis,
            "most_token_efficient": most_token_efficient,
            "best_context_utilization": best_utilization,
        }

    def analyze_question_complexity_impact(self) -> Dict[str, Any]:
        """
        Analyze how different approaches perform across question complexities.

        Returns:
            Dictionary with complexity analysis results
        """
        console.print("[blue]Analyzing question complexity impact...[/blue]")

        # Group by question type and approach
        by_complexity = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            question_type = result["question_type"]
            approach = result["approach"]
            is_correct = result["judge_evaluation"]["is_correct"]
            confidence = result["judge_evaluation"]["confidence"]

            by_complexity[question_type][approach].append({"correct": is_correct, "confidence": confidence})

        complexity_analysis = {}

        for question_type, approaches in by_complexity.items():
            approach_performance = {}

            for approach, results in approaches.items():
                correct_count = sum(1 for r in results if r["correct"])
                total_count = len(results)
                avg_confidence = np.mean([r["confidence"] for r in results]) if results else 0

                approach_performance[approach] = {
                    "accuracy": correct_count / total_count if total_count > 0 else 0,
                    "count": total_count,
                    "avg_confidence": avg_confidence,
                }

            # Find best approach for this complexity
            best_approach = (
                max(approach_performance.items(), key=lambda x: x[1]["accuracy"])[0] if approach_performance else None
            )

            complexity_analysis[question_type] = {
                "approach_performance": approach_performance,
                "best_approach": best_approach,
                "total_questions": sum(p["count"] for p in approach_performance.values()),
            }

        return complexity_analysis

    def generate_summary_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate a comprehensive summary report.

        Args:
            output_file: Optional file to save the report

        Returns:
            Report text
        """
        console.print("[blue]Generating summary report...[/blue]")

        # Run all analyses
        performance_analysis = self.analyze_approach_performance()
        statistical_tests = self.perform_statistical_tests()
        efficiency_analysis = self.analyze_context_efficiency()
        complexity_analysis = self.analyze_question_complexity_impact()

        # Build report
        report_lines = []
        report_lines.append(f"# Reranking Value Experiment Analysis Report")
        report_lines.append(f"")
        report_lines.append(f"**Experiment ID**: {self.experiment_id}")
        report_lines.append(f"**Total Trials**: {len(self.results)}")
        report_lines.append(f"")

        # Performance Summary
        report_lines.append("## Overall Performance")
        report_lines.append("")

        approach_stats = performance_analysis["approach_statistics"]
        for approach, stats in approach_stats.items():
            approach_name = approach.replace("_", " ").title()
            report_lines.append(f"### {approach_name}")
            report_lines.append(f"- **Accuracy**: {stats['accuracy']:.2%} ± {stats['accuracy_std_error']:.3f}")
            report_lines.append(f"- **Total Trials**: {stats['total_trials']}")
            report_lines.append(f"- **Average Confidence**: {stats['confidence_stats']['mean']:.3f}")
            report_lines.append(f"- **Average Context Size**: {stats['context_stats']['mean_tokens']:.0f} tokens")
            report_lines.append("")

        # Statistical Significance
        report_lines.append("## Statistical Significance Tests")
        report_lines.append("")

        significant_comparisons = [c for c in statistical_tests["pairwise_comparisons"] if c["significant_bonferroni"]]

        if significant_comparisons:
            report_lines.append("**Significant Differences (Bonferroni corrected):**")
            for comp in significant_comparisons:
                a_name = comp["approach_a"].replace("_", " ").title()
                b_name = comp["approach_b"].replace("_", " ").title()
                diff = comp["accuracy_difference"]
                p_val = comp["p_value"]
                report_lines.append(f"- {a_name} vs {b_name}: {diff:+.3f} (p={p_val:.4f})")
            report_lines.append("")
        else:
            report_lines.append("No statistically significant differences found between approaches.")
            report_lines.append("")

        # Efficiency Analysis
        report_lines.append("## Context Efficiency")
        report_lines.append("")

        efficiency_stats = efficiency_analysis["efficiency_by_approach"]
        most_efficient = efficiency_analysis["most_token_efficient"]

        report_lines.append(
            f"**Most Token Efficient**: {most_efficient.replace('_', ' ').title()}" if most_efficient else "Unknown"
        )
        report_lines.append("")

        for approach, stats in efficiency_stats.items():
            approach_name = approach.replace("_", " ").title()
            report_lines.append(f"### {approach_name}")
            report_lines.append(f"- **Tokens per Correct Answer**: {stats['tokens_per_correct_answer']:.0f}")
            report_lines.append(f"- **Average Context Utilization**: {stats['avg_utilization_ratio']:.2%}")
            report_lines.append(f"- **Token Efficiency**: {stats['token_efficiency']:.2%}")
            report_lines.append("")

        # Question Complexity
        report_lines.append("## Performance by Question Type")
        report_lines.append("")

        for question_type, analysis in complexity_analysis.items():
            type_name = question_type.replace("_", " ").title()
            best_approach = analysis["best_approach"]
            best_name = best_approach.replace("_", " ").title() if best_approach else "Unknown"

            report_lines.append(f"### {type_name}")
            report_lines.append(f"- **Best Approach**: {best_name}")
            report_lines.append("")

            for approach, perf in analysis["approach_performance"].items():
                approach_name = approach.replace("_", " ").title()
                report_lines.append(f"  - {approach_name}: {perf['accuracy']:.2%} ({perf['count']} trials)")
            report_lines.append("")

        # Key Findings
        report_lines.append("## Key Findings")
        report_lines.append("")

        best_overall = performance_analysis.get("best_overall", "Unknown")
        best_name = best_overall.replace("_", " ").title() if best_overall != "Unknown" else "Unknown"

        report_lines.append(f"1. **Best Overall Approach**: {best_name}")

        if most_efficient:
            efficient_name = most_efficient.replace("_", " ").title()
            report_lines.append(f"2. **Most Token Efficient**: {efficient_name}")

        if significant_comparisons:
            report_lines.append(f"3. **Significant Performance Differences**: {len(significant_comparisons)} found")
        else:
            report_lines.append("3. **No Significant Differences**: Between approaches detected")

        report_text = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            console.print(f"[green]Report saved to: {output_file}[/green]")

        return report_text

    def display_results_table(self):
        """Display results in a formatted table."""
        performance_analysis = self.analyze_approach_performance()

        table = Table(title="Approach Performance Comparison")
        table.add_column("Approach", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Trials", justify="right")
        table.add_column("Avg Confidence", style="yellow")
        table.add_column("Avg Context Tokens", justify="right", style="blue")

        approach_stats = performance_analysis["approach_statistics"]
        for approach, stats in approach_stats.items():
            approach_name = approach.replace("_", " ").title()
            table.add_row(
                approach_name,
                f"{stats['accuracy']:.2%}",
                str(stats["total_trials"]),
                f"{stats['confidence_stats']['mean']:.3f}",
                f"{stats['context_stats']['mean_tokens']:.0f}",
            )

        console.print(table)
