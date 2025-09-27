#!/usr/bin/env python3
"""
Power Analysis and Statistical Planning for Reranking Value Experiment

This module provides power analysis calculations and statistical planning
for the 5-way comparison study to ensure adequate sample sizes and
proper statistical testing procedures.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class EffectSize(Enum):
    """Standard effect size classifications."""
    TRIVIAL = 0.1
    SMALL = 0.2
    MEDIUM = 0.5
    LARGE = 0.8


@dataclass 
class PowerAnalysisConfig:
    """Configuration for power analysis."""
    alpha: float = 0.05  # Type I error rate
    power: float = 0.80  # Desired statistical power (1 - Type II error)
    effect_size: float = 0.3  # Cohen's h for proportions
    num_comparisons: int = 10  # Number of pairwise comparisons
    baseline_accuracy: float = 0.7  # Expected baseline accuracy
    
    
@dataclass
class PowerAnalysisResult:
    """Results from power analysis calculation."""
    required_sample_size: int
    actual_power: float
    effect_size: float
    corrected_alpha: float
    confidence_level: float
    minimum_detectable_effect: float
    
    
@dataclass
class StatisticalTestPlan:
    """Statistical testing plan for the experiment."""
    primary_tests: List[Dict[str, str]]
    secondary_tests: List[Dict[str, str]] 
    multiple_comparison_correction: str
    effect_size_measures: List[str]
    confidence_intervals: bool
    practical_significance_threshold: float


class PowerAnalyzer:
    """Performs power analysis for the reranking value experiment."""
    
    def __init__(self, config: PowerAnalysisConfig = None):
        """Initialize power analyzer with configuration."""
        self.config = config or PowerAnalysisConfig()
        
    def calculate_sample_size_proportions(self, 
                                        p1: float,
                                        p2: float, 
                                        alpha: float = None,
                                        power: float = None) -> int:
        """
        Calculate required sample size for comparing two proportions.
        
        Uses the formula for two-sample proportion test with equal sample sizes.
        """
        alpha = alpha or self.config.alpha
        power = power or self.config.power
        
        # Z-scores for alpha and power
        z_alpha = self._get_z_score(alpha / 2)
        z_beta = self._get_z_score(1 - power)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Cohen's h (effect size for proportions)
        h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
        
        # Sample size calculation
        n = 2 * ((z_alpha + z_beta) ** 2) * p_pooled * (1 - p_pooled) / (p1 - p2) ** 2
        
        return math.ceil(n)
    
    def calculate_sample_size_cohens_h(self,
                                     effect_size: float,
                                     alpha: float = None,
                                     power: float = None) -> int:
        """Calculate sample size using Cohen's h effect size."""
        alpha = alpha or self.config.alpha
        power = power or self.config.power
        
        z_alpha = self._get_z_score(alpha / 2)
        z_beta = self._get_z_score(1 - power)
        
        # Formula for Cohen's h
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return math.ceil(n)
    
    def factorial_power_analysis(self) -> PowerAnalysisResult:
        """
        Perform power analysis for the factorial design.
        
        Our design:
        - 5 approaches: Full Context, Basic RAG±Rerank, Enhanced RAG±Rerank  
        - Multiple models, context groups, question types
        - Primary comparison: Effect of reranking within each RAG type
        """
        
        # Corrected alpha for multiple comparisons (Bonferroni)
        corrected_alpha = self.config.alpha / self.config.num_comparisons
        
        # Primary comparison: Basic RAG vs Basic RAG + Rerank
        n_primary = self.calculate_sample_size_cohens_h(
            effect_size=self.config.effect_size,
            alpha=corrected_alpha,
            power=self.config.power
        )
        
        # Secondary comparisons need larger sample sizes due to smaller expected effects
        n_secondary = self.calculate_sample_size_cohens_h(
            effect_size=self.config.effect_size * 0.7,  # Smaller effect for secondary comparisons
            alpha=corrected_alpha,
            power=self.config.power
        )
        
        # Take the maximum required
        required_n = max(n_primary, n_secondary)
        
        # Calculate actual power with this sample size
        actual_power = self._calculate_power(
            n=required_n,
            effect_size=self.config.effect_size,
            alpha=corrected_alpha
        )
        
        # Minimum detectable effect with this sample size
        min_detectable_effect = self._minimum_detectable_effect(
            n=required_n,
            alpha=corrected_alpha,
            power=self.config.power
        )
        
        return PowerAnalysisResult(
            required_sample_size=required_n,
            actual_power=actual_power,
            effect_size=self.config.effect_size,
            corrected_alpha=corrected_alpha,
            confidence_level=1 - corrected_alpha,
            minimum_detectable_effect=min_detectable_effect
        )
    
    def create_statistical_test_plan(self) -> StatisticalTestPlan:
        """Create comprehensive statistical testing plan."""
        
        primary_tests = [
            {
                "comparison": "Basic RAG vs Basic RAG + Rerank",
                "test": "McNemar's test for paired proportions",
                "hypothesis": "H0: Reranking has no effect on Basic RAG accuracy",
                "alpha": self.config.alpha / 2  # Bonferroni for 2 primary comparisons
            },
            {
                "comparison": "Enhanced RAG vs Enhanced RAG + Rerank", 
                "test": "McNemar's test for paired proportions",
                "hypothesis": "H0: Reranking has no effect on Enhanced RAG accuracy",
                "alpha": self.config.alpha / 2
            }
        ]
        
        secondary_tests = [
            {
                "comparison": "Full Context vs Basic RAG",
                "test": "Chi-square test of independence",
                "hypothesis": "H0: No difference between Full Context and Basic RAG",
                "alpha": self.config.alpha / 8  # Bonferroni for 8 secondary comparisons
            },
            {
                "comparison": "Full Context vs Enhanced RAG",
                "test": "Chi-square test of independence", 
                "hypothesis": "H0: No difference between Full Context and Enhanced RAG",
                "alpha": self.config.alpha / 8
            },
            {
                "comparison": "Basic RAG vs Enhanced RAG",
                "test": "McNemar's test for paired proportions",
                "hypothesis": "H0: No difference between Basic and Enhanced RAG",
                "alpha": self.config.alpha / 8
            },
            {
                "comparison": "Question Type Interaction",
                "test": "Cochran-Mantel-Haenszel test",
                "hypothesis": "H0: No interaction between approach and question type", 
                "alpha": self.config.alpha / 8
            },
            {
                "comparison": "Context Group Interaction",
                "test": "Cochran-Mantel-Haenszel test",
                "hypothesis": "H0: No interaction between approach and context group",
                "alpha": self.config.alpha / 8
            },
            {
                "comparison": "Model Interaction",
                "test": "Mixed-effects logistic regression",
                "hypothesis": "H0: No interaction between approach and model",
                "alpha": self.config.alpha / 8
            }
        ]
        
        return StatisticalTestPlan(
            primary_tests=primary_tests,
            secondary_tests=secondary_tests,
            multiple_comparison_correction="Bonferroni",
            effect_size_measures=["Cohen's h", "Odds Ratio", "Risk Difference"],
            confidence_intervals=True,
            practical_significance_threshold=0.05  # 5% accuracy difference
        )
    
    def experimental_design_requirements(self, 
                                       models: List[str],
                                       context_groups: List[str], 
                                       question_types: List[str],
                                       iterations_per_question: int = 1) -> Dict[str, int]:
        """Calculate total experimental requirements."""
        
        power_result = self.factorial_power_analysis()
        n_per_condition = power_result.required_sample_size
        
        # 5 approaches
        n_approaches = 5
        
        # Total unique questions needed per stratum
        questions_per_stratum = math.ceil(n_per_condition / (len(models) * iterations_per_question))
        
        # Total questions across all strata
        total_strata = len(context_groups) * len(question_types)
        total_questions_needed = questions_per_stratum * total_strata
        
        # Total experimental trials
        total_trials = (
            total_questions_needed * 
            n_approaches * 
            len(models) * 
            iterations_per_question
        )
        
        return {
            "questions_per_stratum": questions_per_stratum,
            "total_questions_needed": total_questions_needed,
            "total_trials": total_trials,
            "trials_per_approach": total_trials // n_approaches,
            "sample_size_per_comparison": n_per_condition,
            "power": power_result.actual_power,
            "minimum_detectable_effect": power_result.minimum_detectable_effect
        }
    
    def cost_benefit_analysis(self, requirements: Dict[str, int]) -> Dict[str, float]:
        """Estimate costs and benefits of the experimental design."""
        
        # Cost estimates (tokens)
        cost_per_trial = {
            "query_processing": 50,      # Question + system prompt
            "context_tokens": 1500,      # Average context size
            "response_generation": 200,   # Model response
            "judge_evaluation": 300,     # LLM judge prompt + response
        }
        
        total_cost_per_trial = sum(cost_per_trial.values())
        total_tokens = requirements["total_trials"] * total_cost_per_trial
        
        # Cost in USD (approximate, using GPT-4 pricing)
        cost_per_1k_tokens = 0.03  # Input tokens
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        # Time estimates
        time_per_trial = 3.0  # seconds (based on our quick test)
        total_time_hours = (requirements["total_trials"] * time_per_trial) / 3600
        
        return {
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "total_time_hours": total_time_hours,
            "cost_per_comparison": estimated_cost / 10,  # 10 main comparisons
            "time_per_comparison_hours": total_time_hours / 10
        }
    
    def _get_z_score(self, p: float) -> float:
        """Get z-score for given probability."""
        # Approximation for common values
        z_scores = {
            0.025: 1.96,   # 95% CI
            0.005: 2.576,  # 99% CI
            0.0005: 3.291, # 99.9% CI
            0.1: 1.28,     # 80% power
            0.05: 1.645,   # 90% power
            0.2: 0.84,     # 80% power
        }
        
        if p in z_scores:
            return z_scores[p]
        
        # Use approximation for other values
        from scipy import stats
        return stats.norm.ppf(1 - p)
    
    def _calculate_power(self, n: int, effect_size: float, alpha: float) -> float:
        """Calculate statistical power for given parameters."""
        z_alpha = self._get_z_score(alpha / 2)
        z_beta = effect_size * math.sqrt(n / 2) - z_alpha
        
        # Convert to power (approximate)
        if z_beta <= 0:
            return alpha
        elif z_beta >= 3:
            return 0.999
        else:
            # Approximation
            power_approx = {
                0.84: 0.8,
                1.28: 0.9,
                1.645: 0.95,
                2.33: 0.99
            }
            
            # Linear interpolation (simplified)
            if z_beta <= 0.84:
                return 0.8 * (z_beta / 0.84)
            elif z_beta <= 1.28:
                return 0.8 + 0.1 * ((z_beta - 0.84) / (1.28 - 0.84))
            else:
                return min(0.99, 0.9 + 0.09 * ((z_beta - 1.28) / (2.33 - 1.28)))
    
    def _minimum_detectable_effect(self, n: int, alpha: float, power: float) -> float:
        """Calculate minimum detectable effect size."""
        z_alpha = self._get_z_score(alpha / 2)
        z_beta = self._get_z_score(1 - power)
        
        return (z_alpha + z_beta) / math.sqrt(n / 2)
    
    def display_power_analysis(self, 
                              models: List[str],
                              context_groups: List[str],
                              question_types: List[str]) -> None:
        """Display comprehensive power analysis results."""
        
        console.print("[bold blue]📊 Power Analysis for Reranking Value Experiment[/bold blue]")
        
        # Power analysis results
        power_result = self.factorial_power_analysis()
        
        power_table = Table(title="🎯 Power Analysis Results", show_header=True, header_style="bold magenta")
        power_table.add_column("Parameter", style="cyan")
        power_table.add_column("Value", style="yellow")
        power_table.add_column("Interpretation", style="green")
        
        power_table.add_row("Required Sample Size", f"{power_result.required_sample_size}", "Per comparison group")
        power_table.add_row("Statistical Power", f"{power_result.actual_power:.3f}", "Probability of detecting true effect")
        power_table.add_row("Effect Size", f"{power_result.effect_size:.2f}", "Cohen's h (medium effect)")
        power_table.add_row("Corrected α", f"{power_result.corrected_alpha:.4f}", "Bonferroni corrected")
        power_table.add_row("Confidence Level", f"{power_result.confidence_level:.1%}", "For confidence intervals")
        power_table.add_row("Min Detectable Effect", f"{power_result.minimum_detectable_effect:.3f}", "Smallest effect we can detect")
        
        console.print(power_table)
        
        # Experimental requirements
        requirements = self.experimental_design_requirements(models, context_groups, question_types)
        
        design_table = Table(title="🧪 Experimental Design Requirements", show_header=True, header_style="bold blue")
        design_table.add_column("Requirement", style="cyan") 
        design_table.add_column("Count", style="yellow", justify="right")
        design_table.add_column("Rationale", style="green")
        
        design_table.add_row("Questions per Stratum", str(requirements["questions_per_stratum"]), "Per context×question type")
        design_table.add_row("Total Questions Needed", str(requirements["total_questions_needed"]), "Across all strata")
        design_table.add_row("Total Trials", str(requirements["total_trials"]), "All approaches × models × iterations")
        design_table.add_row("Trials per Approach", str(requirements["trials_per_approach"]), "For balanced comparison")
        
        console.print(design_table)
        
        # Cost-benefit analysis
        costs = self.cost_benefit_analysis(requirements)
        
        cost_panel = Panel.fit(
            f"[bold blue]💰 Cost-Benefit Analysis[/bold blue]\n"
            f"[dim]Estimated Token Usage:[/dim] [yellow]{costs['total_tokens']:,}[/yellow] tokens\n"
            f"[dim]Estimated Cost:[/dim] [green]${costs['estimated_cost_usd']:.2f}[/green] USD\n"
            f"[dim]Estimated Runtime:[/dim] [cyan]{costs['total_time_hours']:.1f}[/cyan] hours\n"
            f"[dim]Cost per Comparison:[/dim] [magenta]${costs['cost_per_comparison']:.2f}[/magenta] USD\n"
            f"[dim]Time per Comparison:[/dim] [blue]{costs['time_per_comparison_hours']:.1f}[/blue] hours",
            title="📈 Resource Requirements",
            border_style="blue"
        )
        console.print(cost_panel)
        
        # Statistical test plan
        test_plan = self.create_statistical_test_plan()
        
        console.print("\n[bold blue]📋 Statistical Testing Plan[/bold blue]")
        
        primary_table = Table(title="🎯 Primary Comparisons", show_header=True, header_style="bold green")
        primary_table.add_column("Comparison", style="cyan")
        primary_table.add_column("Statistical Test", style="yellow")
        primary_table.add_column("Alpha Level", style="red", justify="right")
        
        for test in test_plan.primary_tests:
            primary_table.add_row(test["comparison"], test["test"], f"{test['alpha']:.4f}")
        
        console.print(primary_table)
        
        # Recommendations
        recommendations = self._generate_recommendations(requirements, costs, power_result)
        
        rec_panel = Panel.fit(
            "\n".join([f"• {rec}" for rec in recommendations]),
            title="💡 Recommendations",
            border_style="green"
        )
        console.print(rec_panel)
    
    def _generate_recommendations(self, 
                                requirements: Dict[str, int], 
                                costs: Dict[str, float],
                                power_result: PowerAnalysisResult) -> List[str]:
        """Generate recommendations based on power analysis."""
        
        recommendations = []
        
        # Sample size recommendations
        if requirements["total_questions_needed"] > 200:
            recommendations.append("Consider reducing context groups or question types to decrease sample size requirements")
        
        if requirements["total_trials"] > 1000:
            recommendations.append("Large experiment detected. Consider phased approach: pilot study first, then full experiment")
        
        # Power recommendations
        if power_result.actual_power < 0.8:
            recommendations.append("Statistical power is below 0.8. Consider increasing sample size or effect size")
        elif power_result.actual_power > 0.95:
            recommendations.append("Very high statistical power. Could potentially reduce sample size to optimize resources")
        
        # Cost recommendations
        if costs["estimated_cost_usd"] > 100:
            recommendations.append("High estimated cost. Consider using smaller models for initial validation")
        
        if costs["total_time_hours"] > 24:
            recommendations.append("Long runtime expected. Implement robust checkpointing and progress monitoring")
        
        # Effect size recommendations
        if power_result.minimum_detectable_effect < 0.1:
            recommendations.append("Experiment can detect very small effects. Good for exploratory research")
        elif power_result.minimum_detectable_effect > 0.5:
            recommendations.append("Only large effects detectable. Consider increasing sample size for more sensitivity")
        
        # Multiple comparisons
        recommendations.append("Use Bonferroni correction for multiple comparisons to control Type I error rate")
        
        # Practical significance
        recommendations.append("Define practical significance threshold (e.g., 5% accuracy improvement) before analysis")
        
        return recommendations


def main():
    """Main function for power analysis demonstration."""
    console.print("[bold blue]📊 Power Analysis for Reranking Value Experiment[/bold blue]")
    
    # Example experimental setup
    models = ["claude-sonnet-3.7", "gpt-4.1"]
    context_groups = ["short", "medium", "long"] 
    question_types = ["factual", "reasoning", "synthesis", "multi_hop"]
    
    # Initialize power analyzer
    config = PowerAnalysisConfig(
        alpha=0.05,
        power=0.80,
        effect_size=0.3,  # Medium effect size
        num_comparisons=10,  # Conservative estimate
        baseline_accuracy=0.7
    )
    
    analyzer = PowerAnalyzer(config)
    analyzer.display_power_analysis(models, context_groups, question_types)


if __name__ == "__main__":
    main()