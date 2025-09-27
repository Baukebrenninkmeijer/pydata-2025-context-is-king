#!/usr/bin/env python3
"""
Cost Optimization for Reranking Value Experiment

This module calculates optimized experimental designs that fit within budget constraints
while maintaining statistical validity. Focuses on cost reduction strategies while
preserving the ability to answer the core research question.
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class BudgetConfig:
    """Budget configuration for experiment optimization."""
    max_budget_usd: float = 50.0  # €50 ≈ $50
    cost_per_1k_input_tokens: float = 0.005  # GPT-4 Turbo pricing via ORQ
    cost_per_1k_output_tokens: float = 0.015  # GPT-4 Turbo pricing via ORQ
    
    
@dataclass
class CostBreakdown:
    """Detailed cost breakdown for experiment."""
    query_tokens: int
    context_tokens: int
    response_tokens: int
    judge_prompt_tokens: int
    judge_response_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


@dataclass 
class OptimizedDesign:
    """Optimized experimental design within budget."""
    questions_per_stratum: int
    total_questions: int
    total_trials: int
    models: List[str]
    context_groups: List[str] 
    question_types: List[str]
    approaches: List[str]
    statistical_power: float
    cost_breakdown: CostBreakdown
    cost_reduction_strategies: List[str]


class CostOptimizer:
    """Optimizes experimental design for cost constraints."""
    
    def __init__(self, budget_config: BudgetConfig = None):
        """Initialize cost optimizer."""
        self.budget = budget_config or BudgetConfig()
        
    def estimate_trial_cost(self, 
                          context_size: str = "medium",
                          approach: str = "enhanced_rag_rerank",
                          model: str = "gpt-4-turbo") -> CostBreakdown:
        """Estimate cost for a single trial."""
        
        # Token estimates based on approach and context size
        base_tokens = {
            "short": {"context": 800, "query": 25, "response": 150},
            "medium": {"context": 1500, "query": 30, "response": 200}, 
            "long": {"context": 2500, "query": 35, "response": 250}
        }
        
        tokens = base_tokens[context_size]
        
        # Approach-specific overhead
        if "full_context" in approach:
            # Full context uses much more context tokens
            tokens["context"] *= 3  # Full document vs chunks
        elif "enhanced" in approach:
            # Enhanced RAG has query rewriting overhead
            tokens["query"] *= 1.5  # Query rewriting
            tokens["context"] *= 1.2  # Dual retrieval
        
        # Judge evaluation tokens (consistent across approaches)
        judge_prompt = 400  # Judge prompt template + question + answer + context
        judge_response = 150  # Structured JSON response
        
        # Calculate total tokens
        input_tokens = tokens["context"] + tokens["query"] + judge_prompt
        output_tokens = tokens["response"] + judge_response
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * self.budget.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.budget.cost_per_1k_output_tokens
        total_cost = input_cost + output_cost
        
        return CostBreakdown(
            query_tokens=tokens["query"],
            context_tokens=tokens["context"],
            response_tokens=tokens["response"],
            judge_prompt_tokens=judge_prompt,
            judge_response_tokens=judge_response,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost_usd=total_cost
        )
    
    def calculate_experiment_cost(self,
                                questions_per_stratum: int,
                                models: List[str],
                                context_groups: List[str],
                                question_types: List[str],
                                approaches: List[str]) -> float:
        """Calculate total experiment cost."""
        
        total_cost = 0.0
        
        for context_group in context_groups:
            for approach in approaches:
                for model in models:
                    # Calculate trials for this combination
                    trials_per_combination = questions_per_stratum * len(question_types)
                    
                    # Estimate cost per trial
                    trial_cost = self.estimate_trial_cost(context_group, approach, model)
                    
                    # Add to total cost
                    total_cost += trials_per_combination * trial_cost.total_cost_usd
        
        return total_cost
    
    def optimize_for_budget(self, target_power: float = 0.8) -> OptimizedDesign:
        """Create optimized experimental design within budget."""
        
        console.print(f"[blue]🎯 Optimizing experiment for ${self.budget.max_budget_usd:.0f} budget...[/blue]")
        
        # Cost reduction strategies to try (in order of preference)
        strategies = [
            self._strategy_reduce_models,
            self._strategy_reduce_context_groups,
            self._strategy_reduce_question_types,
            self._strategy_reduce_sample_size,
            self._strategy_use_cheaper_models,
            self._strategy_reduce_approaches
        ]
        
        # Start with base configuration
        base_config = {
            "models": ["claude-sonnet-3.7"],  # Start with one model
            "context_groups": ["short", "medium", "long"],
            "question_types": ["factual", "reasoning", "synthesis", "multi_hop"], 
            "approaches": ["full_context", "basic_rag_no_rerank", "basic_rag_rerank", 
                         "enhanced_rag_no_rerank", "enhanced_rag_rerank"],
            "questions_per_stratum": 50  # Start conservative
        }
        
        applied_strategies = []
        
        # Apply strategies until we fit in budget
        for strategy in strategies:
            cost = self.calculate_experiment_cost(
                base_config["questions_per_stratum"],
                base_config["models"],
                base_config["context_groups"],
                base_config["question_types"],
                base_config["approaches"]
            )
            
            if cost <= self.budget.max_budget_usd:
                break
                
            # Apply strategy
            base_config, strategy_name = strategy(base_config)
            applied_strategies.append(strategy_name)
            
            console.print(f"[yellow]Applied strategy: {strategy_name}[/yellow]")
        
        # Calculate final cost and power
        final_cost = self.calculate_experiment_cost(
            base_config["questions_per_stratum"],
            base_config["models"], 
            base_config["context_groups"],
            base_config["question_types"],
            base_config["approaches"]
        )
        
        # Estimate statistical power (simplified)
        power = self._estimate_power(
            base_config["questions_per_stratum"],
            len(base_config["models"]),
            len(base_config["context_groups"]) * len(base_config["question_types"])
        )
        
        # Calculate totals
        total_strata = len(base_config["context_groups"]) * len(base_config["question_types"])
        total_questions = base_config["questions_per_stratum"] * total_strata
        total_trials = total_questions * len(base_config["approaches"]) * len(base_config["models"])
        
        # Create detailed cost breakdown for average trial
        avg_cost_breakdown = self.estimate_trial_cost("medium", "enhanced_rag_rerank")
        
        return OptimizedDesign(
            questions_per_stratum=base_config["questions_per_stratum"],
            total_questions=total_questions,
            total_trials=total_trials,
            models=base_config["models"],
            context_groups=base_config["context_groups"],
            question_types=base_config["question_types"], 
            approaches=base_config["approaches"],
            statistical_power=power,
            cost_breakdown=avg_cost_breakdown,
            cost_reduction_strategies=applied_strategies
        )
    
    def _strategy_reduce_models(self, config: Dict) -> Tuple[Dict, str]:
        """Reduce number of models to test."""
        if len(config["models"]) > 1:
            config["models"] = config["models"][:1]  # Keep only first model
            return config, "Reduced to single model (claude-sonnet-3.7)"
        return config, "No model reduction possible"
    
    def _strategy_reduce_context_groups(self, config: Dict) -> Tuple[Dict, str]:
        """Reduce context groups to test."""
        if len(config["context_groups"]) > 2:
            # Keep short and medium (most important for cost/benefit)
            config["context_groups"] = ["short", "medium"]
            return config, "Reduced context groups to short + medium"
        elif len(config["context_groups"]) > 1:
            config["context_groups"] = ["short"]  # Cheapest option
            return config, "Reduced context groups to short only"
        return config, "No context group reduction possible"
    
    def _strategy_reduce_question_types(self, config: Dict) -> Tuple[Dict, str]:
        """Reduce question types to test."""
        if len(config["question_types"]) > 2:
            # Keep most important types
            config["question_types"] = ["factual", "reasoning"]
            return config, "Reduced question types to factual + reasoning"
        elif len(config["question_types"]) > 1:
            config["question_types"] = ["factual"]  # Simplest type
            return config, "Reduced question types to factual only"
        return config, "No question type reduction possible"
    
    def _strategy_reduce_sample_size(self, config: Dict) -> Tuple[Dict, str]:
        """Reduce sample size per stratum."""
        if config["questions_per_stratum"] > 20:
            config["questions_per_stratum"] = max(20, config["questions_per_stratum"] // 2)
            return config, f"Reduced sample size to {config['questions_per_stratum']} per stratum"
        elif config["questions_per_stratum"] > 10:
            config["questions_per_stratum"] = 10
            return config, "Reduced sample size to minimum (10 per stratum)"
        return config, "Cannot reduce sample size further"
    
    def _strategy_use_cheaper_models(self, config: Dict) -> Tuple[Dict, str]:
        """Switch to cheaper models."""
        expensive_models = ["gpt-4", "gpt-4-turbo", "claude-sonnet-4"]
        cheaper_alternatives = {
            "gpt-4": "gpt-4o-mini", 
            "gpt-4-turbo": "gpt-4o-mini",
            "claude-sonnet-4": "claude-sonnet-3.5",
            "claude-sonnet-3.7": "claude-haiku-3.5"  # Much cheaper
        }
        
        new_models = []
        changed = False
        for model in config["models"]:
            if model in cheaper_alternatives:
                new_models.append(cheaper_alternatives[model])
                changed = True
            else:
                new_models.append(model)
        
        if changed:
            config["models"] = new_models
            return config, f"Switched to cheaper models: {new_models}"
        
        return config, "No cheaper model alternatives available"
    
    def _strategy_reduce_approaches(self, config: Dict) -> Tuple[Dict, str]:
        """Reduce approaches to core comparison."""
        if len(config["approaches"]) > 3:
            # Keep only essential approaches for reranking comparison
            config["approaches"] = ["basic_rag_no_rerank", "basic_rag_rerank", "enhanced_rag_rerank"]
            return config, "Reduced to core reranking comparison (3 approaches)"
        elif len(config["approaches"]) > 2:
            config["approaches"] = ["basic_rag_no_rerank", "basic_rag_rerank"] 
            return config, "Reduced to basic reranking comparison only"
        return config, "Cannot reduce approaches further"
    
    def _estimate_power(self, n_per_stratum: int, n_models: int, n_strata: int) -> float:
        """Estimate statistical power (simplified calculation)."""
        effective_n = n_per_stratum * n_models * n_strata
        
        # Simplified power calculation based on effective sample size
        if effective_n >= 200:
            return 0.9
        elif effective_n >= 100:
            return 0.8
        elif effective_n >= 50:
            return 0.7
        elif effective_n >= 25:
            return 0.6
        else:
            return 0.5
    
    def create_budget_alternatives(self) -> List[OptimizedDesign]:
        """Create multiple budget-optimized alternatives."""
        
        alternatives = []
        
        # Alternative 1: Minimal but complete (all 5 approaches)
        budget_configs = [
            BudgetConfig(max_budget_usd=20.0),  # Conservative
            BudgetConfig(max_budget_usd=35.0),  # Moderate  
            BudgetConfig(max_budget_usd=50.0),  # Full budget
        ]
        
        for i, budget_config in enumerate(budget_configs):
            optimizer = CostOptimizer(budget_config)
            design = optimizer.optimize_for_budget()
            alternatives.append(design)
        
        return alternatives
    
    def display_optimization_results(self, design: OptimizedDesign):
        """Display optimization results."""
        
        console.print(f"[bold blue]💰 Cost-Optimized Experimental Design[/bold blue]")
        
        # Design overview
        design_table = Table(title="🎯 Optimized Design Parameters", show_header=True, header_style="bold magenta")
        design_table.add_column("Parameter", style="cyan")
        design_table.add_column("Value", style="yellow") 
        design_table.add_column("Impact", style="green")
        
        design_table.add_row("Total Questions", str(design.total_questions), "Across all conditions")
        design_table.add_row("Total Trials", str(design.total_trials), "All approach×model combinations")
        design_table.add_row("Models", ', '.join(design.models), "Reduced for cost")
        design_table.add_row("Context Groups", ', '.join(design.context_groups), "Focused on key sizes")  
        design_table.add_row("Question Types", ', '.join(design.question_types), "Essential types only")
        design_table.add_row("Approaches", f"{len(design.approaches)} approaches", "Core reranking comparison")
        design_table.add_row("Statistical Power", f"{design.statistical_power:.1%}", "Probability of detecting effects")
        
        console.print(design_table)
        
        # Cost breakdown
        cost_table = Table(title="💸 Cost Breakdown", show_header=True, header_style="bold red")
        cost_table.add_column("Component", style="cyan")
        cost_table.add_column("Tokens", style="yellow", justify="right")
        cost_table.add_column("Cost per Trial", style="red", justify="right")
        
        cb = design.cost_breakdown
        cost_table.add_row("Query Processing", f"{cb.query_tokens:,}", f"${cb.query_tokens/1000 * self.budget.cost_per_1k_input_tokens:.4f}")
        cost_table.add_row("Context Retrieval", f"{cb.context_tokens:,}", f"${cb.context_tokens/1000 * self.budget.cost_per_1k_input_tokens:.4f}")
        cost_table.add_row("Response Generation", f"{cb.response_tokens:,}", f"${cb.response_tokens/1000 * self.budget.cost_per_1k_output_tokens:.4f}")
        cost_table.add_row("Judge Evaluation", f"{cb.judge_prompt_tokens + cb.judge_response_tokens:,}", 
                          f"${(cb.judge_prompt_tokens/1000 * self.budget.cost_per_1k_input_tokens + cb.judge_response_tokens/1000 * self.budget.cost_per_1k_output_tokens):.4f}")
        cost_table.add_row("[bold]Total per Trial[/bold]", f"[bold]{cb.total_input_tokens + cb.total_output_tokens:,}[/bold]", 
                          f"[bold]${cb.total_cost_usd:.4f}[/bold]")
        
        console.print(cost_table)
        
        # Total cost calculation
        total_experiment_cost = design.total_trials * cb.total_cost_usd
        
        cost_summary = Panel.fit(
            f"[bold green]💰 Total Experiment Cost[/bold green]\n"
            f"[dim]Cost per Trial:[/dim] [yellow]${cb.total_cost_usd:.4f}[/yellow]\n"
            f"[dim]Total Trials:[/dim] [cyan]{design.total_trials:,}[/cyan]\n"
            f"[dim]Total Cost:[/dim] [bold green]${total_experiment_cost:.2f}[/bold green]\n"
            f"[dim]Budget:[/dim] [blue]${self.budget.max_budget_usd:.2f}[/blue]\n"
            f"[dim]Budget Remaining:[/dim] [magenta]${self.budget.max_budget_usd - total_experiment_cost:.2f}[/magenta]",
            title="📊 Budget Analysis",
            border_style="green" if total_experiment_cost <= self.budget.max_budget_usd else "red"
        )
        console.print(cost_summary)
        
        # Cost reduction strategies applied
        if design.cost_reduction_strategies:
            strategies_text = "\n".join([f"• {strategy}" for strategy in design.cost_reduction_strategies])
            strategies_panel = Panel.fit(
                strategies_text,
                title="🔧 Cost Reduction Strategies Applied",
                border_style="yellow"
            )
            console.print(strategies_panel)
        
        # Recommendations
        recommendations = []
        
        if design.statistical_power < 0.7:
            recommendations.append("⚠️ Low statistical power - consider increasing sample size or budget")
        
        if total_experiment_cost > self.budget.max_budget_usd * 0.95:
            recommendations.append("💸 Near budget limit - consider additional cost reductions")
        
        if len(design.context_groups) == 1:
            recommendations.append("📏 Only one context group - limits generalizability")
        
        if len(design.question_types) == 1:
            recommendations.append("❓ Only one question type - may miss approach×complexity interactions")
        
        recommendations.append("🎯 Focus on primary reranking comparison for maximum impact")
        recommendations.append("📈 Consider staged approach: pilot study first, then scale up")
        
        if recommendations:
            rec_panel = Panel.fit(
                "\n".join(recommendations),
                title="💡 Recommendations",
                border_style="blue"
            )
            console.print(rec_panel)


def main():
    """Demonstrate cost optimization."""
    console.print("[bold blue]💰 Cost Optimization for Reranking Value Experiment[/bold blue]")
    
    # Create cost optimizer with €50 budget
    budget_config = BudgetConfig(max_budget_usd=50.0)  # €50 ≈ $50
    optimizer = CostOptimizer(budget_config)
    
    # Create optimized design
    design = optimizer.optimize_for_budget(target_power=0.7)  # Lower power for budget
    optimizer.display_optimization_results(design)
    
    console.print("\n[bold blue]📋 Alternative Budget Scenarios[/bold blue]")
    
    # Show alternatives
    alternatives = optimizer.create_budget_alternatives()
    
    alt_table = Table(title="💵 Budget Alternatives", show_header=True, header_style="bold blue")
    alt_table.add_column("Budget", style="green", justify="right")
    alt_table.add_column("Trials", style="yellow", justify="right")
    alt_table.add_column("Power", style="cyan", justify="right")
    alt_table.add_column("Strategies Applied", style="magenta")
    
    budgets = [20, 35, 50]
    for i, alt in enumerate(alternatives):
        strategies_short = f"{len(alt.cost_reduction_strategies)} reductions" if alt.cost_reduction_strategies else "None"
        alt_table.add_row(
            f"${budgets[i]}",
            str(alt.total_trials),
            f"{alt.statistical_power:.1%}",
            strategies_short
        )
    
    console.print(alt_table)
    
    console.print("\n[bold green]✅ Ready to run cost-optimized experiment![/bold green]")


if __name__ == "__main__":
    main()