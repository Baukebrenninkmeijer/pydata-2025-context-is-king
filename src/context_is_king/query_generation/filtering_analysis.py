"""
Utility functions for analyzing stored WikiText filtering results.

This module provides tools to load and analyze filtering results saved by the DocumentFilter class,
making it easy to inspect filtering outcomes and understand why documents were accepted or rejected.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class FilteringAnalyzer:
    """Analyzer for stored filtering results."""
    
    def __init__(self, results_dir: str = "logs/filtering_results"):
        """Initialize the analyzer with the results directory."""
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            console.print(f"[yellow]Warning: Results directory {self.results_dir} does not exist[/yellow]")
    
    def list_filtering_runs(self) -> List[Dict[str, Any]]:
        """List all available filtering runs with their metadata."""
        if not self.results_dir.exists():
            return []
        
        runs = []
        summary_files = list(self.results_dir.glob("*_summary.md"))
        
        for summary_file in summary_files:
            # Extract run info from filename
            parts = summary_file.stem.split('_')
            if len(parts) >= 3:
                run_id = '_'.join(parts[:-2])  # Everything except timestamp and "summary"
                timestamp = parts[-2]
                
                # Try to load detailed results for more metadata
                detailed_file = self.results_dir / f"{run_id}_{timestamp}_detailed.json"
                if detailed_file.exists():
                    try:
                        with open(detailed_file, 'r') as f:
                            data = json.load(f)
                            metadata = data.get('metadata', {})
                            
                            runs.append({
                                'run_id': run_id,
                                'timestamp': timestamp,
                                'summary_file': summary_file,
                                'detailed_file': detailed_file,
                                'total_documents': metadata.get('total_documents', 0),
                                'filtered_documents': metadata.get('filtered_documents', 0),
                                'filter_rate': metadata.get('filter_rate', 0),
                                'filter_model': metadata.get('filter_model', 'unknown'),
                                'duration': metadata.get('processing_duration_seconds', 0)
                            })
                    except Exception as e:
                        console.print(f"[red]Error reading {detailed_file}: {e}[/red]")
        
        # Sort by timestamp (most recent first)
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        return runs
    
    def show_runs_summary(self):
        """Display a summary of all filtering runs."""
        runs = self.list_filtering_runs()
        
        if not runs:
            console.print("[yellow]No filtering runs found[/yellow]")
            return
        
        table = Table(title="📊 WikiText Filtering Runs", show_header=True, header_style="bold blue")
        table.add_column("Run ID", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Documents", style="yellow", justify="right")
        table.add_column("Passed", style="green", justify="right") 
        table.add_column("Pass Rate", style="magenta", justify="right")
        table.add_column("Model", style="blue")
        table.add_column("Duration", style="dim", justify="right")
        
        for run in runs:
            table.add_row(
                run['run_id'],
                run['timestamp'].replace('-', ':'),
                f"{run['total_documents']:,}",
                f"{run['filtered_documents']:,}",
                f"{run['filter_rate']:.1%}",
                run['filter_model'],
                f"{run['duration']:.1f}s"
            )
        
        console.print(table)
    
    def load_filtering_run(self, run_id: str, timestamp: str = None) -> Optional[Dict[str, Any]]:
        """Load complete data for a specific filtering run."""
        if timestamp is None:
            # Find the most recent run with this run_id
            runs = [r for r in self.list_filtering_runs() if r['run_id'] == run_id]
            if not runs:
                console.print(f"[red]No runs found for run_id: {run_id}[/red]")
                return None
            run_info = runs[0]  # Most recent
            timestamp = run_info['timestamp']
        
        # Load all associated files
        base_path = self.results_dir / f"{run_id}_{timestamp}"
        
        run_data = {
            'run_id': run_id,
            'timestamp': timestamp,
            'files': {}
        }
        
        # Try to load each file type
        file_types = {
            'detailed': 'detailed.json',
            'criteria_analysis': 'criteria_analysis.json',
            'rejected_documents': 'rejected_documents.json',
            'summary': 'summary.md'
        }
        
        for key, suffix in file_types.items():
            file_path = Path(str(base_path) + '_' + suffix)
            if file_path.exists():
                try:
                    if suffix.endswith('.json'):
                        with open(file_path, 'r') as f:
                            run_data['files'][key] = json.load(f)
                    else:
                        with open(file_path, 'r') as f:
                            run_data['files'][key] = f.read()
                    
                    console.print(f"[green]✓ Loaded {key}[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to load {key}: {e}[/red]")
            else:
                console.print(f"[yellow]⚠ {key} file not found[/yellow]")
        
        # Try to load CSV file
        csv_path = Path(str(base_path) + '_filtered_documents.csv')
        if csv_path.exists():
            try:
                run_data['files']['filtered_csv'] = pl.read_csv(csv_path)
                console.print("[green]✓ Loaded filtered documents CSV[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load CSV: {e}[/red]")
        
        return run_data
    
    def analyze_criteria_performance(self, run_data: Dict[str, Any]):
        """Analyze and display criteria performance for a run."""
        criteria_analysis = run_data['files'].get('criteria_analysis')
        if not criteria_analysis:
            console.print("[red]No criteria analysis data available[/red]")
            return
        
        # Check if this is the new question quality format
        question_quality = criteria_analysis.get('question_quality_performance')
        if question_quality:
            self._display_question_quality_analysis(criteria_analysis)
        else:
            # Legacy format
            self._display_legacy_criteria_analysis(criteria_analysis)
    
    def _display_question_quality_analysis(self, criteria_analysis: Dict[str, Any]):
        """Display analysis for the new question quality filtering approach."""
        metadata = criteria_analysis.get('run_metadata', {})
        performance = criteria_analysis.get('question_quality_performance', {})
        summary = criteria_analysis.get('filtering_summary', {})
        
        # Display run metadata
        console.print(Panel.fit(
            f"[bold blue]🎯 Question Quality Filtering Analysis[/bold blue]\n"
            f"[dim]Approach:[/dim] [cyan]{summary.get('filtering_approach', 'Unknown')}[/cyan]\n"
            f"[dim]Consecutive Chunks:[/dim] [yellow]{metadata.get('consecutive_chunks_count', 'Unknown')}[/yellow]\n"
            f"[dim]Question Gen Model:[/dim] [magenta]{metadata.get('question_generation_model', 'Unknown')}[/magenta]\n"
            f"[dim]Question Eval Model:[/dim] [magenta]{metadata.get('question_evaluation_model', 'Unknown')}[/magenta]",
            border_style="blue"
        ))
        
        # Performance table
        table = Table(title="📊 Question Quality Performance", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow", justify="right")
        
        table.add_row("Total Chunk Groups", f"{summary.get('total_chunk_groups_evaluated', 'N/A'):,}")
        table.add_row("Total Chunks Processed", f"{summary.get('total_chunks_processed', 'N/A'):,}")
        table.add_row("Chunks Passed", f"{performance.get('passed_count', 'N/A'):,}")
        table.add_row("Chunks Failed", f"{performance.get('failed_count', 'N/A'):,}")
        table.add_row("Overall Pass Rate", f"{performance.get('pass_rate', 0):.1%}")
        
        console.print(table)
    
    def _display_legacy_criteria_analysis(self, criteria_analysis: Dict[str, Any]):
        """Display analysis for the legacy criteria-based filtering approach."""
        performance = criteria_analysis.get('criteria_performance', {})
        
        table = Table(title="🎯 Criteria Performance Analysis", show_header=True, header_style="bold green")
        table.add_column("Criterion", style="cyan")
        table.add_column("Passed", style="green", justify="right")
        table.add_column("Failed", style="red", justify="right")
        table.add_column("Pass Rate", style="magenta", justify="right")
        table.add_column("Criterion Text", style="dim")
        
        for criterion_label, stats in performance.items():
            criterion_preview = stats['criterion_text'][:80] + "..." if len(stats['criterion_text']) > 80 else stats['criterion_text']
            table.add_row(
                criterion_label,
                f"{stats['passed_count']:,}",
                f"{stats['failed_count']:,}",
                f"{stats['pass_rate']:.1%}",
                criterion_preview
            )
        
        console.print(table)
    
    def analyze_failure_patterns(self, run_data: Dict[str, Any], top_n: int = 10):
        """Analyze and display common failure patterns."""
        criteria_analysis = run_data['files'].get('criteria_analysis')
        if not criteria_analysis:
            console.print("[red]No criteria analysis data available[/red]")
            return
        
        # Check if this is the new question quality format
        question_quality = criteria_analysis.get('question_quality_performance')
        if question_quality:
            self._display_question_quality_failures(criteria_analysis)
        else:
            # Legacy format
            self._display_legacy_failure_patterns(criteria_analysis, top_n)
    
    def _display_question_quality_failures(self, criteria_analysis: Dict[str, Any]):
        """Display failure analysis for question quality filtering."""
        performance = criteria_analysis.get('question_quality_performance', {})
        summary = criteria_analysis.get('filtering_summary', {})
        
        failed_count = performance.get('failed_count', 0)
        total_count = summary.get('total_chunks_processed', 0)
        
        if failed_count == 0:
            console.print("[green]🎉 No failures found - all generated questions passed human-likeness evaluation![/green]")
            return
        
        console.print(Panel.fit(
            f"[bold red]❌ Question Quality Failures[/bold red]\n"
            f"[dim]Failed Chunks:[/dim] [red]{failed_count:,}[/red] out of [cyan]{total_count:,}[/cyan]\n"
            f"[dim]Failure Rate:[/dim] [red]{(1 - performance.get('pass_rate', 0)):.1%}[/red]\n"
            f"[dim]Reason:[/dim] [yellow]Generated questions failed human-likeness evaluation[/yellow]",
            border_style="red"
        ))
        
        console.print("\n[dim]💡 Common reasons for failure might include:[/dim]")
        console.print("  • Questions that are too broad or vague")
        console.print("  • Questions that don't sound natural or conversational") 
        console.print("  • Questions about irrelevant or minor details")
        console.print("  • Questions that are too technical or domain-specific")
        console.print("  • Malformed or grammatically incorrect questions")
    
    def _display_legacy_failure_patterns(self, criteria_analysis: Dict[str, Any], top_n: int = 10):
        """Display failure patterns for legacy criteria-based filtering."""
        failure_analysis = criteria_analysis.get('failure_analysis', {})
        failure_combinations = failure_analysis.get('failure_combinations', {})
        
        if not failure_combinations:
            console.print("[green]No failure patterns found - all documents passed![/green]")
            return
        
        table = Table(title=f"🔍 Top {top_n} Failure Patterns", show_header=True, header_style="bold red")
        table.add_column("Rank", style="dim", justify="right")
        table.add_column("Failed Criteria", style="red")
        table.add_column("Document Count", style="yellow", justify="right")
        table.add_column("% of Total Failed", style="magenta", justify="right")
        
        total_failed = failure_analysis.get('total_failed_documents', 0)
        
        for rank, (pattern, count) in enumerate(list(failure_combinations.items())[:top_n], 1):
            failed_criteria = pattern.replace('|', ', ')
            percentage = (count / total_failed * 100) if total_failed > 0 else 0
            
            table.add_row(
                str(rank),
                failed_criteria,
                f"{count:,}",
                f"{percentage:.1f}%"
            )
        
        console.print(table)
    
    def show_rejected_documents_sample(self, run_data: Dict[str, Any], max_docs: int = 5):
        """Show a sample of rejected documents with their failure reasons."""
        rejected_data = run_data['files'].get('rejected_documents')
        if not rejected_data:
            console.print("[red]No rejected documents data available[/red]")
            return
        
        rejected_docs = rejected_data.get('documents', [])[:max_docs]
        
        for i, doc in enumerate(rejected_docs, 1):
            failed_criteria = doc.get('failed_criteria', [])
            
            # Handle both old and new format
            if failed_criteria:
                failed_criteria_str = ", ".join(failed_criteria)
            else:
                # New format - check if question_quality failed
                evaluations = doc.get('criteria_evaluations', {})
                if not evaluations.get('question_quality', True):
                    failed_criteria_str = "Generated question failed human-likeness evaluation"
                else:
                    failed_criteria_str = "Unknown failure reason"
            
            panel_content = f"""[bold]Document ID:[/bold] {doc['document_id']}
[bold]Source:[/bold] {doc.get('source_title', 'Unknown')}
[bold]Content Length:[/bold] {doc.get('content_length', 0):,} characters
[bold]Failure Reason:[/bold] {failed_criteria_str}

[bold]Content Preview:[/bold]
{doc.get('content_preview', 'No preview available')}"""
            
            panel = Panel(
                panel_content,
                title=f"🚫 Rejected Document {i}/{len(rejected_docs)}",
                border_style="red"
            )
            console.print(panel)
    
    def compare_runs(self, run_ids: List[str]):
        """Compare multiple filtering runs."""
        run_data_list = []
        
        for run_id in run_ids:
            run_data = self.load_filtering_run(run_id)
            if run_data and 'detailed' in run_data['files']:
                run_data_list.append(run_data)
        
        if len(run_data_list) < 2:
            console.print("[red]Need at least 2 valid runs to compare[/red]")
            return
        
        # Create comparison table
        table = Table(title="⚖️ Filtering Runs Comparison", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        
        for run_data in run_data_list:
            table.add_column(
                f"{run_data['run_id']}\n({run_data['timestamp']})", 
                style="yellow", 
                justify="right"
            )
        
        # Extract metrics for comparison
        metrics = [
            ("Total Documents", lambda x: x['files']['detailed']['metadata']['total_documents']),
            ("Filtered Documents", lambda x: x['files']['detailed']['metadata']['filtered_documents']),
            ("Pass Rate", lambda x: f"{x['files']['detailed']['metadata']['filter_rate']:.1%}"),
            ("Filter Model", lambda x: x['files']['detailed']['metadata']['filter_model']),
            ("Duration (s)", lambda x: f"{x['files']['detailed']['metadata']['processing_duration_seconds']:.1f}"),
        ]
        
        for metric_name, extractor in metrics:
            row = [metric_name]
            for run_data in run_data_list:
                try:
                    value = extractor(run_data)
                    row.append(str(value))
                except Exception:
                    row.append("N/A")
            table.add_row(*row)
        
        console.print(table)


def analyze_latest_run(results_dir: str = "logs/filtering_results"):
    """Quick function to analyze the most recent filtering run."""
    analyzer = FilteringAnalyzer(results_dir)
    runs = analyzer.list_filtering_runs()
    
    if not runs:
        console.print("[yellow]No filtering runs found[/yellow]")
        return
    
    latest_run = runs[0]
    console.print(f"[blue]Analyzing latest run: {latest_run['run_id']}[/blue]")
    
    run_data = analyzer.load_filtering_run(latest_run['run_id'], latest_run['timestamp'])
    if run_data:
        analyzer.analyze_criteria_performance(run_data)
        console.print()
        analyzer.analyze_failure_patterns(run_data)
        console.print()
        analyzer.show_rejected_documents_sample(run_data, max_docs=3)


def list_all_runs(results_dir: str = "logs/filtering_results"):
    """Quick function to list all filtering runs."""
    analyzer = FilteringAnalyzer(results_dir)
    analyzer.show_runs_summary()


if __name__ == "__main__":
    # Example usage
    console.print("[bold blue]WikiText Filtering Results Analyzer[/bold blue]")
    analyze_latest_run()