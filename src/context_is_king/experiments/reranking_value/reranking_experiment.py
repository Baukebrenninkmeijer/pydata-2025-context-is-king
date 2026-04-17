#!/usr/bin/env python3
"""
Reranking Value Experiment Controller

🎯 DATASET: LongMemEval Conversations
📊 PURPOSE: Answer "Does reranking still make sense in a post-RAG world?"

This module coordinates the three-way comparison experiment to answer:
"Does reranking still make sense in a post-RAG world?"

Compares:
1. Full Context Retrieval (no RAG)
2. Enhanced RAG without Reranking
3. Enhanced RAG with Reranking

DATASET REQUIREMENTS:
- Uses LongMemEval conversation dataset specifically
- Requires focused vs full conversation structure
- Questions with expected answers for evaluation
- Compatible with LongMemEval ground truth extraction

EVALUATION:
- Comprehensive evaluation with LLM-as-a-judge
- Retrieval quality assessment with IR metrics
- Cost and latency analysis across approaches
"""

import json
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Import retrieval quality assessment components
from retrieval_quality import (
    GroundTruthAnnotator,
    RetrievalQualityAnalyzer,
    RetrievalQualityEvaluator,
    RetrievalQualityResult,
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.table import Table

from context_is_king.evaluation import (
    ContextMeasurement,
    ContextTracker,
    JudgeEvaluation,
    JudgeEvaluator,
    LongMemEvalEvaluator,
)

# Import reusable components from main package
from context_is_king.models import ModelInterface
from context_is_king.rag_pipeline import (
    ContextAssembly,
    DualRetrieval,
    EmbeddingStorageManager,
    QueryRewriter,
    RerankerModule,
    RetrievalEngine,
)
from context_is_king.rag_pipeline.types import DocumentChunk, PipelineConfig, RetrievalResult

console = Console()


@dataclass
class ExperimentConfig:
    """Configuration for reranking value experiment."""

    experiment_id: str
    output_dir: str

    # Data sources
    longmemeval_path: str
    document_collections: list[str]

    # Models and approaches
    models: list[str]
    context_groups: list[str]  # short, medium, long
    question_types: list[str]  # factual, reasoning, synthesis, multi_hop

    # Retrieval settings
    retrieval_k: int = 20
    fusion_method: str = "rrf"

    # Experiment settings
    iterations_per_question: int = 1
    save_intermediate: bool = True
    checkpoint_interval: int = 10
    max_questions: int | None = None  # Limit number of questions to process
    auto_confirm: bool = False  # Skip user confirmation for automated runs

    # Judge settings
    judge_model: str = "azure/gpt-4.1"
    judge_temperature: float = 0.1

    @classmethod
    def cost_optimized(cls, **kwargs):
        """Create cost-optimized configuration for €50 budget."""
        required_fields = ["experiment_id", "output_dir", "longmemeval_path", "document_collections"]

        # Cost-optimized defaults
        defaults = {
            "models": ["claude-sonnet-3.7"],  # Single high-quality model
            "context_groups": ["short", "medium"],  # Focus on key sizes
            "question_types": ["factual", "reasoning", "synthesis", "multi_hop"],  # All essential types
            "retrieval_k": 20,
            "fusion_method": "rrf",
            "iterations_per_question": 1,
            "save_intermediate": True,
            "checkpoint_interval": 10,
            "judge_model": "gpt-4.1",
            "judge_temperature": 0.1,
        }

        # Merge with provided kwargs
        defaults.update(kwargs)

        # Validate required fields are provided
        for field in required_fields:
            if field not in defaults:
                raise ValueError(f"Required field '{field}' must be provided")

        return cls(**defaults)


@dataclass
class ExperimentResult:
    """Result from a single experimental trial."""

    experiment_id: str
    trial_id: str
    approach: str  # full_context, enhanced_rag_no_rerank, enhanced_rag_rerank
    model: str
    question_id: str
    question: str
    question_type: str
    context_group: str
    expected_answer: str
    model_answer: str
    judge_evaluation: JudgeEvaluation
    context_measurement: ContextMeasurement
    approach_metadata: dict[str, Any]
    retrieval_quality: RetrievalQualityResult | None  # Retrieval quality metrics
    timestamp: float


@dataclass
class ExperimentSummary:
    """Summary of complete experiment results."""

    experiment_id: str
    config: ExperimentConfig
    results: list[ExperimentResult]
    approach_comparison: dict[str, Any]
    context_analysis: dict[str, Any]
    cost_analysis: dict[str, Any]
    metadata: dict[str, Any]


class RerankingValueExperiment:
    """Main controller for reranking value experiment."""

    def __init__(self, config: ExperimentConfig):
        # Load environment variables from root .env file
        root_dir = Path(__file__).parent.parent.parent
        env_file = root_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)
            console.print(f"[green]Loaded environment from: {env_file}[/green]")
        else:
            console.print(f"[yellow]Warning: .env file not found at {env_file}[/yellow]")

        self.config = config
        self.experiment_id = config.experiment_id
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components with ChromaDB configuration
        project_root = Path(__file__).parent.parent.parent.parent.parent
        chroma_path = project_root / "data" / "vector_stores" / "longmemeval"

        # Configure pipeline with SentenceTransformer fallback model
        # Primary retrieval uses direct ChromaDB queries with Azure/OpenAI embeddings
        # EmbeddingStorageManager needs a valid SentenceTransformer model for fallback cases
        self.pipeline_config = PipelineConfig(
            nvidia_model=config.models[0] if config.models else "moonshotai/kimi-k2-instruct",
            judge_model=config.judge_model,
            # ChromaDB configuration - point to existing ingested data
            chroma_db_path=str(chroma_path),
            # SentenceTransformer model for fallback (primary path uses OpenAI directly)
            embedding_model="all-MiniLM-L6-v2",
            # GPU configuration for reranker
            reranker_device="auto",  # Use GPU if available (CUDA/MPS)
            # Retrieval settings
            default_k=config.retrieval_k,
            # Other pipeline settings
            chunk_size=500,  # Match LongMemEval chunking
            chunk_overlap=50,
        )

        # Initialize shared model interface (uses OpenAI SDK with ORQ proxy)
        orq_api_key = os.getenv("ORQ_API_KEY")
        if not orq_api_key:
            raise ValueError("ORQ_API_KEY environment variable is required")

        self.model_interface = ModelInterface(api_key=orq_api_key)

        # Initialize pipeline components
        self._initialize_pipeline_components()

        # Initialize direct ChromaDB connection for LongMemEval data
        self._initialize_chroma_connection()

        # Initialize evaluation components
        self.judge_evaluator = JudgeEvaluator(self.model_interface, config.judge_model)
        self.context_tracker = ContextTracker()
        self.longmemeval_evaluator = LongMemEvalEvaluator()

        # Initialize retrieval quality assessment
        self.ground_truth_annotator = GroundTruthAnnotator()
        self.retrieval_quality_evaluator = RetrievalQualityEvaluator(self.ground_truth_annotator)
        self.retrieval_quality_analyzer = RetrievalQualityAnalyzer()

        # Load LongMemEval full conversation data for full context approach
        self._load_longmemeval_full_data()

        # Experiment state
        self.results: list[ExperimentResult] = []
        self.interrupted = False

        # Setup interrupt handling
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        # Display initialization success
        panel = Panel.fit(
            f"[bold green]✓ Reranking Value Experiment Initialized[/bold green]\n"
            f"[dim]Experiment ID:[/dim] [cyan]{self.experiment_id}[/cyan]\n"
            f"[dim]Models:[/dim] [yellow]{', '.join(self.config.models)}[/yellow]\n"
            f"[dim]Approaches:[/dim] [magenta]Full Context, Enhanced RAG (±Reranking)[/magenta]\n"
            f"[dim]Output:[/dim] [blue]{self.config.output_dir}[/blue]",
            title="🧪 Experiment Ready",
            border_style="green",
        )
        console.print(panel)

    def _initialize_chroma_connection(self):
        """Initialize direct ChromaDB connection for LongMemEval collection with OpenAI embeddings."""
        try:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            chroma_path = project_root / "data" / "vector_stores" / "longmemeval"

            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            console.print(f"[green]✓ Connected to ChromaDB at {chroma_path}[/green]")

            # Initialize OpenAI client for embeddings (using ORQ proxy)
            self.openai_client = OpenAI(api_key=os.getenv("ORQ_API_KEY"), base_url="https://api.orq.ai/v2/proxy")

            # Try to get the existing collection
            collection_name = "longmemeval_conversations"
            try:
                self.longmemeval_collection = self.chroma_client.get_collection(name=collection_name)
                collection_count = self.longmemeval_collection.count()
                console.print(
                    f"[green]✓ Connected to ChromaDB collection: {collection_name} ({collection_count:,} chunks)[/green]"
                )
                self.chroma_available = True
            except Exception:
                console.print(f"[red]Warning: Collection '{collection_name}' not found[/red]")
                # List available collections for debugging
                collections = self.chroma_client.list_collections()
                if collections:
                    console.print("[yellow]Available collections might have different names[/yellow]")
                    for col in collections:
                        console.print(f"  - {col.name} ({col.count()} items)")
                else:
                    console.print("[yellow]No collections found in ChromaDB[/yellow]")

                # Try to get any available collection as fallback
                if collections:
                    fallback_collection = collections[0]
                    console.print(f"[yellow]Using fallback collection: {fallback_collection.name}[/yellow]")
                    self.longmemeval_collection = fallback_collection
                    self.chroma_available = True
                else:
                    self.longmemeval_collection = None
                    self.chroma_available = False

        except Exception as e:
            console.print(f"[red]Failed to initialize ChromaDB connection: {e}[/red]")
            console.print("[yellow]Disabling ChromaDB integration, will use mock retrieval[/yellow]")
            self.chroma_client = None
            self.longmemeval_collection = None
            self.chroma_available = False

    def _load_longmemeval_full_data(self):
        """Load LongMemEval full conversation data from CSV for full context approach."""
        try:
            project_root = Path(__file__).parent.parent.parent
            full_csv_path = project_root / "data" / "LongMemEval" / "cleaned_longmemeval_s_full.csv"

            if not full_csv_path.exists():
                raise FileNotFoundError(f"LongMemEval full CSV not found at: {full_csv_path}")

            console.print(f"[blue]Loading LongMemEval full conversation data from {full_csv_path}[/blue]")

            # Load the full CSV
            full_df = pd.read_csv(full_csv_path)
            console.print(f"[green]✓ Loaded {len(full_df)} full conversations[/green]")

            # Validate required columns
            required_columns = ["custom_id", "full_prompt"]
            missing_columns = [col for col in required_columns if col not in full_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in full CSV: {missing_columns}")

            # Create lookup dictionary for fast access
            self.longmemeval_full_data = dict(zip(full_df["custom_id"], full_df["full_prompt"]))
            console.print(
                f"[green]✓ Created full conversation lookup for {len(self.longmemeval_full_data)} conversations[/green]"
            )

            # Store custom_ids for validation against question data later
            self.longmemeval_full_ids = set(full_df["custom_id"])

        except Exception as e:
            console.print(f"[red]Failed to load LongMemEval full conversation data: {e}[/red]")
            console.print("[red]Full context approach will not work correctly without this data[/red]")
            self.longmemeval_full_data = {}
            self.longmemeval_full_ids = set()

    def _validate_conversation_id_matching(self, questions: list[dict[str, Any]]):
        """Validate that question custom_ids match available full conversation data."""

        if not hasattr(self, "longmemeval_full_ids") or not self.longmemeval_full_ids:
            console.print("[yellow]⚠️  No full conversation data loaded - skipping ID validation[/yellow]")
            return

        console.print(f"[blue]🔍 Validating conversation ID matching for {len(questions)} questions...[/blue]")

        # Extract custom_ids from questions
        question_ids = set(q.get("id") for q in questions if q.get("id"))

        # Find mismatches
        missing_in_full = question_ids - self.longmemeval_full_ids
        extra_in_full = self.longmemeval_full_ids - question_ids

        # Report validation results
        validation_table = Table(title="🔗 Conversation ID Validation", show_header=True, header_style="bold blue")
        validation_table.add_column("Data Set", style="cyan")
        validation_table.add_column("Count", style="yellow", justify="right")
        validation_table.add_column("Status", justify="center")

        validation_table.add_row("Question IDs", str(len(question_ids)), "📝")
        validation_table.add_row("Full Conversation IDs", str(len(self.longmemeval_full_ids)), "💬")
        validation_table.add_row(
            "Matching IDs",
            str(len(question_ids & self.longmemeval_full_ids)),
            "✅" if len(missing_in_full) == 0 else "⚠️",
        )

        console.print(validation_table)

        # Handle missing full conversations
        if missing_in_full:
            console.print(f"[red]❌ {len(missing_in_full)} questions have no matching full conversation:[/red]")
            for missing_id in sorted(list(missing_in_full)[:10]):  # Show first 10
                console.print(f"  - {missing_id}")
            if len(missing_in_full) > 10:
                console.print(f"  ... and {len(missing_in_full) - 10} more")

            console.print("[red]Full context approach will fail for these questions.[/red]")
            raise ValueError(
                f"Missing full conversation data for {len(missing_in_full)} questions. "
                f"Ensure all question IDs exist in the LongMemEval full CSV file."
            )

        # Log extra conversations (informational only)
        if extra_in_full:
            console.print(
                f"[dim]ℹ️  {len(extra_in_full)} extra conversations in full CSV (not used in this experiment)[/dim]"
            )

        console.print(
            f"[green]✅ ID validation passed: All {len(question_ids)} questions have matching full conversations[/green]"
        )

    def _query_longmemeval_collection(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Query the LongMemEval collection directly with OpenAI embeddings."""
        if not self.chroma_available or not self.longmemeval_collection:
            return []

        try:
            # Generate OpenAI embedding for the query
            try:
                embedding_response = self.openai_client.embeddings.create(
                    model="azure/text-embedding-3-small",
                    input=query,
                    dimensions=768,  # Match ChromaDB collection dimension
                )
            except (AttributeError, Exception) as e:
                if "'str' object has no attribute 'data'" in str(e):
                    console.print(f"[red]ORQ proxy returned error string instead of embedding response[/red]")
                elif "404" in str(e) or "not found" in str(e).lower():
                    console.print(
                        f"[red]Embedding model 'azure/text-embedding-3-small' not configured in ORQ workspace[/red]"
                    )
                else:
                    console.print(f"[red]Embedding API error: {e}[/red]")
                console.print(f"[yellow]Falling back to ground truth data[/yellow]")
                return []

            # Safe access to embedding data
            if not hasattr(embedding_response, "data") or len(embedding_response.data) == 0:
                console.print(f"[red]Invalid embedding response format: {type(embedding_response)}[/red]")
                return []

            query_embedding = embedding_response.data[0].embedding

            # Query ChromaDB collection
            results = self.longmemeval_collection.query(query_embeddings=[query_embedding], n_results=k)

            # Convert to RetrievalResult objects
            retrieval_results = []
            if results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0] if results["documents"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []
                distances = results["distances"][0] if results["distances"] else []

                for i, (chunk_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances, strict=False)
                ):
                    # Create DocumentChunk
                    doc_chunk = DocumentChunk(
                        content=document,
                        doc_id=metadata.get("doc_id", "unknown") if metadata else "unknown",
                        chunk_id=chunk_id,
                        start_char=metadata.get("start_char", 0) if metadata else 0,
                        end_char=metadata.get("end_char", len(document)) if metadata else len(document),
                        metadata=metadata or {},
                    )

                    # Create RetrievalResult
                    retrieval_result = RetrievalResult(
                        chunk=doc_chunk,
                        similarity_score=1.0 - distance,  # Convert distance to similarity
                        rank=i + 1,
                    )
                    retrieval_results.append(retrieval_result)

            return retrieval_results

        except Exception as e:
            console.print(f"[red]Error querying LongMemEval collection: {e}[/red]")
            return []

    def _initialize_pipeline_components(self):
        """Initialize RAG pipeline components with rich progress feedback."""

        with Status("[blue]🔧 Initializing RAG pipeline components...", spinner="dots") as status:
            components = []
            errors = []

            # Core components - only initialize what we actually need
            # Note: For this experiment, we use ModelInterface instead of GenerationInterface
            # to avoid NVIDIA API key requirements and leverage ORQ proxy
            try:
                status.update("Loading embedding manager...")
                self.embedding_manager = EmbeddingStorageManager(self.pipeline_config)
                components.append("✓ Embedding Manager")
                console.print(
                    "[dim]Note: Primary retrieval uses Azure OpenAI embeddings via direct ChromaDB queries[/dim]"
                )

                status.update("Initializing retrieval engine...")
                self.retrieval_engine = RetrievalEngine(self.embedding_manager, self.pipeline_config)
                components.append("✓ Retrieval Engine")

                status.update("Setting up reranker...")
                self.reranker = RerankerModule(self.pipeline_config)
                components.append("✓ Reranker Module")

                status.update("Configuring context assembly...")
                self.context_assembly = ContextAssembly(self.pipeline_config)
                components.append("✓ Context Assembly")

            except Exception as e:
                errors.append(f"Core pipeline components: {e!s}")
                self.embedding_manager = None
                self.retrieval_engine = None
                self.reranker = None
                self.context_assembly = None
                components.append("⚠ Using mock components (some dependencies unavailable)")

            # Enhanced components (use shared model interface with ORQ proxy)
            try:
                status.update("Initializing query rewriter...")
                self.query_rewriter = QueryRewriter(self.model_interface)
                components.append("✓ Query Rewriter (ORQ)")

                status.update("Setting up dual retrieval...")
                self.dual_retrieval = (
                    DualRetrieval(self.retrieval_engine, self.query_rewriter) if self.retrieval_engine else None
                )
                components.append("✓ Dual Retrieval" if self.dual_retrieval else "⚠ Dual Retrieval (mock mode)")

            except Exception as e:
                errors.append(f"Enhanced components: {e!s}")

        # Display results in a nice table
        table = Table(title="🔧 Pipeline Components Status", show_header=False, box=None)
        table.add_column("Component", style="cyan")

        for component in components:
            if "✓" in component:
                table.add_row(f"[green]{component}[/green]")
            elif "⚠" in component:
                table.add_row(f"[yellow]{component}[/yellow]")
            else:
                table.add_row(component)

        console.print(table)

        # Show warnings if any
        if errors:
            warning_text = "\n".join([f"• {error}" for error in errors])
            warning_panel = Panel(warning_text, title="⚠️ Initialization Warnings", border_style="yellow")
            console.print(warning_panel)

    def run_experiment(self) -> ExperimentSummary:
        """
        Run the complete three-way reranking experiment.

        Returns:
            ExperimentSummary with all results and analysis
        """
        # Show experiment start banner
        start_panel = Panel.fit(
            f"[bold blue]🚀 Starting Reranking Value Experiment[/bold blue]\n"
            f"[dim]Research Question:[/dim] [white]Does reranking still make sense in a post-RAG world?[/white]\n"
            f"[dim]Experiment ID:[/dim] [cyan]{self.experiment_id}[/cyan]\n"
            f"[dim]Started at:[/dim] [green]{time.strftime('%Y-%m-%d %H:%M:%S')}[/green]",
            title="📊 Experiment Launch",
            border_style="blue",
        )
        console.print(start_panel)
        start_time = time.time()

        try:
            # Load experiment data
            with Status("📚 Loading LongMemEval questions...", spinner="dots"):
                questions = self._load_longmemeval_questions()
            console.print(f"[green]✓ Loaded {len(questions)} LongMemEval questions[/green]")

            # Generate all experimental conditions
            with Status("🔄 Generating experimental conditions...", spinner="dots"):
                conditions = self._generate_experimental_conditions(questions)

            # Show experiment overview
            overview_table = Table(title="📋 Experiment Overview", show_header=True, header_style="bold magenta")
            overview_table.add_column("Metric", style="cyan")
            overview_table.add_column("Value", style="yellow")
            overview_table.add_row("Total Trials", str(len(conditions)))
            overview_table.add_row("Models", ", ".join(self.config.models))
            overview_table.add_row("Approaches", "Full Context, Basic RAG (±Reranking), Enhanced RAG (±Reranking)")
            overview_table.add_row("Context Groups", ", ".join(self.config.context_groups))
            overview_table.add_row("Question Types", ", ".join(self.config.question_types))
            overview_table.add_row("Checkpoint Interval", f"Every {self.config.checkpoint_interval} trials")
            console.print(overview_table)

            # Ask for confirmation before starting large experiments
            if len(conditions) > 20:
                trial_breakdown = (
                    f"{len(questions)} questions × "
                    f"{len(self.config.models)} model(s) × "
                    f"5 approaches × "
                    f"{self.config.iterations_per_question} iteration(s) = "
                    f"{len(conditions)} trials"
                )

                confirmation_panel = Panel.fit(
                    f"[bold yellow]⚠️ Large Experiment Warning[/bold yellow]\n"
                    f"[dim]Trial Calculation:[/dim] [white]{trial_breakdown}[/white]\n"
                    f"[dim]Estimated Duration:[/dim] [yellow]~{len(conditions) * 0.5:.0f} minutes[/yellow] (rough estimate)\n"
                    f"[dim]Cost Estimate:[/dim] [cyan]Depends on model and context size[/cyan]\n\n"
                    f"[dim]Tip:[/dim] Use [bold]--max-questions N[/bold] to limit experiment size",
                    title="🔍 Experiment Size Check",
                    border_style="yellow",
                )
                console.print(confirmation_panel)

                if not self.config.auto_confirm:
                    confirm = input(f"\nThis will run {len(conditions)} trials. Continue? [y/N]: ")
                    if confirm.lower() not in ["y", "yes"]:
                        console.print("[yellow]Experiment cancelled by user[/yellow]")
                        return ExperimentSummary(
                            experiment_id=self.experiment_id,
                            config=self.config,
                            results=[],
                            approach_comparison={"cancelled": True},
                            context_analysis={"cancelled": True},
                            cost_analysis={"cancelled": True},
                            metadata={"cancelled": True, "total_trials": 0},
                        )
                else:
                    console.print(f"[green]Auto-confirming experiment with {len(conditions)} trials[/green]")

            # Run experiments with detailed progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
            ) as progress:
                main_task = progress.add_task("[cyan]🔬 Running experiment trials", total=len(conditions))

                success_count = 0
                failure_count = 0
                approach_counts = {
                    "full_context": 0,
                    "basic_rag_no_rerank": 0,
                    "basic_rag_rerank": 0,
                    "enhanced_rag_no_rerank": 0,
                    "enhanced_rag_rerank": 0,
                }

                for i, condition in enumerate(conditions):
                    if self.interrupted:
                        progress.update(main_task, description="[yellow]⏸️ Experiment interrupted, saving results...")
                        console.print("\n[bold yellow]⚠️ Experiment interrupted by user[/bold yellow]")
                        break

                    # Update progress description with current trial info
                    approach = condition["approach"]
                    model = condition["model"]
                    trial_desc = f"[cyan]Trial {i + 1}/{len(conditions)} • {approach}"
                    progress.update(main_task, description=trial_desc)

                    try:
                        result = self._run_single_trial(condition)
                        self.results.append(result)
                        success_count += 1
                        approach_counts[approach] += 1

                        # Check if judge evaluation failed and log it
                        if not result.judge_evaluation.is_correct and hasattr(result.judge_evaluation, "metadata"):
                            judge_error = result.judge_evaluation.metadata.get("error")
                            if judge_error and "failed" in judge_error.lower():
                                progress.stop()
                                console.print(f"[yellow]⚠️ Judge evaluation issue in trial {i + 1}:[/yellow]")
                                console.print(f"[yellow]   Question: {result.question[:60]}...[/yellow]")
                                console.print(f"[yellow]   Judge Error: {judge_error}[/yellow]")
                                progress.start()

                        # Save checkpoint periodically
                        if (i + 1) % self.config.checkpoint_interval == 0:
                            progress.update(main_task, description="[blue]💾 Saving checkpoint...")
                            self._save_checkpoint()
                            console.print(f"[dim]💾 Checkpoint saved ({i + 1} trials completed)[/dim]")

                    except Exception as e:
                        failure_count += 1
                        progress.stop()
                        console.print(f"[red]❌ Trial {i + 1} failed: {e}[/red]")
                        progress.start()
                        continue

                    progress.update(main_task, advance=1)

                # Final progress update
                progress.update(
                    main_task, description=f"[green]✅ Completed {success_count} trials ({failure_count} failed)"
                )

            # Show completion statistics
            stats_table = Table(title="📊 Trial Statistics", show_header=True, header_style="bold green")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Count", style="yellow", justify="right")
            stats_table.add_row("✅ Successful Trials", str(success_count))
            stats_table.add_row("❌ Failed Trials", str(failure_count))
            stats_table.add_row("📈 Full Context", str(approach_counts["full_context"]))
            stats_table.add_row("🔍 Basic RAG (no rerank)", str(approach_counts["basic_rag_no_rerank"]))
            stats_table.add_row("🎯 Basic RAG (rerank)", str(approach_counts["basic_rag_rerank"]))
            stats_table.add_row("🚀 Enhanced RAG (no rerank)", str(approach_counts["enhanced_rag_no_rerank"]))
            stats_table.add_row("⭐ Enhanced RAG (rerank)", str(approach_counts["enhanced_rag_rerank"]))
            console.print(stats_table)

            # Generate final analysis with status
            with Status("📊 Analyzing results and generating summary...", spinner="dots"):
                summary = self._generate_experiment_summary()

            # Save final results
            with Status("💾 Saving final results...", spinner="dots"):
                self._save_final_results(summary)

            # Show completion summary
            duration = time.time() - start_time
            completion_panel = Panel.fit(
                f"[bold green]🎉 Experiment Completed Successfully![/bold green]\n"
                f"[dim]Duration:[/dim] [yellow]{duration:.1f} seconds[/yellow] ([blue]{duration / 60:.1f} minutes[/blue])\n"
                f"[dim]Total Trials:[/dim] [cyan]{success_count}[/cyan] successful, [red]{failure_count}[/red] failed\n"
                f"[dim]Results:[/dim] [blue]{self.output_dir}[/blue]\n"
                f"[dim]Best Approach:[/dim] [magenta]{summary.approach_comparison.get('best_accuracy', 'TBD')}[/magenta]",
                title="✅ Experiment Complete",
                border_style="green",
            )
            console.print(completion_panel)

            # Display retrieval quality summary
            if summary.metadata.get("retrieval_quality_summary", {}).get("total_questions_evaluated", 0) > 0:
                with Status("📊 Generating retrieval quality report...", spinner="dots"):
                    self.retrieval_quality_analyzer.display_results_summary()

            return summary

        except KeyboardInterrupt:
            # Graceful interrupt handling
            interrupt_panel = Panel.fit(
                f"[bold yellow]⚠️ Experiment Interrupted by User[/bold yellow]\n"
                f"[dim]Trials completed:[/dim] [cyan]{len(self.results)}[/cyan]\n"
                f"[dim]Checkpoint saved:[/dim] [blue]{self.output_dir}[/blue]\n"
                f"[dim]Resume with:[/dim] [green]--resume-from {self.experiment_id}[/green]",
                title="⏸️ Interrupted",
                border_style="yellow",
            )
            console.print(interrupt_panel)
            if self.results:
                self._save_checkpoint()
            raise

        except Exception as e:
            # Error handling with details
            error_panel = Panel.fit(
                f"[bold red]❌ Experiment Failed[/bold red]\n"
                f"[dim]Error:[/dim] [red]{e!s}[/red]\n"
                f"[dim]Trials completed:[/dim] [cyan]{len(self.results)}[/cyan]\n"
                f"[dim]Checkpoint saved:[/dim] [blue]{self.output_dir if self.results else 'None'}[/blue]",
                title="💥 Error",
                border_style="red",
            )
            console.print(error_panel)
            if self.results:
                self._save_checkpoint()
            raise

    def _load_longmemeval_questions(self) -> list[dict[str, Any]]:
        """Load LongMemEval questions from ground truth data file."""
        # Look for ground truth data in project root data directory
        project_root = Path(__file__).parent.parent.parent
        ground_truth_file = project_root / "data" / "longmemeval_ground_truth.json"

        if not ground_truth_file.exists():
            console.print(f"[yellow]Ground truth file not found at {ground_truth_file}[/yellow]")
            console.print("[yellow]Falling back to sample data...[/yellow]")
            return self._load_sample_questions()

        console.print(f"[green]Loading ground truth from {ground_truth_file}[/green]")

        with open(ground_truth_file) as f:
            ground_truth_data = json.load(f)

        # Extract questions from ground truth format
        gt_questions = ground_truth_data.get("questions", [])

        # Transform to experiment format and add ground truth information
        questions = []
        for gt_question in gt_questions:
            # Categorize by context size (based on focused content tokens)
            focused_tokens = gt_question.get("focused_content_tokens", 0)
            if focused_tokens <= 25000:
                context_group = "short"
            elif focused_tokens <= 75000:
                context_group = "medium"
            else:
                context_group = "long"

            # Infer question type from question content
            question_text = gt_question.get("question", "").lower()
            if any(keyword in question_text for keyword in ["what", "when", "where", "who", "how many"]):
                question_type = "factual"
            elif any(keyword in question_text for keyword in ["why", "how", "explain", "analyze"]):
                question_type = "reasoning"
            elif any(keyword in question_text for keyword in ["connect", "relate", "combine", "synthesis"]):
                question_type = "synthesis"
            else:
                question_type = "multi_hop"

            question_data = {
                "id": gt_question.get("custom_id"),
                "question": gt_question.get("question"),
                "expected_answer": gt_question.get("expected_answer"),
                "context_group": context_group,
                "question_type": question_type,
                "focused_content": gt_question.get("focused_content"),
                "focused_content_tokens": focused_tokens,
                "focused_content_length": gt_question.get("focused_content_length", 0),
                # Ground truth chunk information for retrieval evaluation
                "relevant_chunks": gt_question.get("relevant_chunks", []),
                "ground_truth_source": "longmemeval_extraction",
            }
            questions.append(question_data)

        # Filter questions based on config with detailed feedback
        filtered_questions = []
        context_filter_count = 0
        question_type_filter_count = 0

        for question in questions:
            context_match = (
                not self.config.context_groups or question.get("context_group") in self.config.context_groups
            )
            type_match = not self.config.question_types or question.get("question_type") in self.config.question_types

            if context_match and type_match:
                filtered_questions.append(question)
            else:
                if not context_match:
                    context_filter_count += 1
                if not type_match:
                    question_type_filter_count += 1

        # Show filtering statistics if any questions were filtered out
        if len(filtered_questions) < len(questions):
            filter_table = Table(title="🔍 Question Filtering Results", show_header=True, header_style="bold cyan")
            filter_table.add_column("Filter Type", style="cyan")
            filter_table.add_column("Criteria", style="yellow")
            filter_table.add_column("Excluded", style="red", justify="right")
            filter_table.add_column("Status", justify="center")

            context_criteria = ", ".join(self.config.context_groups) if self.config.context_groups else "All"
            type_criteria = ", ".join(self.config.question_types) if self.config.question_types else "All"

            filter_table.add_row(
                "Context Groups",
                context_criteria,
                str(context_filter_count),
                "✓" if context_filter_count == 0 else "🔍",
            )
            filter_table.add_row(
                "Question Types",
                type_criteria,
                str(question_type_filter_count),
                "✓" if question_type_filter_count == 0 else "🔍",
            )
            filter_table.add_row(
                "[bold]Total Retained[/bold]",
                "[bold]After Filtering[/bold]",
                f"[bold green]{len(filtered_questions)}[/bold green]",
                "✅",
            )

            console.print(filter_table)

        # Apply max_questions limit if specified
        if self.config.max_questions and len(filtered_questions) > self.config.max_questions:
            console.print(
                f"[yellow]📊 Applying question limit: {self.config.max_questions} (from {len(filtered_questions)} available)[/yellow]"
            )
            filtered_questions = filtered_questions[: self.config.max_questions]

        # Validate conversation ID matching between questions and full CSV data
        self._validate_conversation_id_matching(filtered_questions)

        console.print(f"[green]✅ Final question set: {len(filtered_questions)} questions ready for experiment[/green]")
        return filtered_questions

    def _load_sample_questions(self) -> list[dict[str, Any]]:
        """Fallback to load sample questions if ground truth not available."""
        data_file = Path(self.config.longmemeval_path) / "longmemeval_selected.json"

        if not data_file.exists():
            # Fallback to project data directory
            project_root = Path(__file__).parent.parent.parent.parent.parent
            data_file = project_root / "data" / "longmemeval_selected.json"

        if not data_file.exists():
            raise FileNotFoundError(f"LongMemEval data not found at {data_file}")

        with open(data_file) as f:
            data = json.load(f)

        # Filter questions based on config
        questions = data.get("questions", [])
        filtered_questions = []

        for question in questions:
            if not self.config.context_groups or question.get("context_group") in self.config.context_groups:
                if not self.config.question_types or question.get("question_type") in self.config.question_types:
                    filtered_questions.append(question)

        # Apply max_questions limit if specified
        if self.config.max_questions and len(filtered_questions) > self.config.max_questions:
            console.print(
                f"[yellow]Limiting to {self.config.max_questions} questions (from {len(filtered_questions)} filtered)[/yellow]"
            )
            filtered_questions = filtered_questions[: self.config.max_questions]

        return filtered_questions

    def _generate_experimental_conditions(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate all experimental conditions to test."""
        conditions = []

        approaches = [
            "full_context",
            "basic_rag_no_rerank",
            "basic_rag_rerank",
            "enhanced_rag_no_rerank",
            "enhanced_rag_rerank",
        ]

        for question in questions:
            for model in self.config.models:
                for approach in approaches:
                    for iteration in range(self.config.iterations_per_question):
                        condition = {
                            "question_data": question,
                            "model": model,
                            "approach": approach,
                            "iteration": iteration,
                            "trial_id": str(uuid.uuid4()),
                        }
                        conditions.append(condition)

        return conditions

    def _run_single_trial(self, condition: dict[str, Any]) -> ExperimentResult:
        """
        Run a single experimental trial.

        Args:
            condition: Dictionary with trial configuration

        Returns:
            ExperimentResult with all measurements and evaluations
        """
        question_data = condition["question_data"]
        approach = condition["approach"]
        model = condition["model"]
        trial_id = condition["trial_id"]

        question = question_data["question"]
        expected_answer = question_data["expected_answer"]

        # Execute the appropriate approach
        retrieval_quality_result = None
        if approach == "full_context":
            model_answer, context_measurement, approach_metadata = self._run_full_context_approach(
                question, question_data, model
            )
            # Full context doesn't involve retrieval, so no retrieval quality metrics
        elif approach == "basic_rag_no_rerank":
            model_answer, context_measurement, approach_metadata, retrieval_quality_result = (
                self._run_basic_rag_approach(question, question_data, model, use_reranking=False)
            )
        elif approach == "basic_rag_rerank":
            model_answer, context_measurement, approach_metadata, retrieval_quality_result = (
                self._run_basic_rag_approach(question, question_data, model, use_reranking=True)
            )
        elif approach == "enhanced_rag_no_rerank":
            model_answer, context_measurement, approach_metadata, retrieval_quality_result = (
                self._run_enhanced_rag_approach(question, question_data, model, use_reranking=False)
            )
        elif approach == "enhanced_rag_rerank":
            model_answer, context_measurement, approach_metadata, retrieval_quality_result = (
                self._run_enhanced_rag_approach(question, question_data, model, use_reranking=True)
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

        # Evaluate with LLM judge
        judge_evaluation = self.judge_evaluator.evaluate_answer(
            question=question,
            expected_answer=expected_answer,
            model_answer=model_answer,
            context_preview=approach_metadata.get("context_preview", ""),
            approach_used=approach,
        )

        # Add retrieval quality result to analyzer (if available)
        if retrieval_quality_result:
            self.retrieval_quality_analyzer.add_result(retrieval_quality_result)

        # Create result
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            trial_id=trial_id,
            approach=approach,
            model=model,
            question_id=question_data["id"],
            question=question,
            question_type=question_data["question_type"],
            context_group=question_data["context_group"],
            expected_answer=expected_answer,
            model_answer=model_answer,
            judge_evaluation=judge_evaluation,
            context_measurement=context_measurement,
            approach_metadata=approach_metadata,
            retrieval_quality=retrieval_quality_result,
            timestamp=time.time(),
        )

        # Trial result is now tracked by the progress bar
        return result

    def _run_full_context_approach(
        self, question: str, question_data: dict[str, Any], model: str
    ) -> tuple[str, ContextMeasurement, dict[str, Any]]:
        """Run full context approach - load complete documents."""
        # Load full document context (placeholder - would load actual documents)
        full_context = self._load_full_document_context(question_data)

        # Measure context
        context_measurement = self.context_tracker.measure_full_context_approach(
            query=question,
            full_document_context=full_context,
            system_prompt="Answer the question based on the provided context.",
        )

        # Generate answer using full context
        complete_prompt = f"Context:\n{full_context}\n\nQuestion: {question}\n\nAnswer:"

        result = self.model_interface.query_model(model_name=model, prompt=complete_prompt, max_tokens=500)

        model_answer = result.response if result.success else "Error: Failed to generate response"

        approach_metadata = {
            "full_context_length": len(full_context),
            "context_preview": full_context[:500],
            "model_result": asdict(result) if hasattr(result, "__dict__") else str(result),
        }

        return model_answer, context_measurement, approach_metadata

    def _perform_triple_retrieval_with_hypothetical_answer(self, question: str):
        """
        Perform triple retrieval using OpenAI embeddings with:
        1. Original query
        2. AI-rewritten query
        3. Hypothetical answer (HyDE approach)

        This custom implementation bypasses the pipeline's SentenceTransformer model
        to use OpenAI embeddings that match our ChromaDB collection dimensions.
        """
        from context_is_king.rag_pipeline.query_enhancement.dual_retrieval import DualRetrievalResult

        start_time = time.time()

        # Step 1: Generate rewritten query
        rewrite_result = self.query_rewriter.rewrite_query(question)
        rewritten_query = rewrite_result.rewritten_query if rewrite_result.success else question

        # Step 2: Generate hypothetical answer
        hypothetical_answer = self._generate_hypothetical_answer(question)
        console.print(f"[dim]Generated hypothetical answer: {hypothetical_answer[:100]}...[/dim]")

        # Step 3: Retrieve using original query
        original_results = self._query_longmemeval_collection(question, k=self.config.retrieval_k)
        console.print(f"[dim]Original query retrieved {len(original_results)} results[/dim]")

        # Step 4: Retrieve using rewritten query (if different)
        if rewritten_query != question:
            rewritten_results = self._query_longmemeval_collection(rewritten_query, k=self.config.retrieval_k)
            console.print(f"[dim]Rewritten query retrieved {len(rewritten_results)} results[/dim]")
        else:
            rewritten_results = []
            console.print("[dim]Rewritten query identical to original, skipping duplicate retrieval[/dim]")

        # Step 5: Retrieve using hypothetical answer
        hypothetical_results = self._query_longmemeval_collection(hypothetical_answer, k=self.config.retrieval_k)
        console.print(f"[dim]Hypothetical answer retrieved {len(hypothetical_results)} results[/dim]")

        # Step 6: Combine all three result sets using RRF (Reciprocal Rank Fusion)
        combined_results = self._fuse_triple_retrieval_results(
            original_results, rewritten_results, hypothetical_results, method="rrf"
        )

        # Step 7: Create result object (extend DualRetrievalResult for compatibility)
        total_time_ms = (time.time() - start_time) * 1000

        dual_result = DualRetrievalResult(
            original_query=question,
            rewritten_query=rewritten_query,
            original_results=original_results,
            rewritten_results=rewritten_results,
            combined_results=combined_results,
            fusion_metadata={
                "method": "custom_openai_triple_rrf",
                "original_count": len(original_results),
                "rewritten_count": len(rewritten_results),
                "hypothetical_count": len(hypothetical_results),
                "combined_count": len(combined_results),
                "query_rewrite_success": rewritten_query != question,
                "hypothetical_answer": hypothetical_answer[:200] + "..."
                if len(hypothetical_answer) > 200
                else hypothetical_answer,
            },
            total_time_ms=total_time_ms,
        )

        return dual_result

    def _generate_hypothetical_answer(self, question: str) -> str:
        """
        Generate a hypothetical answer to the question without any context.
        This answer will be used for embedding-based retrieval (HyDE approach).
        """
        from textwrap import dedent

        hyde_prompt = dedent(f"""Provide a response that could be a good answer to the posed question. In this case, it's ok to not know if everything is correct. Do not mention that it depends, plainly give something that could be considered a good answer to the question.

Question: {question}

Answer:""")

        try:
            # Use a fast, cost-effective model for hypothetical answer generation
            result = self.model_interface.query_model(
                model_name="gpt-4o-mini",  # Fast and cost-effective
                prompt=hyde_prompt,
                max_tokens=150,  # Keep hypothetical answers concise
            )

            if result.success and result.response:
                hypothetical_answer = result.response.strip()
                return hypothetical_answer
            else:
                console.print("[yellow]Failed to generate hypothetical answer, using original question[/yellow]")
                return question

        except Exception as e:
            console.print(f"[red]Error generating hypothetical answer: {e}[/red]")
            return question

    def _fuse_triple_retrieval_results(self, original_results, rewritten_results, hypothetical_results, method="rrf"):
        """Combine results from three retrieval approaches: original query, rewritten query, and hypothetical answer."""

        if method == "rrf":
            # Reciprocal Rank Fusion for three result sets
            k = 60  # RRF parameter
            result_scores = {}

            # Score original results
            for rank, result in enumerate(original_results, 1):
                chunk_id = result.chunk.chunk_id
                if chunk_id not in result_scores:
                    result_scores[chunk_id] = {"result": result, "score": 0.0}
                result_scores[chunk_id]["score"] += 1.0 / (k + rank)

            # Score rewritten results
            for rank, result in enumerate(rewritten_results, 1):
                chunk_id = result.chunk.chunk_id
                if chunk_id not in result_scores:
                    result_scores[chunk_id] = {"result": result, "score": 0.0}
                result_scores[chunk_id]["score"] += 1.0 / (k + rank)

            # Score hypothetical answer results
            for rank, result in enumerate(hypothetical_results, 1):
                chunk_id = result.chunk.chunk_id
                if chunk_id not in result_scores:
                    result_scores[chunk_id] = {"result": result, "score": 0.0}
                result_scores[chunk_id]["score"] += 1.0 / (k + rank)

            # Sort by combined score and return top k
            sorted_results = sorted(result_scores.values(), key=lambda x: x["score"], reverse=True)
            return [item["result"] for item in sorted_results[: self.config.retrieval_k]]

        elif method == "concat":
            # Simple concatenation with deduplication across three sets
            seen_ids = set()
            combined = []

            # Add original results first
            for result in original_results:
                if result.chunk.chunk_id not in seen_ids:
                    combined.append(result)
                    seen_ids.add(result.chunk.chunk_id)

            # Add rewritten results
            for result in rewritten_results:
                if result.chunk.chunk_id not in seen_ids and len(combined) < self.config.retrieval_k:
                    combined.append(result)
                    seen_ids.add(result.chunk.chunk_id)

            # Add hypothetical results
            for result in hypothetical_results:
                if result.chunk.chunk_id not in seen_ids and len(combined) < self.config.retrieval_k:
                    combined.append(result)
                    seen_ids.add(result.chunk.chunk_id)

            return combined[: self.config.retrieval_k]

        else:
            # Default to original results if method unknown
            return original_results

    def _fuse_retrieval_results(self, original_results, rewritten_results, method="rrf"):
        """Combine results from original and rewritten queries using specified fusion method."""

        if method == "rrf":
            # Reciprocal Rank Fusion
            k = 60  # RRF parameter
            result_scores = {}

            # Score original results
            for rank, result in enumerate(original_results, 1):
                chunk_id = result.chunk.chunk_id
                if chunk_id not in result_scores:
                    result_scores[chunk_id] = {"result": result, "score": 0.0}
                result_scores[chunk_id]["score"] += 1.0 / (k + rank)

            # Score rewritten results
            for rank, result in enumerate(rewritten_results, 1):
                chunk_id = result.chunk.chunk_id
                if chunk_id not in result_scores:
                    result_scores[chunk_id] = {"result": result, "score": 0.0}
                result_scores[chunk_id]["score"] += 1.0 / (k + rank)

            # Sort by combined score and return top k
            sorted_results = sorted(result_scores.values(), key=lambda x: x["score"], reverse=True)
            return [item["result"] for item in sorted_results[: self.config.retrieval_k]]

        elif method == "concat":
            # Simple concatenation with deduplication
            seen_ids = set()
            combined = []

            # Add original results first
            for result in original_results:
                if result.chunk.chunk_id not in seen_ids:
                    combined.append(result)
                    seen_ids.add(result.chunk.chunk_id)

            # Add rewritten results that aren't already included
            for result in rewritten_results:
                if result.chunk.chunk_id not in seen_ids and len(combined) < self.config.retrieval_k:
                    combined.append(result)
                    seen_ids.add(result.chunk.chunk_id)

            return combined[: self.config.retrieval_k]

        else:
            # Default to original results if method unknown
            return original_results

    def _run_enhanced_rag_approach(
        self, question: str, question_data: dict[str, Any], model: str, use_reranking: bool
    ) -> tuple[str, ContextMeasurement, dict[str, Any], RetrievalQualityResult | None]:
        """Run enhanced RAG approach with dual retrieval (query rewrite + hypothetical answer)."""

        # Initialize dual_result for metadata
        dual_result = None

        # Use enhanced triple retrieval with OpenAI embeddings (query + rewrite + hypothetical answer)
        if self.chroma_available and self.query_rewriter:
            console.print(
                "[cyan]Enhanced RAG using triple retrieval: query rewrite + hypothetical answer + original[/cyan]"
            )

            try:
                # Perform triple retrieval using our custom implementation
                dual_result = self._perform_triple_retrieval_with_hypothetical_answer(question)
                retrieved_results = dual_result.combined_results
                console.print(f"[green]Retrieved {len(retrieved_results)} chunks via enhanced triple retrieval[/green]")
                console.print(f"[dim]Original query: {dual_result.original_query}[/dim]")
                console.print(f"[dim]Rewritten query: {dual_result.rewritten_query}[/dim]")

            except Exception as retrieval_error:
                console.print(f"[red]Enhanced triple retrieval failed: {retrieval_error}[/red]")
                console.print("[yellow]Falling back to basic retrieval[/yellow]")
                # Fallback to basic retrieval
                retrieved_results = self._query_longmemeval_collection(question, k=self.config.retrieval_k)

        elif self.chroma_available:
            # Fallback to basic retrieval if query rewriter not available
            console.print("[yellow]Query rewriter not available, using basic retrieval[/yellow]")
            retrieved_results = self._query_longmemeval_collection(question, k=self.config.retrieval_k)
        else:
            # Final fallback to mock retrieval for testing
            console.print("[yellow]Using mock retrieval (components not available)[/yellow]")
            return self._mock_rag_approach(question, question_data, model, use_reranking)

        # Apply reranking if requested
        if use_reranking and retrieved_results:
            retrieved_results = self.reranker.rerank_results(question, retrieved_results)

        # Assemble context
        context = self.context_assembly.assemble_rag_context(retrieved_results, question)

        # Extract text chunks for context measurement
        retrieved_chunks = [result.chunk.content for result in retrieved_results]

        # Measure context
        context_measurement = self.context_tracker.measure_rag_approach(
            query=question,
            retrieved_chunks=retrieved_chunks,
            system_prompt="Answer the question based on the retrieved context.",
            approach_name=f"enhanced_rag_{'rerank' if use_reranking else 'no_rerank'}",
            retrieval_metadata={
                "triple_retrieval": dual_result.fusion_metadata if dual_result else {"method": "direct_chromadb"},
                "reranking_used": use_reranking,
                "retrieval_k": self.config.retrieval_k,
            },
        )

        # Generate answer
        complete_prompt = f"Context:\n{context.context}\n\nQuestion: {question}\n\nAnswer:"

        result = self.model_interface.query_model(model_name=model, prompt=complete_prompt, max_tokens=500)

        model_answer = result.response if result.success else "Error: Failed to generate response"

        approach_metadata = {
            "triple_retrieval_metadata": dual_result.fusion_metadata if dual_result else {"method": "direct_chromadb"},
            "reranking_used": use_reranking,
            "retrieved_chunks_count": len(retrieved_chunks),
            "context_preview": context.context[:500] if hasattr(context, "context") else str(context)[:500],
            "query_rewrite_success": dual_result.rewritten_query != dual_result.original_query
            if dual_result
            else False,
            "hypothetical_answer_used": bool(dual_result and "hypothetical_answer" in dual_result.fusion_metadata),
            "model_result": asdict(result) if hasattr(result, "__dict__") else str(result),
        }

        # Evaluate retrieval quality
        retrieval_quality_result = self._evaluate_retrieval_quality(
            question=question,
            expected_answer=question_data["expected_answer"],
            retrieved_results=retrieved_results,
            approach=f"enhanced_rag_{'rerank' if use_reranking else 'no_rerank'}",
            question_id=question_data["id"],
            question_data=question_data,
        )

        return model_answer, context_measurement, approach_metadata, retrieval_quality_result

    def _run_basic_rag_approach(
        self, question: str, question_data: dict[str, Any], model: str, use_reranking: bool
    ) -> tuple[str, ContextMeasurement, dict[str, Any], RetrievalQualityResult | None]:
        """Run basic RAG approach with simple retrieval (no query enhancement)."""

        # Check if ChromaDB connection is available for direct query
        if self.chroma_available:
            # Use direct ChromaDB query with OpenAI embeddings (azure/text-embedding-3-small)
            retrieved_results = self._query_longmemeval_collection(question, k=self.config.retrieval_k)
            console.print(
                f"[green]Retrieved {len(retrieved_results)} chunks via direct ChromaDB query with Azure OpenAI embeddings[/green]"
            )
        elif self.retrieval_engine:
            # Fallback to regular retrieval engine
            collection = self._get_document_collection(question_data)
            retrieved_results = self.retrieval_engine.retrieve_chunks(
                query=question, collection=collection, k=self.config.retrieval_k
            )
        else:
            # Final fallback to mock retrieval
            console.print("[yellow]Using mock basic RAG (components not available)[/yellow]")
            return self._mock_basic_rag_approach(question, question_data, model, use_reranking)

        # Apply reranking if requested
        if use_reranking and retrieved_results:
            retrieved_results = self.reranker.rerank_results(question, retrieved_results)

        # Assemble context
        context = self.context_assembly.assemble_rag_context(retrieved_results, question)

        # Extract text chunks for context measurement
        retrieved_chunks = [result.chunk.content for result in retrieved_results]

        # Measure context
        context_measurement = self.context_tracker.measure_rag_approach(
            query=question,
            retrieved_chunks=retrieved_chunks,
            system_prompt="Answer the question based on the retrieved context.",
            approach_name=f"basic_rag_{'rerank' if use_reranking else 'no_rerank'}",
            retrieval_metadata={
                "basic_retrieval": True,
                "reranking_used": use_reranking,
                "retrieval_k": self.config.retrieval_k,
            },
        )

        # Generate answer
        complete_prompt = f"Context:\n{context.context}\n\nQuestion: {question}\n\nAnswer:"

        result = self.model_interface.query_model(model_name=model, prompt=complete_prompt, max_tokens=500)

        model_answer = result.response if result.success else "Error: Failed to generate response"

        approach_metadata = {
            "basic_retrieval": True,
            "reranking_used": use_reranking,
            "retrieved_chunks_count": len(retrieved_chunks),
            "context_preview": context.context[:500] if hasattr(context, "context") else str(context)[:500],
            "model_result": asdict(result) if hasattr(result, "__dict__") else str(result),
        }

        # Evaluate retrieval quality
        retrieval_quality_result = self._evaluate_retrieval_quality(
            question=question,
            expected_answer=question_data["expected_answer"],
            retrieved_results=retrieved_results,
            approach=f"basic_rag_{'rerank' if use_reranking else 'no_rerank'}",
            question_id=question_data["id"],
            question_data=question_data,
        )

        return model_answer, context_measurement, approach_metadata, retrieval_quality_result

    def _mock_basic_rag_approach(
        self, question: str, question_data: dict[str, Any], model: str, use_reranking: bool
    ) -> tuple[str, ContextMeasurement, dict[str, Any], RetrievalQualityResult | None]:
        """Mock basic RAG approach for testing when retrieval components are unavailable."""

        # Generate mock retrieved chunks based on context group (simpler than enhanced)
        context_group = question_data.get("context_group", "short")

        if context_group == "short":
            chunk_count = 2  # Fewer chunks than enhanced
            chunk_size = 150
        elif context_group == "medium":
            chunk_count = 3
            chunk_size = 200
        else:  # long
            chunk_count = 4
            chunk_size = 250

        mock_chunks = []
        for i in range(chunk_count):
            chunk_content = f"This is basic mock retrieved chunk {i + 1} for the question: '{question}'. " * chunk_size
            mock_chunks.append(chunk_content[: chunk_size * 3])  # Shorter than enhanced

        # Simulate context assembly
        context = "\n\n".join(mock_chunks)

        # Measure context
        context_measurement = self.context_tracker.measure_rag_approach(
            query=question,
            retrieved_chunks=mock_chunks,
            system_prompt="Answer the question based on the retrieved context.",
            approach_name=f"basic_rag_{'rerank' if use_reranking else 'no_rerank'}",
            retrieval_metadata={
                "mock_basic_retrieval": True,
                "reranking_used": use_reranking,
                "retrieval_k": self.config.retrieval_k,
            },
        )

        # Generate answer
        complete_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        result = self.model_interface.query_model(model_name=model, prompt=complete_prompt, max_tokens=500)

        model_answer = result.response if result.success else "Error: Failed to generate response"

        approach_metadata = {
            "mock_basic_retrieval": True,
            "reranking_used": use_reranking,
            "retrieved_chunks_count": len(mock_chunks),
            "context_preview": context[:500],
            "model_result": asdict(result) if hasattr(result, "__dict__") else str(result),
        }

        # No retrieval quality metrics for mock approaches
        return model_answer, context_measurement, approach_metadata, None

    def _mock_rag_approach(
        self, question: str, question_data: dict[str, Any], model: str, use_reranking: bool
    ) -> tuple[str, ContextMeasurement, dict[str, Any], RetrievalQualityResult | None]:
        """Mock RAG approach for testing when retrieval components are unavailable."""

        # Generate mock retrieved chunks based on context group
        context_group = question_data.get("context_group", "short")

        if context_group == "short":
            chunk_count = 3
            chunk_size = 200
        elif context_group == "medium":
            chunk_count = 5
            chunk_size = 300
        else:  # long
            chunk_count = 8
            chunk_size = 400

        mock_chunks = []
        for i in range(chunk_count):
            chunk_content = f"This is mock retrieved chunk {i + 1} related to the question: '{question}'. " * chunk_size
            mock_chunks.append(chunk_content[: chunk_size * 5])  # Truncate to reasonable size

        # Simulate context assembly
        context = "\n\n".join(mock_chunks)

        # Measure context
        context_measurement = self.context_tracker.measure_rag_approach(
            query=question,
            retrieved_chunks=mock_chunks,
            system_prompt="Answer the question based on the retrieved context.",
            approach_name=f"enhanced_rag_{'rerank' if use_reranking else 'no_rerank'}",
            retrieval_metadata={
                "mock_retrieval": True,
                "reranking_used": use_reranking,
                "retrieval_k": self.config.retrieval_k,
            },
        )

        # Generate answer
        complete_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        result = self.model_interface.query_model(model_name=model, prompt=complete_prompt, max_tokens=500)

        model_answer = result.response if result.success else "Error: Failed to generate response"

        approach_metadata = {
            "mock_retrieval": True,
            "reranking_used": use_reranking,
            "retrieved_chunks_count": len(mock_chunks),
            "context_preview": context[:500],
            "model_result": asdict(result) if hasattr(result, "__dict__") else str(result),
        }

        # No retrieval quality metrics for mock approaches
        return model_answer, context_measurement, approach_metadata, None

    def _load_full_document_context(self, question_data: dict[str, Any]) -> str:
        """Load full document context for a question from LongMemEval CSV data."""

        # Get the conversation ID from question data
        custom_id = question_data.get("id")
        if not custom_id:
            raise ValueError(f"No custom_id found in question data: {question_data.keys()}")

        console.print(f"[blue]Loading full conversation context for custom_id: {custom_id}[/blue]")

        # Validate that we have the full conversation data loaded
        if not hasattr(self, "longmemeval_full_data") or not self.longmemeval_full_data:
            raise RuntimeError("LongMemEval full conversation data not loaded. Cannot provide full context.")

        # Look up the full conversation in our loaded data
        if custom_id not in self.longmemeval_full_data:
            available_ids = list(self.longmemeval_full_data.keys())[:5]  # Show first 5 for debugging
            raise ValueError(
                f"No full context found for custom_id: '{custom_id}'. "
                f"Available IDs in CSV data: {available_ids}... (showing first 5 of {len(self.longmemeval_full_data)} total)"
            )

        # Get the full conversation from CSV data
        full_context = self.longmemeval_full_data[custom_id]
        console.print(f"[green]✓ Loaded full conversation context: {len(full_context):,} characters[/green]")

        return full_context

    def _get_document_collection(self, question_data: dict[str, Any]):
        """Get ChromaDB collection for a question."""
        collection_name = "longmemeval_conversations"

        # Use the working direct ChromaDB connection instead of embedding_manager
        if self.chroma_available and self.longmemeval_collection:
            try:
                # Log collection info for the first time
                if not hasattr(self, "_collection_stats_logged"):
                    collection_count = self.longmemeval_collection.count()
                    console.print(f"[green]Using ChromaDB collection: {collection_name}[/green]")
                    console.print(f"[dim]Documents: {collection_count:,} chunks[/dim]")
                    self._collection_stats_logged = True

                # Return the actual collection object, not the name
                return self.longmemeval_collection

            except Exception as e:
                console.print(f"[red]Error accessing collection '{collection_name}': {e}[/red]")
                return None
        else:
            console.print(
                f"[yellow]Direct ChromaDB connection not available (chroma_available={self.chroma_available})[/yellow]"
            )
            # Return None to trigger fallback behavior
            return None

    def _evaluate_retrieval_quality(
        self,
        question: str,
        expected_answer: str,
        retrieved_results: list[Any],
        approach: str,
        question_id: str,
        question_data: dict[str, Any] | None = None,
    ) -> RetrievalQualityResult | None:
        """Evaluate retrieval quality for retrieved results."""

        try:
            # Convert retrieved results to chunk dictionaries for evaluation
            retrieved_chunks = []
            retrieved_chunk_ids = []

            for i, result in enumerate(retrieved_results):
                # Handle different result types - this is flexible to work with various retrieval interfaces
                if hasattr(result, "chunk"):
                    # Standard retrieval result with chunk object
                    chunk_content = getattr(result.chunk, "content", str(result.chunk))
                    chunk_id = getattr(result.chunk, "chunk_id", f"chunk_{i}")
                elif hasattr(result, "content"):
                    # Direct content result
                    chunk_content = result.content
                    chunk_id = getattr(result, "id", f"chunk_{i}")
                elif isinstance(result, dict):
                    # Dictionary result
                    chunk_content = result.get("content", result.get("text", str(result)))
                    chunk_id = result.get("id", f"chunk_{i}")
                else:
                    # Fallback - treat as string content
                    chunk_content = str(result)
                    chunk_id = f"chunk_{i}"

                retrieved_chunks.append({"id": chunk_id, "content": chunk_content})
                retrieved_chunk_ids.append(chunk_id)

            # Use pre-computed ground truth if available
            if question_data and "relevant_chunks" in question_data:
                ground_truth_chunks = question_data["relevant_chunks"]

                # Extract relevant chunk IDs from ground truth
                relevant_chunk_ids = []
                relevance_scores = []

                for gt_chunk in ground_truth_chunks:
                    if gt_chunk.get("is_relevant", False):
                        relevant_chunk_ids.append(gt_chunk["chunk_id"])
                        relevance_scores.append(gt_chunk.get("primary_score", 1.0))

                # Calculate retrieval quality metrics directly
                retrieval_quality_result = self._calculate_retrieval_metrics(
                    question_id=question_id,
                    approach=approach,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    ground_truth_chunk_ids=relevant_chunk_ids,
                    relevance_scores=relevance_scores,
                )

                console.print(
                    f"[green]Used pre-computed ground truth: {len(relevant_chunk_ids)} relevant chunks[/green]"
                )
                return retrieval_quality_result

            # Fallback to standard evaluator if no ground truth available
            console.print("[yellow]No pre-computed ground truth available, using semantic similarity[/yellow]")
            return self.retrieval_quality_evaluator.evaluate_retrieval(
                question=question,
                expected_answer=expected_answer,
                retrieved_chunks=retrieved_chunks,
                approach=approach,
                question_id=question_id,
            )

        except Exception as e:
            console.print(f"[yellow]Warning: Could not evaluate retrieval quality: {e}[/yellow]")
            return None

    def _calculate_retrieval_metrics(
        self,
        question_id: str,
        approach: str,
        retrieved_chunk_ids: list[str],
        ground_truth_chunk_ids: list[str],
        relevance_scores: list[float],
    ) -> RetrievalQualityResult:
        """Calculate retrieval quality metrics using pre-computed ground truth."""

        # Convert to sets for easier intersection calculation
        retrieved_set = set(retrieved_chunk_ids)
        relevant_set = set(ground_truth_chunk_ids)

        # Calculate intersection
        relevant_retrieved = retrieved_set & relevant_set

        # Recall at different K values (normalized by total relevant chunks)
        total_relevant = len(relevant_set)
        if total_relevant == 0:
            # No ground truth relevant chunks - perfect recall is undefined, use 0
            recall_at_1 = recall_at_3 = recall_at_5 = recall_at_10 = 0.0
        else:
            recall_at_1 = len(relevant_retrieved & set(retrieved_chunk_ids[:1])) / total_relevant
            recall_at_3 = len(relevant_retrieved & set(retrieved_chunk_ids[:3])) / total_relevant
            recall_at_5 = len(relevant_retrieved & set(retrieved_chunk_ids[:5])) / total_relevant
            recall_at_10 = len(relevant_retrieved & set(retrieved_chunk_ids[:10])) / total_relevant

        # Precision at 5 (normalized by retrieved chunks at K=5)
        retrieved_at_5 = min(5, len(retrieved_chunk_ids))
        if retrieved_at_5 == 0:
            # No chunks retrieved - precision is undefined, use 0
            precision_at_5 = 0.0
        else:
            precision_at_5 = len(relevant_retrieved & set(retrieved_chunk_ids[:5])) / retrieved_at_5

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, chunk_id in enumerate(retrieved_chunk_ids):
            if chunk_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        # NDCG calculation (simplified)
        def dcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
            dcg = 0.0
            for i, chunk_id in enumerate(retrieved_ids[:k]):
                if chunk_id in relevant_ids:
                    # Binary relevance: relevant=1, not relevant=0
                    relevance = 1.0
                    dcg += relevance / max(1.0, math.log2(i + 2))  # +2 because log2(1) = 0
            return dcg

        def ideal_dcg_at_k(relevant_ids: list[str], k: int) -> float:
            # For binary relevance, ideal DCG is sum of 1/log2(i+2) for min(k, num_relevant) items
            num_relevant = min(k, len(relevant_ids))
            if num_relevant == 0:
                return 0.0
            idcg = sum(1.0 / max(1.0, math.log2(i + 2)) for i in range(num_relevant))
            return idcg

        # Import math for log2
        import math

        # Calculate NDCG with proper normalization and edge case handling
        if len(retrieved_chunk_ids) == 0 or total_relevant == 0:
            # No retrieval results or no relevant chunks - NDCG is 0
            ndcg_at_3 = ndcg_at_5 = ndcg_at_10 = 0.0
        else:
            dcg_3 = dcg_at_k(retrieved_chunk_ids, ground_truth_chunk_ids, 3)
            idcg_3 = ideal_dcg_at_k(ground_truth_chunk_ids, 3)
            ndcg_at_3 = dcg_3 / idcg_3 if idcg_3 > 0 else 0.0

            dcg_5 = dcg_at_k(retrieved_chunk_ids, ground_truth_chunk_ids, 5)
            idcg_5 = ideal_dcg_at_k(ground_truth_chunk_ids, 5)
            ndcg_at_5 = dcg_5 / idcg_5 if idcg_5 > 0 else 0.0

            dcg_10 = dcg_at_k(retrieved_chunk_ids, ground_truth_chunk_ids, 10)
            idcg_10 = ideal_dcg_at_k(ground_truth_chunk_ids, 10)
            ndcg_at_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0

        # Create and return RetrievalQualityResult
        return RetrievalQualityResult(
            question_id=question_id,
            approach=approach,
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg_at_3=ndcg_at_3,
            ndcg_at_5=ndcg_at_5,
            ndcg_at_10=ndcg_at_10,
            total_retrieved=len(retrieved_chunk_ids),
            relevant_retrieved=len(relevant_retrieved),
            total_relevant=len(relevant_set),
            precision_at_5=precision_at_5,
            ground_truth_chunks=ground_truth_chunk_ids,
            retrieved_chunks=retrieved_chunk_ids,
            relevance_scores=relevance_scores,
        )

    def _generate_experiment_summary(self) -> ExperimentSummary:
        """Generate comprehensive experiment summary and analysis."""
        # Analyze results by approach
        approach_comparison = self._analyze_approach_performance()

        # Analyze context measurements
        context_analysis = self._analyze_context_patterns()

        # Calculate cost analysis
        cost_analysis = self._analyze_cost_efficiency()

        # Generate retrieval quality summary
        retrieval_quality_summary = self.retrieval_quality_analyzer.generate_summary_report()

        summary = ExperimentSummary(
            experiment_id=self.experiment_id,
            config=self.config,
            results=self.results,
            approach_comparison=approach_comparison,
            context_analysis=context_analysis,
            cost_analysis=cost_analysis,
            metadata={
                "total_trials": len(self.results),
                "completion_time": time.time(),
                "interrupted": self.interrupted,
                "retrieval_quality_summary": retrieval_quality_summary,
            },
        )

        return summary

    def _analyze_approach_performance(self) -> dict[str, Any]:
        """Analyze performance differences between approaches."""
        if not self.results:
            return {"error": "No results to analyze"}

        # Group results by approach
        by_approach = {}
        for result in self.results:
            approach = result.approach
            if approach not in by_approach:
                by_approach[approach] = []
            by_approach[approach].append(result)

        # Calculate statistics for each approach
        approach_stats = {}
        for approach, results in by_approach.items():
            correct_count = sum(1 for r in results if r.judge_evaluation.is_correct)
            context_correct_count = sum(1 for r in results if r.judge_evaluation.is_correct_given_context)
            total_count = len(results)

            confidences = [r.judge_evaluation.confidence for r in results]
            context_confidences = [r.judge_evaluation.context_grounded_confidence for r in results]
            retrieval_gaps = [r.judge_evaluation.retrieval_gap for r in results]
            context_sizes = [r.context_measurement.total_tokens for r in results]

            approach_stats[approach] = {
                "total_trials": total_count,
                "correct_trials": correct_count,
                "accuracy": correct_count / total_count if total_count > 0 else 0,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                # New dual correctness metrics
                "context_correct_trials": context_correct_count,
                "context_accuracy": context_correct_count / total_count if total_count > 0 else 0,
                "avg_context_confidence": sum(context_confidences) / len(context_confidences)
                if context_confidences
                else 0,
                "avg_retrieval_gap": sum(retrieval_gaps) / len(retrieval_gaps) if retrieval_gaps else 0,
                # Context size metrics
                "avg_context_tokens": sum(context_sizes) / len(context_sizes) if context_sizes else 0,
                "min_context_tokens": min(context_sizes) if context_sizes else 0,
                "max_context_tokens": max(context_sizes) if context_sizes else 0,
            }

        return {
            "approaches_tested": list(approach_stats.keys()),
            "approach_statistics": approach_stats,
            "best_accuracy": max(approach_stats.items(), key=lambda x: x[1]["accuracy"])[0] if approach_stats else None,
            "most_efficient": min(approach_stats.items(), key=lambda x: x[1]["avg_context_tokens"])[0]
            if approach_stats
            else None,
        }

    def _analyze_context_patterns(self) -> dict[str, Any]:
        """Analyze context usage patterns across approaches."""
        context_measurements = [r.context_measurement for r in self.results]

        if not context_measurements:
            return {"error": "No context measurements to analyze"}

        return self.context_tracker.compare_measurements(context_measurements)

    def _analyze_cost_efficiency(self) -> dict[str, Any]:
        """Analyze cost efficiency of different approaches."""
        # Calculate cost per correct answer for each approach
        by_approach = {}
        for result in self.results:
            approach = result.approach
            if approach not in by_approach:
                by_approach[approach] = {
                    "total_tokens": 0,
                    "correct_answers": 0,
                    "context_correct_answers": 0,
                    "total_trials": 0,
                }

            by_approach[approach]["total_tokens"] += result.context_measurement.total_tokens
            by_approach[approach]["total_trials"] += 1
            if result.judge_evaluation.is_correct:
                by_approach[approach]["correct_answers"] += 1
            if result.judge_evaluation.is_correct_given_context:
                by_approach[approach]["context_correct_answers"] += 1

        cost_efficiency = {}
        for approach, stats in by_approach.items():
            if stats["correct_answers"] > 0:
                cost_per_correct = stats["total_tokens"] / stats["correct_answers"]
            else:
                cost_per_correct = float("inf")

            if stats["context_correct_answers"] > 0:
                cost_per_context_correct = stats["total_tokens"] / stats["context_correct_answers"]
            else:
                cost_per_context_correct = float("inf")

            cost_efficiency[approach] = {
                "tokens_per_correct_answer": cost_per_correct,
                "tokens_per_context_correct_answer": cost_per_context_correct,
                "avg_tokens_per_trial": stats["total_tokens"] / stats["total_trials"]
                if stats["total_trials"] > 0
                else 0,
                "accuracy": stats["correct_answers"] / stats["total_trials"] if stats["total_trials"] > 0 else 0,
                "context_accuracy": stats["context_correct_answers"] / stats["total_trials"]
                if stats["total_trials"] > 0
                else 0,
            }

        return {
            "cost_efficiency_by_approach": cost_efficiency,
            "most_cost_efficient": min(cost_efficiency.items(), key=lambda x: x[1]["tokens_per_correct_answer"])[0]
            if cost_efficiency
            else None,
        }

    def _save_checkpoint(self):
        """Save experiment checkpoint."""
        checkpoint_file = self.output_dir / f"{self.experiment_id}_checkpoint.json"

        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "config": asdict(self.config),
            "results_count": len(self.results),
            "results": [asdict(r) for r in self.results],
            "timestamp": time.time(),
            "interrupted": self.interrupted,
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        console.print(f"[blue]Checkpoint saved: {checkpoint_file}[/blue]")

    def _save_final_results(self, summary: ExperimentSummary):
        """Save final experiment results."""
        results_file = self.output_dir / f"{self.experiment_id}_results.json"

        with open(results_file, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        # Save detailed results
        detailed_file = self.output_dir / f"{self.experiment_id}_detailed.json"
        detailed_data = {"experiment_id": self.experiment_id, "results": [asdict(r) for r in self.results]}

        with open(detailed_file, "w") as f:
            json.dump(detailed_data, f, indent=2, default=str)

        # Generate summary report
        summary_file = self._generate_summary_report(summary)

        # Print file summary at the end
        self._print_results_summary(results_file, detailed_file, summary_file)

    def _generate_summary_report(self, summary: ExperimentSummary):
        """Generate human-readable summary report."""
        report_file = self.output_dir / f"{self.experiment_id}_summary.md"

        with open(report_file, "w") as f:
            f.write("# Reranking Value Experiment Report\n\n")
            f.write(f"**Experiment ID**: {self.experiment_id}\n")
            f.write(f"**Total Trials**: {len(self.results)}\n\n")

            f.write("## Approach Performance\n\n")

            approach_stats = summary.approach_comparison.get("approach_statistics", {})
            for approach, stats in approach_stats.items():
                f.write(f"### {approach.replace('_', ' ').title()}\n")
                f.write(f"- **Absolute Accuracy**: {stats['accuracy']:.2%}\n")
                f.write(f"- **Context-Grounded Accuracy**: {stats.get('context_accuracy', 0):.2%}\n")
                f.write(f"- **Retrieval Gap**: {stats.get('avg_retrieval_gap', 0):+.3f}\n")
                f.write(f"- Average Confidence: {stats['avg_confidence']:.3f}\n")
                f.write(f"- Average Context Confidence: {stats.get('avg_context_confidence', 0):.3f}\n")
                f.write(f"- Average Context Tokens: {stats['avg_context_tokens']:.0f}\n\n")

            f.write("## Key Findings\n\n")
            best_accuracy = summary.approach_comparison.get("best_accuracy", "Unknown")
            most_efficient = summary.approach_comparison.get("most_efficient", "Unknown")

            f.write(f"- **Best Accuracy**: {best_accuracy}\n")
            f.write(f"- **Most Context Efficient**: {most_efficient}\n\n")

            f.write("## Cost Analysis\n\n")
            cost_stats = summary.cost_analysis.get("cost_efficiency_by_approach", {})
            for approach, costs in cost_stats.items():
                f.write(
                    f"- **{approach}**: {costs['tokens_per_correct_answer']:.0f} tokens per absolute correct, {costs.get('tokens_per_context_correct_answer', float('inf')):.0f} tokens per context correct\n"
                )

            # Add retrieval quality analysis if available
            retrieval_quality = summary.metadata.get("retrieval_quality_summary", {})
            if retrieval_quality and retrieval_quality.get("total_questions_evaluated", 0) > 0:
                f.write("\n## Retrieval Quality Analysis\n\n")

                approach_metrics = retrieval_quality.get("approach_metrics", {})
                if approach_metrics:
                    f.write("### Average Retrieval Metrics by Approach\n\n")
                    for approach, metrics in approach_metrics.items():
                        f.write(f"#### {approach.replace('_', ' ').title()}\n")
                        f.write(f"- Recall@5: {metrics.get('avg_recall_at_5', 0):.3f}\n")
                        f.write(f"- MRR: {metrics.get('avg_mrr', 0):.3f}\n")
                        f.write(f"- NDCG@5: {metrics.get('avg_ndcg_at_5', 0):.3f}\n")
                        f.write(f"- Precision@5: {metrics.get('avg_precision_at_5', 0):.3f}\n\n")

                best_approaches = retrieval_quality.get("best_approaches", {})
                if best_approaches:
                    f.write("### Best Retrieval Performance\n\n")
                    for metric, approach in best_approaches.items():
                        metric_name = metric.replace("avg_", "").replace("_", "@").title()
                        f.write(f"- **{metric_name}**: {approach}\n")

                f.write(
                    f"\n**Questions with Retrieval Quality Assessment**: {retrieval_quality['total_questions_evaluated']}\n"
                )

        return report_file

    def _print_results_summary(self, results_file, detailed_file, summary_file):
        """Print a summary of all generated files at the end of the experiment."""
        from rich.panel import Panel
        from rich.text import Text

        # Create file listing
        files_text = Text()
        files_text.append("📁 Results Files Generated:\n\n", style="bold green")

        files_text.append("1. ", style="dim")
        files_text.append("Summary JSON", style="bold blue")
        files_text.append(f": {results_file}\n", style="dim")
        files_text.append("   └─ Experiment summary with dual correctness metrics\n\n", style="dim")

        files_text.append("2. ", style="dim")
        files_text.append("Detailed JSON", style="bold blue")
        files_text.append(f": {detailed_file}\n", style="dim")
        files_text.append("   └─ Complete trial data with all evaluations\n\n", style="dim")

        files_text.append("3. ", style="dim")
        files_text.append("Summary Report", style="bold blue")
        files_text.append(f": {summary_file}\n", style="dim")
        files_text.append("   └─ Human-readable markdown report with dual correctness analysis\n\n", style="dim")

        # Add dual correctness info
        files_text.append("✨ ", style="yellow")
        files_text.append("Enhanced with Dual Correctness Assessment", style="bold yellow")
        files_text.append(":\n", style="yellow")
        files_text.append("   • Absolute Accuracy: Matches ground truth\n", style="dim")
        files_text.append("   • Context-Grounded Accuracy: Reasonable given context\n", style="dim")
        files_text.append("   • Retrieval Gap: Measures retrieval quality impact\n", style="dim")

        # Create panel
        panel = Panel(
            files_text, title="🎯 Experiment Complete", title_align="left", border_style="green", padding=(1, 2)
        )

        console.print("\n")
        console.print(panel)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        console.print(f"\n[yellow]Received signal {signum}, gracefully shutting down...[/yellow]")
        self.interrupted = True

        if self.results:
            console.print("[blue]Saving checkpoint before exit...[/blue]")
            self._save_checkpoint()

        console.print("[yellow]Experiment interrupted. Use resume functionality to continue.[/yellow]")
        sys.exit(0)


def main():
    """Example usage of the reranking experiment."""
    # Load environment variables from root .env file
    root_dir = Path(__file__).parent.parent.parent
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        console.print(f"[green]✓ Environment loaded from: {env_file}[/green]")

    # Check for required API key
    if not os.getenv("ORQ_API_KEY"):
        console.print("[red]Error: ORQ_API_KEY not found in environment[/red]")
        console.print("[yellow]Please add ORQ_API_KEY to your .env file[/yellow]")
        sys.exit(1)

    config = ExperimentConfig(
        experiment_id=f"rerank_value_{int(time.time())}",
        output_dir="experiments/reranking_value/results",
        longmemeval_path="data/LongMemEval/",
        document_collections=["paul_graham", "arxiv"],
        models=["claude-sonnet-3.7"],
        context_groups=["short", "medium"],
        question_types=["factual", "reasoning"],
        retrieval_k=10,
        iterations_per_question=1,
    )

    experiment = RerankingValueExperiment(config)
    summary = experiment.run_experiment()

    console.print("[green]Experiment completed successfully![/green]")
    print(f"Results summary: {summary.metadata}")


if __name__ == "__main__":
    main()
