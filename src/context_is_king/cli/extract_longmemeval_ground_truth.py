#!/usr/bin/env python3
"""
LongMemEval Ground Truth Extraction Script

🎯 DATASET: LongMemEval Conversations
📊 PURPOSE: Generate ground truth chunk relevance for retrieval evaluation

This script is SPECIFICALLY designed for the LongMemEval dataset structure:
- Processes LongMemEval focused vs full conversation pairs
- Automatically strips prompt instructions (keeps only conversation data)
- Uses ChromaDB for efficient similarity search with OpenAI embeddings (768d)
- Applies Jaccard similarity and ROUGE-L metrics optimized for conversations
- Generates ground truth suitable for retrieval evaluation experiments

USAGE:
    # Basic extraction with default settings
    python scripts/longmemeval_ground_truth_extraction.py
    
    # Custom parameters for LongMemEval
    python scripts/longmemeval_ground_truth_extraction.py \
        --similarity-threshold 0.6 \
        --max-chunks 3 \
        --primary-metric rouge_l \
        --output data/longmemeval_ground_truth.json

INPUT FILES (LongMemEval format):
    - cleaned_longmemeval_s_focused.csv: Focused (relevant) conversation sections
    - cleaned_longmemeval_s_full.csv: Full conversations with distractors

OUTPUT:
    - JSON file with ground truth relevance annotations per question
    - ChromaDB collection for future similarity searches

REQUIREMENTS:
    - pandas, chromadb, tiktoken, rich
    - rouge-score (optional, for ROUGE-L metric)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
import dotenv
import chromadb

dotenv.load_dotenv(override=True)
# Import LongMemEval-specific components
try:
    from context_is_king.datasets.longmemeval import GroundTruthExtractor

    console = Console()
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you're running from the project root and the package is installed:")
    print("   pip install -e .")
    sys.exit(1)


def validate_longmemeval_files(focused_csv: str, full_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and load LongMemEval CSV files.

    Args:
        focused_csv: Path to focused conversations CSV
        full_csv: Path to full conversations CSV

    Returns:
        Tuple of (focused_df, full_df)

    Raises:
        SystemExit: If files are invalid or missing required columns
    """

    console.print("[blue]📋 Validating LongMemEval dataset files...[/blue]")

    # Check file existence
    for file_path, file_type in [(focused_csv, "focused"), (full_csv, "full")]:
        if not Path(file_path).exists():
            console.print(f"[red]❌ {file_type.title()} file not found: {file_path}[/red]")
            console.print(f"[yellow]💡 Expected LongMemEval {file_type} conversation file[/yellow]")
            sys.exit(1)

    # Load datasets
    try:
        focused_df = pd.read_csv(focused_csv)
        full_df = pd.read_csv(full_csv)
    except Exception as e:
        console.print(f"[red]❌ Error loading CSV files: {e}[/red]")
        sys.exit(1)

    # Validate LongMemEval structure
    focused_required = ["custom_id", "focused_prompt", "question"]
    full_required = ["custom_id", "full_prompt"]

    focused_missing = [col for col in focused_required if col not in focused_df.columns]
    full_missing = [col for col in full_required if col not in full_df.columns]

    if focused_missing:
        console.print(f"[red]❌ Focused CSV missing LongMemEval columns: {focused_missing}[/red]")
        console.print(f"[yellow]📋 Expected columns: {focused_required}[/yellow]")
        sys.exit(1)

    if full_missing:
        console.print(f"[red]❌ Full CSV missing LongMemEval columns: {full_missing}[/red]")
        console.print(f"[yellow]📋 Expected columns: {full_required}[/yellow]")
        sys.exit(1)

    # Verify matching conversations
    focused_ids = set(focused_df["custom_id"])
    full_ids = set(full_df["custom_id"])

    if not focused_ids.issubset(full_ids):
        missing_ids = focused_ids - full_ids
        console.print(f"[yellow]⚠️  {len(missing_ids)} focused conversations missing from full dataset[/yellow]")
        console.print(f"[dim]First few missing: {list(missing_ids)[:5]}[/dim]")

    console.print("[green]✅ LongMemEval validation passed[/green]")
    console.print(f"[dim]Focused conversations: {len(focused_df)}[/dim]")
    console.print(f"[dim]Full conversations: {len(full_df)}[/dim]")
    console.print(f"[dim]Matching conversations: {len(focused_ids & full_ids)}[/dim]")

    return focused_df, full_df


def main() -> None:
    """Main execution function for LongMemEval ground truth extraction."""

    parser = argparse.ArgumentParser(
        description="Extract ground truth chunk relevance from LongMemEval dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Basic extraction with optimized defaults
    python scripts/longmemeval_ground_truth_extraction.py
    
    # Custom similarity threshold for stricter relevance
    python scripts/longmemeval_ground_truth_extraction.py --similarity-threshold 0.7
    
    # Use Jaccard similarity instead of ROUGE-L (token presence only, no order) 
    python scripts/longmemeval_ground_truth_extraction.py --primary-metric jaccard
    
    # Custom embedding dimensions for different quality/cost trade-offs
    python scripts/longmemeval_ground_truth_extraction.py --embedding-dimensions 1536
    
    # Fewer chunks per question for focused evaluation
    python scripts/longmemeval_ground_truth_extraction.py --max-chunks 3

DATASET REQUIREMENTS:
    This script expects LongMemEval CSV files with specific column structure.
    See documentation for details on LongMemEval dataset format.
        """,
    )

    # File arguments - LongMemEval specific paths
    parser.add_argument(
        "--focused-csv",
        default="data/LongMemEval/cleaned_longmemeval_s_focused.csv",
        help="Path to LongMemEval focused conversations CSV (default: data/LongMemEval/cleaned_longmemeval_s_focused.csv)",
    )
    parser.add_argument(
        "--full-csv",
        default="data/LongMemEval/cleaned_longmemeval_s_full.csv",
        help="Path to LongMemEval full conversations CSV (default: data/LongMemEval/cleaned_longmemeval_s_full.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/longmemeval_ground_truth.json",
        help="Output path for ground truth JSON (default: data/longmemeval_ground_truth.json)",
    )

    # LongMemEval-optimized parameters
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Similarity threshold for chunk relevance (default: 0.35, lowered for better recall)",
    )
    parser.add_argument("--max-chunks", type=int, default=10, help="Maximum relevant chunks per question (default: 10)")
    parser.add_argument(
        "--primary-metric",
        choices=["jaccard", "rouge_l", "token_containment", "asymmetric_jaccard"],
        default="rouge_l",
        help="Primary similarity metric (default: rouge_l - preserves sequence order, better for conversation flow)",
    )
    parser.add_argument(
        "--query-strategy",
        choices=["focused", "question", "answer", "combined"],
        default="combined",
        help="Query strategy: focused (full content), question (Q only), answer (Q+A), combined (Q for query, focused for validation) (default: combined)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show similarity scores for analysis",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: Process only 1 document for testing (fast)",
    )
    parser.add_argument(
        "--limit-ingestion",
        type=int,
        help="Limit number of conversations to ingest into ChromaDB (for partial processing)",
    )
    parser.add_argument(
        "--limit-extraction",
        type=int,
        help="Limit number of questions to process for ground truth extraction",
    )
    parser.add_argument(
        "--validate-collection",
        action="store_true",
        help="Run collection health check before processing",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Path to checkpoint file for resuming interrupted ingestion (default: auto-generated)",
    )
    parser.add_argument(
        "--chroma-path",
        default="chroma_longmemeval",
        help="ChromaDB storage path for LongMemEval (default: chroma_longmemeval)",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=768,
        help="Embedding dimensions (768 recommended for quality/cost balance) (default: 768)",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["openai", "sentence-transformer", "huggingface"],
        default="openai",
        help="Embedding model type: 'openai' (API), 'sentence-transformer' (local GPU), "
        "'huggingface' (API) (default: openai)",
    )

    # Execution options
    parser.add_argument(
        "--skip-ingestion", action="store_true", help="Skip ChromaDB ingestion if collection already exists"
    )
    parser.add_argument("--force-reingest", action="store_true", help="Force re-ingestion even if collection exists")
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip dataset validation check (assume existing collection contains required conversations)",
    )

    args = parser.parse_args()

    # Set up checkpoint path early (needed for header display)
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path and not args.skip_ingestion:
        # Auto-generate checkpoint path based on chroma path
        checkpoint_path = f"{args.chroma_path}_ingestion_checkpoint.json"

    # Display dataset-specific header
    header_panel = Panel.fit(
        f"[bold blue]🎯 LongMemEval Ground Truth Extraction[/bold blue]\n"
        f"[dim]Dataset:[/dim] [cyan]LongMemEval Conversations[/cyan]\n"
        f"[dim]Purpose:[/dim] [yellow]Generate chunk relevance ground truth for retrieval evaluation[/yellow]\n"
        f"[dim]Method:[/dim] [magenta]{args.primary_metric} similarity + ChromaDB + "
        f"{args.embedding_model} embeddings[/magenta]\n"
        f"[dim]Query Strategy:[/dim] [yellow]{args.query_strategy}[/yellow] | "
        f"[dim]Debug:[/dim] [cyan]{'ON' if args.debug else 'OFF'}[/cyan]\n"
        f"[dim]Test Mode:[/dim] [{'green' if args.test_mode else 'dim'}]{'ON' if args.test_mode else 'OFF'}[/{'green' if args.test_mode else 'dim'}] | "
        f"[dim]Validate:[/dim] [cyan]{'ON' if args.validate_collection else 'OFF'}[/cyan] | "
        f"[dim]Skip Dataset Validation:[/dim] [{'yellow' if args.skip_dataset_validation else 'dim'}]{'ON' if args.skip_dataset_validation else 'OFF'}[/{'yellow' if args.skip_dataset_validation else 'dim'}]\n"
        f"[dim]Ingestion Limit:[/dim] [yellow]{args.limit_ingestion or 'ALL'}[/yellow] | "
        f"[dim]Extraction Limit:[/dim] [yellow]{args.limit_extraction or 'ALL'}[/yellow] | "
        f"[dim]Checkpoint:[/dim] [green]{'ENABLED' if checkpoint_path else 'DISABLED'}[/green]\n"
        f"[dim]Embeddings:[/dim] [cyan]{args.embedding_model} ({args.embedding_dimensions}d)[/cyan]\n"
        f"[dim]Threshold:[/dim] [green]{args.similarity_threshold}[/green] | "
        f"[dim]Max chunks:[/dim] [blue]{args.max_chunks}[/blue]",
        title="📊 LongMemEval Processing",
        border_style="blue",
    )
    console.print(header_panel)

    # Validate LongMemEval dataset files
    focused_df, full_df = validate_longmemeval_files(args.focused_csv, args.full_csv)

    # Handle force-reingest BEFORE initializing extractor to avoid dimension conflicts
    if args.force_reingest:
        console.print("[bold yellow]🔄 Force re-ingestion requested[/bold yellow]")

        # Check what will be deleted
        temp_client = chromadb.PersistentClient(path=args.chroma_path)
        existing_collections = []
        total_documents = 0

        # Find all LongMemEval conversation collections
        try:
            all_collections = temp_client.list_collections()
            for coll in all_collections:
                if coll.name.startswith("longmemeval_conv_"):
                    existing_collections.append(coll.name)
                    total_documents += coll.count()

            if existing_collections:
                console.print(f"[bold red]⚠️  This will permanently delete:[/bold red]")
                console.print(
                    f"  • {len(existing_collections)} ChromaDB conversation collections with {total_documents:,} total documents"
                )
                console.print(f"  • Collection pattern: longmemeval_conv_*")
                if checkpoint_path and Path(checkpoint_path).exists():
                    console.print(f"  • Checkpoint file: {checkpoint_path}")
            else:
                console.print("[yellow]No existing conversation collections found to delete[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Error checking existing collections: {e}[/yellow]")
            existing_collections = []

        if existing_collections:
            # Ask for confirmation
            confirm = input("\n🔴 Are you sure you want to proceed with force re-ingestion? [y/N]: ").strip().lower()
            if confirm not in ["y", "yes"]:
                console.print("[yellow]Force re-ingestion cancelled[/yellow]")
                del temp_client
                return

        console.print("[bold yellow]Clearing existing data...[/bold yellow]")
        deleted_count = 0
        for collection_name in existing_collections:
            try:
                temp_client.delete_collection(collection_name)
                deleted_count += 1
            except Exception as e:
                console.print(f"[yellow]⚠️  Error deleting collection {collection_name}: {e}[/yellow]")

        if deleted_count > 0:
            console.print(f"[green]✅ Successfully deleted {deleted_count} conversation collections[/green]")
        else:
            console.print("[yellow]⚠️  No collections were deleted[/yellow]")

        # Clear checkpoint file if it exists
        if checkpoint_path and Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            console.print(f"[yellow]✓ Cleared checkpoint: {checkpoint_path}[/yellow]")

        del temp_client

    # Initialize LongMemEval-specific extractor
    console.print("[blue]🔧 Initializing LongMemEval ground truth extractor...[/blue]")
    extractor = GroundTruthExtractor(
        chroma_path=args.chroma_path,
        embedding_dimensions=args.embedding_dimensions,
        embedding_model=args.embedding_model,
    )

    # Run collection validation if requested
    if args.validate_collection:
        console.print("[blue]🔍 Running collection health check...[/blue]")
        # Get conversation IDs from the dataset for validation
        conversation_ids = full_df["custom_id"].head(10).tolist()  # Sample for validation
        if not extractor.validate_collections_health(conversation_ids):
            console.print("[red]❌ Collection validation failed![/red]")
            console.print("[yellow]Consider using --force-reingest to recreate the collections[/yellow]")
            return

    # Check ingestion status with better feedback
    console.print("[blue]🔍 Checking ChromaDB collections status...[/blue]")

    # Check if any collections exist by sampling a few conversation IDs
    sample_conversation_ids = full_df["custom_id"].head(5).tolist()
    existing_collections_count = 0
    total_existing_chunks = 0

    for conv_id in sample_conversation_ids:
        try:
            collection = extractor._get_or_create_collection(conv_id)
            chunk_count = collection.count()
            if chunk_count > 0:
                existing_collections_count += 1
                total_existing_chunks += chunk_count
        except Exception:
            pass

    if existing_collections_count == 0:
        console.print("[yellow]📥 No existing conversation collections found - starting ingestion process...[/yellow]")
        console.print(f"[dim]📋 Will process {len(full_df)} conversations from LongMemEval dataset[/dim]")
        extractor.ingest_longmemeval_conversations(
            full_df, test_mode=args.test_mode, num_docs=args.limit_ingestion, checkpoint_path=checkpoint_path
        )
    elif args.force_reingest:
        console.print("[yellow]🔄 Force re-ingestion proceeding with fresh collection...[/yellow]")
        console.print(f"[dim]📋 Will process {len(full_df)} conversations from LongMemEval dataset[/dim]")
        extractor.ingest_longmemeval_conversations(
            full_df, test_mode=args.test_mode, num_docs=args.limit_ingestion, checkpoint_path=checkpoint_path
        )
    elif args.skip_ingestion:
        console.print(
            f"[blue]⏩ Using existing ChromaDB collections with {existing_collections_count} sample collections[/blue]"
        )
        console.print("[dim]💡 Skipping ingestion as requested[/dim]")
    elif args.skip_dataset_validation:
        console.print(f"[blue]⏩ Using existing ChromaDB collections with {total_existing_chunks} total chunks[/blue]")
        console.print("[dim]💡 Skipping dataset validation as requested[/dim]")
    else:
        # Check if we need more comprehensive validation
        console.print(
            f"[blue]🔍 Found {existing_collections_count} existing collections with {total_existing_chunks} chunks - validating dataset coverage...[/blue]"
        )

        # Check a larger sample for validation
        validation_sample_size = min(20, max(5, len(full_df) // 10))
        validation_sample_ids = full_df["custom_id"].head(validation_sample_size).tolist()

        missing_conversations = []
        for conv_id in validation_sample_ids:
            try:
                collection = extractor._get_or_create_collection(conv_id)
                chunk_count = collection.count()
                if chunk_count == 0:
                    missing_conversations.append(conv_id)
            except Exception:
                missing_conversations.append(conv_id)

        missing_count = len(missing_conversations)
        missing_percentage = (missing_count / validation_sample_size) * 100

        if missing_count > 0:
            console.print(
                f"[yellow]⚠️  Dataset validation: {missing_count}/{validation_sample_size} ({missing_percentage:.1f}%) sample conversations missing[/yellow]"
            )

            # If many conversations are missing, likely need full ingestion
            if missing_percentage > 20:  # More lenient for per-conversation collections
                console.print("[yellow]🔄 Starting ingestion process to ensure complete dataset coverage...[/yellow]")
                console.print(f"[dim]📋 Will process {len(full_df)} conversations from LongMemEval dataset[/dim]")
                extractor.ingest_longmemeval_conversations(
                    full_df, test_mode=args.test_mode, num_docs=args.limit_ingestion, checkpoint_path=checkpoint_path
                )
            else:
                console.print(
                    f"[green]✅ Dataset validation acceptable - {validation_sample_size - missing_count} of {validation_sample_size} sample conversations found[/green]"
                )
        else:
            console.print(
                f"[green]✅ Dataset validation passed - all {validation_sample_size} sample conversations found[/green]"
            )
            console.print("[dim]💡 Use --force-reingest to re-ingest or --skip-ingestion to suppress this check[/dim]")

    # Extract LongMemEval ground truth
    console.print(f"[blue]🎯 Starting ground truth extraction for {len(focused_df)} LongMemEval questions...[/blue]")
    console.print("[dim]💡 This involves similarity search + overlap analysis for each question[/dim]")

    ground_truth_data = extractor.extract_longmemeval_ground_truth(
        focused_df,
        similarity_threshold=args.similarity_threshold,
        max_chunks_per_question=args.max_chunks,
        primary_metric=args.primary_metric,
        query_strategy=args.query_strategy,
        debug_mode=args.debug,
        limit_questions=args.limit_extraction,
    )

    # Save results with progress indicator
    console.print("[blue]💾 Saving ground truth data...[/blue]")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor.save_longmemeval_ground_truth(ground_truth_data, str(output_path))

    console.print(f"[green]✅ Ground truth saved to: {output_path}[/green]")

    # Display collection statistics
    stats = extractor.get_collection_stats()
    stats_panel = Panel.fit(
        f"[bold green]📈 ChromaDB Collections Statistics[/bold green]\n"
        f"[dim]Strategy:[/dim] [cyan]{stats['collection_strategy']}[/cyan]\n"
        f"[dim]Total collections:[/dim] [yellow]{stats['total_collections']}[/yellow]\n"
        f"[dim]Total chunks:[/dim] [yellow]{stats['total_chunks']}[/yellow]\n"
        f"[dim]Avg chunks per collection:[/dim] [blue]{stats['avg_chunks_per_collection']:.1f}[/blue]\n"
        f"[dim]Chunk size:[/dim] [blue]{stats['chunk_size']} tokens[/blue]\n"
        f"[dim]Overlap:[/dim] [magenta]{stats['overlap']} tokens[/magenta]",
        title="🗃️ ChromaDB Stats",
        border_style="green",
    )
    console.print(stats_panel)

    console.print("[bold green]🎉 LongMemEval ground truth extraction completed successfully![/bold green]")
    console.print(f"[dim]Output saved to: {output_path}[/dim]")


if __name__ == "__main__":
    main()
