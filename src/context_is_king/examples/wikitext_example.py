#!/usr/bin/env python3
"""
Example: WikiText Query Generation Pipeline

This example demonstrates how to use the WikiText query generation pipeline
to process a small sample of the Natural Questions dataset.

Run this example:
    python examples/wikitext_example.py
"""

import asyncio
import sys
from pathlib import Path

# Import from parent context_is_king module
from context_is_king.query_generation import WikiTextProcessor, WikiTextConfig


async def main():
    """Run a small example of the WikiText processing pipeline."""

    print("🚀 WikiText Query Generation Example")
    print("=" * 50)

    # Create configuration for a small test run
    config = WikiTextConfig(
        # Use local processed data
        local_data_file=Path("data/processed/nq_question_answer.parquet"),
        sample_size=50,  # Very small sample
        # Text processing settings
        chunk_size=1000,  # Smaller chunks for demo
        chunk_overlap=100,
        max_token_length=1200,
        # Embedding settings (smaller batches)
        embedding_batch_size=10,
        embedding_max_batch_size=25,
        # Filtering settings (conservative limits)
        max_concurrent_requests=5,
        requests_per_minute=30,
        tokens_per_minute=50000,
        # Output
        data_dir=Path("example_data"),
        log_level="INFO",
    )

    print(f"📊 Configuration:")
    print(f"  Local data file: {config.local_data_file}")
    print(f"  Sample size: {config.sample_size}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Data directory: {config.data_dir}")
    print()

    # Initialize processor
    processor = WikiTextProcessor(config)

    try:
        print("🔄 Running pipeline...")
        print("  Note: This will make API calls and may take a few minutes")
        print()

        # Run the complete pipeline
        results = await processor.process_pipeline(
            skip_embedding=False,  # Generate embeddings
            skip_filtering=True,  # Skip filtering to save API calls
            skip_chroma=False,  # Store in ChromaDB
            use_async_filtering=True,
        )

        print("✅ Pipeline completed successfully!")
        print("=" * 50)

        # Display results
        summary = results["pipeline_summary"]
        print(f"📈 Results Summary:")
        print(f"  Total stages: {summary['total_stages']}")
        print(f"  Completed stages: {summary['stages_completed']}")
        print(f"  Progress: {summary['progress_percent']:.1f}%")
        print(f"  Documents processed: {summary['processed_documents']}")
        print(f"  Documents with embeddings: {summary['embedded_documents']}")
        print(f"  Final document count: {results['final_document_count']}")
        print()

        # Stage details
        print("📋 Stage Details:")
        for stage_name, stage_data in results["stages"].items():
            if stage_data["status"] == "completed":
                print(f"  ✅ {stage_name}")
                if "documents_loaded" in stage_data:
                    print(f"     └─ Documents: {stage_data['documents_loaded']}")
                if "chunks_created" in stage_data:
                    print(f"     └─ Chunks: {stage_data['chunks_created']}")
                if "embeddings_generated" in stage_data:
                    print(f"     └─ Embeddings: {stage_data['embeddings_generated']}")
            elif stage_data["status"] == "skipped":
                print(f"  ⏭️  {stage_name} (skipped)")
            else:
                print(f"  ❌ {stage_name} ({stage_data['status']})")
        print()

        # ChromaDB information
        if "chroma_ingestion" in results["stages"] and results["stages"]["chroma_ingestion"]["status"] == "completed":
            chroma_stats = results["stages"]["chroma_ingestion"]["ingestion_stats"]
            collection_name = results["stages"]["chroma_ingestion"]["collection_name"]
            print(f"🗄️  ChromaDB Collection: {collection_name}")
            print(f"  Documents stored: {chroma_stats['documents_added']}")
            print(f"  Successful batches: {chroma_stats['successful_batches']}")
            print()

        print("🎉 Example completed successfully!")
        print(f"📁 Output data saved to: {config.data_dir}")
        print()
        print("Next steps:")
        print("  1. Examine the generated data files")
        print("  2. Query the ChromaDB collection")
        print("  3. Use the processed data for experiments")

    except KeyboardInterrupt:
        print("\n⚠️  Example interrupted by user")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Check that your ORQ_API_KEY is set correctly")


if __name__ == "__main__":
    # Check for API key
    import os

    if not os.getenv("ORQ_API_KEY"):
        print("❌ Error: ORQ_API_KEY environment variable is required")
        print("Please set your ORQ API key:")
        print("  export ORQ_API_KEY=your_key_here")
        sys.exit(1)

    asyncio.run(main())
