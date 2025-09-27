#!/usr/bin/env python3
"""
Data ingestion script for real Chroma Context Rot datasets
"""

import json
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from rich.progress import track

from src.rag_pipeline.config import load_config
from src.rag_pipeline.document_processing.processor import DocumentProcessor
from src.rag_pipeline.embedding_storage.manager import EmbeddingStorageManager
from src.rag_pipeline.types import Document

console = Console()


class ChromaDataIngestionManager:
    """Manages ingestion of real Chroma Context Rot datasets"""

    def __init__(self, config_override: dict[str, Any] = None):
        """Initialize with pipeline configuration"""
        self.config = load_config()

        # Override config if provided
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)

        self.processor = DocumentProcessor(self.config)
        self.storage = EmbeddingStorageManager(self.config)

        console.print("[blue]ChromaDataIngestionManager initialized[/blue]")

    def ingest_paul_graham_chroma(self) -> dict[str, Any]:
        """Ingest complete Paul Graham essays from Chroma corpus"""
        console.print("[blue]Ingesting Paul Graham essays (Chroma corpus)...[/blue]")

        data_dir = Path("context_rot_data/NIAH/PaulGrahamEssays")
        collection_name = "paul_graham_chroma"

        if not data_dir.exists():
            console.print(f"[red]Directory not found: {data_dir}[/red]")
            return {"success": False, "error": "Directory not found"}

        documents = []

        # Process all .txt files
        essay_files = list(data_dir.glob("*.txt"))

        for essay_file in track(essay_files, description="Loading Chroma PG essays..."):
            try:
                with open(essay_file, encoding="utf-8") as f:
                    content = f.read()

                document = Document(
                    content=content,
                    doc_id=essay_file.stem,
                    source=str(essay_file),
                    doc_type="paul_graham_chroma",
                    metadata={"filename": essay_file.name, "collection": collection_name, "source": "chroma_official"},
                )
                documents.append(document)

            except Exception as e:
                console.print(f"[red]Failed to load {essay_file.name}: {e}[/red]")

        # Process and store documents
        console.print(f"Processing {len(documents)} Paul Graham essays from Chroma...")
        chunks = self.processor.process_documents(documents)

        console.print(f"Storing {len(chunks)} chunks in collection '{collection_name}'...")
        success = self.storage.store_chunks(chunks, collection_name)

        result = {
            "collection_name": collection_name,
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "success": success,
        }

        if success:
            console.print("[green]✅ Paul Graham Chroma corpus ingested successfully[/green]")
        else:
            console.print("[red]❌ Failed to ingest Paul Graham Chroma corpus[/red]")

        return result

    def ingest_longmemeval_focused(self) -> dict[str, Any]:
        """Ingest LongMemEval focused dataset"""
        console.print("[blue]Ingesting LongMemEval focused dataset...[/blue]")

        csv_path = Path("context_rot_data/LongMemEval/cleaned_longmemeval_s_focused.csv")
        collection_name = "longmemeval_focused"

        if not csv_path.exists():
            console.print(f"[red]File not found: {csv_path}[/red]")
            return {"success": False, "error": "File not found"}

        documents = []

        # Read CSV file using Polars for better handling of large fields
        console.print("Loading focused dataset with Polars...")
        try:
            df = pl.read_csv(csv_path, ignore_errors=True)
            console.print(f"Found {len(df)} rows in focused dataset")

            # Convert to Python dicts for processing
            rows = df.to_dicts()

            for i, row in enumerate(track(rows, description="Processing focused conversations...")):
                try:
                    custom_id = row.get("custom_id", f"focused_{i}")
                    focused_prompt = row.get("focused_prompt", "")
                    token_count = row.get("token_count", 0)
                    question = row.get("question", "")
                    answer = row.get("answer", "")

                    # Handle token count conversion
                    if isinstance(token_count, str):
                        token_count = int(token_count) if token_count.isdigit() else 0

                    # Create document from the conversation
                    document = Document(
                        content=str(focused_prompt),
                        doc_id=f"longmemeval_focused_{custom_id}",
                        source=str(csv_path),
                        doc_type="longmemeval_focused",
                        metadata={
                            "custom_id": str(custom_id),
                            "token_count": token_count,
                            "question": str(question),
                            "answer": str(answer),
                            "collection": collection_name,
                            "dataset_type": "focused",
                        },
                    )
                    documents.append(document)

                except Exception as e:
                    console.print(f"[red]Failed to process row {i}: {e}[/red]")
                    continue

        except Exception as e:
            console.print(f"[red]Failed to load CSV with Polars: {e}[/red]")
            return {"success": False, "error": str(e)}

        # Process and store documents
        console.print(f"Processing {len(documents)} focused conversations...")
        chunks = self.processor.process_documents(documents)

        console.print(f"Storing {len(chunks)} chunks in collection '{collection_name}'...")
        success = self.storage.store_chunks(chunks, collection_name)

        result = {
            "collection_name": collection_name,
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "success": success,
        }

        if success:
            console.print("[green]✅ LongMemEval focused dataset ingested successfully[/green]")
        else:
            console.print("[red]❌ Failed to ingest LongMemEval focused dataset[/red]")

        return result

    def ingest_longmemeval_full(self) -> dict[str, Any]:
        """Ingest LongMemEval full dataset"""
        console.print("[blue]Ingesting LongMemEval full dataset...[/blue]")

        csv_path = Path("context_rot_data/LongMemEval/cleaned_longmemeval_s_full.csv")
        collection_name = "longmemeval_full"

        if not csv_path.exists():
            console.print(f"[red]File not found: {csv_path}[/red]")
            return {"success": False, "error": "File not found"}

        documents = []

        # Read CSV file using Polars with streaming for large dataset
        console.print("Loading full dataset with Polars (streaming mode)...")
        try:
            # Use lazy loading for large dataset
            df_lazy = pl.scan_csv(csv_path, ignore_errors=True)

            # Get total count
            total_rows = df_lazy.select(pl.count()).collect().item()
            console.print(f"Found {total_rows:,} rows in full dataset")

            # Process in batches for memory efficiency
            batch_size = 1000
            total_batches = (total_rows + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size

                # Read batch using slice
                batch_df = df_lazy.slice(start_idx, batch_size).collect()
                batch_rows = batch_df.to_dicts()

                console.print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_rows)} rows)")

                for i, row in enumerate(track(batch_rows, description=f"Batch {batch_idx + 1}")):
                    try:
                        global_idx = start_idx + i
                        custom_id = row.get("custom_id", f"full_{global_idx}")
                        full_prompt = row.get("full_prompt", "")
                        token_count = row.get("token_count", 0)
                        question = row.get("question", "")
                        answer = row.get("answer", "")

                        # Handle token count conversion
                        if isinstance(token_count, str):
                            token_count = int(token_count) if token_count.isdigit() else 0

                        # Create document from the conversation
                        document = Document(
                            content=str(full_prompt),
                            doc_id=f"longmemeval_full_{custom_id}",
                            source=str(csv_path),
                            doc_type="longmemeval_full",
                            metadata={
                                "custom_id": str(custom_id),
                                "token_count": token_count,
                                "question": str(question),
                                "answer": str(answer),
                                "collection": collection_name,
                                "dataset_type": "full",
                            },
                        )
                        documents.append(document)

                    except Exception as e:
                        console.print(f"[red]Failed to process row {global_idx}: {e}[/red]")
                        continue

                # Memory management - process and store each batch
                if documents and len(documents) >= 5000:  # Store every 5K documents
                    console.print(f"Intermediate storage: {len(documents)} documents...")
                    batch_chunks = self.processor.process_documents(documents)

                    # Store or append to collection
                    if batch_idx == 0:
                        # First batch - create collection
                        success = self.storage.store_chunks(batch_chunks, collection_name)
                        if not success:
                            return {"success": False, "error": "Failed to create collection"}
                    else:
                        # Subsequent batches - append to existing collection
                        try:
                            collection = self.storage.chroma_client.get_collection(name=collection_name)
                            self.storage.batch_embed_and_store(batch_chunks, collection)
                        except Exception as e:
                            console.print(f"[red]Failed to append batch: {e}[/red]")
                            return {"success": False, "error": str(e)}

                    documents = []  # Clear for memory

        except Exception as e:
            console.print(f"[red]Failed to load CSV with Polars: {e}[/red]")
            return {"success": False, "error": str(e)}

        # Process and store documents in batches
        console.print(f"Processing {len(documents)} full conversations...")

        # Process documents in chunks for memory efficiency
        all_chunks = []
        doc_batch_size = 500

        for i in range(0, len(documents), doc_batch_size):
            batch_docs = documents[i : i + doc_batch_size]
            console.print(
                f"Processing document batch {i // doc_batch_size + 1}/{(len(documents) + doc_batch_size - 1) // doc_batch_size}"
            )
            batch_chunks = self.processor.process_documents(batch_docs)
            all_chunks.extend(batch_chunks)

        console.print(f"Storing {len(all_chunks)} chunks in collection '{collection_name}'...")
        success = self.storage.store_chunks(all_chunks, collection_name)

        result = {
            "collection_name": collection_name,
            "documents_count": len(documents),
            "chunks_count": len(all_chunks),
            "success": success,
        }

        if success:
            console.print("[green]✅ LongMemEval full dataset ingested successfully[/green]")
        else:
            console.print("[red]❌ Failed to ingest LongMemEval full dataset[/red]")

        return result

    def ingest_needles_and_distractors(self) -> dict[str, Any]:
        """Ingest needles and distractors with similarity scores"""
        console.print("[blue]Ingesting needles and distractors...[/blue]")

        json_path = Path("context_rot_data/NIAH/needles_and_distractors.json")

        if not json_path.exists():
            console.print(f"[red]File not found: {json_path}[/red]")
            return {"success": False, "error": "File not found"}

        # Load needles and distractors
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        results = {}

        # Process different similarity thresholds
        similarity_collections = [
            ("needles_high_sim", 0.7, 1.0),
            ("needles_med_sim", 0.5, 0.7),
            ("needles_low_sim", 0.0, 0.5),
        ]

        for collection_name, min_sim, max_sim in similarity_collections:
            documents = []

            for domain, domain_data in data.items():
                question = domain_data.get("question", "")
                needles = domain_data.get("needles", {})

                for needle_id, needle_data in needles.items():
                    similarity = needle_data.get("needle_question_sim", 0.0)
                    needle_text = needle_data.get("needle", "")

                    # Filter by similarity threshold
                    if min_sim <= similarity < max_sim:
                        document = Document(
                            content=needle_text,
                            doc_id=f"{domain}_{needle_id}",
                            source=str(json_path),
                            doc_type="needle",
                            metadata={
                                "domain": domain,
                                "needle_id": needle_id,
                                "question": question,
                                "similarity_score": similarity,
                                "similarity_range": f"{min_sim}-{max_sim}",
                                "collection": collection_name,
                            },
                        )
                        documents.append(document)

            if documents:
                # Process and store
                console.print(f"Processing {len(documents)} needles for {collection_name}...")
                chunks = self.processor.process_documents(documents)

                console.print(f"Storing {len(chunks)} chunks in collection '{collection_name}'...")
                success = self.storage.store_chunks(chunks, collection_name)

                results[collection_name] = {
                    "collection_name": collection_name,
                    "documents_count": len(documents),
                    "chunks_count": len(chunks),
                    "success": success,
                    "similarity_range": f"{min_sim}-{max_sim}",
                }

                if success:
                    console.print(f"[green]✅ {collection_name} ingested successfully[/green]")
                else:
                    console.print(f"[red]❌ Failed to ingest {collection_name}[/red]")
            else:
                console.print(
                    f"[yellow]No needles found for {collection_name} (similarity {min_sim}-{max_sim})[/yellow]"
                )
                results[collection_name] = {"success": False, "documents_count": 0}

        return results

    def cleanup_old_collections(self) -> dict[str, bool]:
        """Clean up old collections before re-ingesting"""
        console.print("[yellow]Cleaning up old collections...[/yellow]")

        old_collections = [
            "paul_graham_essays",  # Our old partial essays
            "longmemeval",  # Old sample data
            "mixed_pg_arxiv",  # Will recreate after ingestion
        ]

        results = {}
        for collection_name in old_collections:
            try:
                self.storage.chroma_client.delete_collection(name=collection_name)
                results[collection_name] = True
                console.print(f"   ✅ Deleted: {collection_name}")
            except Exception as e:
                results[collection_name] = False
                console.print(f"   ⚠️  Could not delete {collection_name}: {e}")

        return results

    def create_mixed_collections(self) -> dict[str, Any]:
        """Create mixed collections for cross-domain experiments"""
        console.print("[blue]Creating mixed collections for cross-domain experiments...[/blue]")

        results = {}

        # PG needles in arXiv corpus for cross-domain testing
        pg_arxiv_chunks = []

        # Get some PG needles
        pg_results = self.storage.search("startup advice", collection_name="needles_high_sim", k=5)
        if pg_results:
            pg_arxiv_chunks.extend([r.chunk for r in pg_results])

        # Get some arXiv chunks
        arxiv_results = self.storage.search("machine learning", collection_name="arxiv_papers", k=10)
        if arxiv_results:
            pg_arxiv_chunks.extend([r.chunk for r in arxiv_results])

        if pg_arxiv_chunks:
            success = self.storage.store_chunks(pg_arxiv_chunks, "mixed_pg_arxiv")
            results["mixed_pg_arxiv"] = {"chunks_count": len(pg_arxiv_chunks), "success": success}

        return results

    def get_all_collection_stats(self) -> dict[str, Any]:
        """Get statistics for all collections"""
        console.print("[blue]Getting comprehensive collection statistics...[/blue]")

        expected_collections = [
            "paul_graham_chroma",
            "longmemeval_focused",
            "longmemeval_full",
            "needles_high_sim",
            "needles_med_sim",
            "needles_low_sim",
            "arxiv_papers",  # Keep existing
            "mixed_pg_arxiv",
        ]

        stats = {}

        for collection_name in expected_collections:
            try:
                collection = self.storage.chroma_client.get_collection(name=collection_name)
                count = collection.count()
                stats[collection_name] = {"chunk_count": count, "exists": True}
                console.print(f"   {collection_name}: {count:,} chunks")
            except Exception as e:
                stats[collection_name] = {"chunk_count": 0, "exists": False, "error": str(e)}
                console.print(f"   {collection_name}: not found")

        return stats


def main():
    """Main ingestion function for real Chroma data"""
    console.print("🚀 [bold blue]PyData 2025 Context Is King - Chroma Data Ingestion[/bold blue]")
    console.print("=" * 70)

    # Initialize ingestion manager
    ingestion_manager = ChromaDataIngestionManager()

    # Clean up old collections
    console.print("\n🧹 Cleanup phase:")
    cleanup_results = ingestion_manager.cleanup_old_collections()

    # Check existing collections
    console.print("\n📊 Current collection status:")
    stats = ingestion_manager.get_all_collection_stats()

    # Ingest all real datasets
    results = {}

    console.print("\n🔄 Starting Chroma data ingestion...")

    # 1. Paul Graham Chroma corpus (49 essays)
    results["paul_graham_chroma"] = ingestion_manager.ingest_paul_graham_chroma()

    # 2. LongMemEval focused (5K conversations)
    results["longmemeval_focused"] = ingestion_manager.ingest_longmemeval_focused()

    # 3. Needles and distractors with similarity thresholds
    needles_results = ingestion_manager.ingest_needles_and_distractors()
    results.update(needles_results)

    # 4. LongMemEval full (60K conversations) - Using Polars for large CSV
    console.print(
        "\n⚠️  [yellow]Starting LongMemEval full dataset (60K rows) with Polars - this may take 15-20 minutes...[/yellow]"
    )
    results["longmemeval_full"] = ingestion_manager.ingest_longmemeval_full()

    # 5. Create mixed collections
    console.print("\n🔀 Creating mixed collections...")
    mixed_results = ingestion_manager.create_mixed_collections()
    results.update(mixed_results)

    # Final statistics
    console.print("\n📊 Final collection statistics:")
    final_stats = ingestion_manager.get_all_collection_stats()

    # Summary
    console.print("\n📋 [bold green]Chroma Data Ingestion Summary[/bold green]")
    total_chunks = 0
    for dataset, result in results.items():
        if isinstance(result, dict) and result.get("success", False):
            chunk_count = result.get("chunks_count", 0)
            total_chunks += chunk_count
            console.print(f"   ✅ {dataset}: {chunk_count:,} chunks")
        elif isinstance(result, dict):
            console.print(f"   ❌ {dataset}: failed")
        else:
            console.print(f"   ℹ️  {dataset}: {result}")

    console.print(f"\n🎯 [bold green]Total chunks ingested: {total_chunks:,}[/bold green]")
    console.print("\n✅ [bold green]Chroma data ingestion completed![/bold green]")

    console.print("\n📁 Available collections for experiments:")
    for collection_name, stats in final_stats.items():
        if stats["exists"]:
            console.print(f"   - {collection_name}: {stats['chunk_count']:,} chunks")


if __name__ == "__main__":
    main()
