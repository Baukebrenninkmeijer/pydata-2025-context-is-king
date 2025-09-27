#!/usr/bin/env python3
"""
Data ingestion script for PyData 2025 Context Is King experiments
Creates separate ChromaDB collections for each dataset
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from src.rag_pipeline.config import load_config
from src.rag_pipeline.document_processing.processor import DocumentProcessor
from src.rag_pipeline.embedding_storage.manager import EmbeddingStorageManager
from src.rag_pipeline.types import Document

console = Console()


class DataIngestionManager:
    """Manages data ingestion into ChromaDB collections"""

    def __init__(self, config_override: dict[str, Any] = None):
        """Initialize with pipeline configuration"""
        self.config = load_config()

        # Override config if provided
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)

        self.processor = DocumentProcessor(self.config)
        self.storage = EmbeddingStorageManager(self.config)

        console.print("[blue]DataIngestionManager initialized[/blue]")

    def ingest_paul_graham_essays(self) -> dict[str, Any]:
        """Ingest Paul Graham essays into ChromaDB collection"""
        console.print("[blue]Ingesting Paul Graham essays...[/blue]")

        data_dir = Path("data/paul_graham_essays")
        collection_name = "paul_graham_essays"

        # Load metadata
        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        documents = []

        for essay_info in track(metadata, description="Loading essays..."):
            essay_path = data_dir / essay_info["filename"]

            if essay_path.exists():
                with open(essay_path, encoding="utf-8") as f:
                    content = f.read()

                document = Document(
                    content=content,
                    doc_id=Path(essay_info["filename"]).stem,
                    source=str(essay_path),
                    doc_type="paul_graham",
                    metadata={
                        "title": essay_info["title"],
                        "url": essay_info.get("url", ""),
                        "length": len(content),
                        "collection": collection_name,
                    },
                )
                documents.append(document)

        # Process and store documents
        console.print(f"Processing {len(documents)} Paul Graham essays...")
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
            console.print("[green]✅ Paul Graham essays ingested successfully[/green]")
        else:
            console.print("[red]❌ Failed to ingest Paul Graham essays[/red]")

        return result

    def ingest_arxiv_papers(self) -> dict[str, Any]:
        """Ingest arXiv papers into ChromaDB collection"""
        console.print("[blue]Ingesting arXiv papers...[/blue]")

        data_dir = Path("data/arxiv_papers")
        collection_name = "arxiv_papers"

        # Load metadata
        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        documents = []

        for paper_info in track(metadata, description="Loading papers..."):
            paper_path = data_dir / paper_info["filename"]

            if paper_path.exists():
                with open(paper_path, encoding="utf-8") as f:
                    content = f.read()

                document = Document(
                    content=content,
                    doc_id=Path(paper_info["filename"]).stem,
                    source=str(paper_path),
                    doc_type="arxiv",
                    metadata={
                        "title": paper_info["title"],
                        "abstract": paper_info.get("abstract", ""),
                        "length": len(content),
                        "collection": collection_name,
                    },
                )
                documents.append(document)

        # Process and store documents
        console.print(f"Processing {len(documents)} arXiv papers...")
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
            console.print("[green]✅ arXiv papers ingested successfully[/green]")
        else:
            console.print("[red]❌ Failed to ingest arXiv papers[/red]")

        return result

    def ingest_longmemeval(self) -> dict[str, Any]:
        """Ingest LongMemEval conversations into ChromaDB collection"""
        console.print("[blue]Ingesting LongMemEval conversations...[/blue]")

        data_dir = Path("data/longmemeval")
        collection_name = "longmemeval"

        # Load conversations
        conversations_path = data_dir / "sample_conversations.json"
        with open(conversations_path, encoding="utf-8") as f:
            conversations = json.load(f)

        documents = []

        for conv in track(conversations, description="Loading conversations..."):
            # Convert conversation to document format
            content_parts = []
            for msg in conv["messages"]:
                content_parts.append(f"{msg['role']}: {msg['content']}")

            content = "\n\n".join(content_parts)

            document = Document(
                content=content,
                doc_id=conv["conversation_id"],
                source=str(conversations_path),
                doc_type="conversation",
                metadata={
                    "conversation_id": conv["conversation_id"],
                    "needle": conv.get("needle", ""),
                    "question": conv.get("question", ""),
                    "answer": conv.get("answer", ""),
                    "message_count": len(conv["messages"]),
                    "collection": collection_name,
                },
            )
            documents.append(document)

        # Process and store documents
        console.print(f"Processing {len(documents)} conversations...")
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
            console.print("[green]✅ LongMemEval conversations ingested successfully[/green]")
        else:
            console.print("[red]❌ Failed to ingest LongMemEval conversations[/red]")

        return result

    def create_mixed_collection(self, pg_sample_size: int = 5, arxiv_sample_size: int = 10) -> dict[str, Any]:
        """Create a mixed collection with samples from both PG and arXiv for cross-domain experiments"""
        console.print(f"[blue]Creating mixed collection (PG: {pg_sample_size}, arXiv: {arxiv_sample_size})...[/blue]")

        collection_name = "mixed_pg_arxiv"

        # Get sample documents from both collections
        pg_results = self.storage.search("startup", collection_name="paul_graham_essays", k=pg_sample_size)
        arxiv_results = self.storage.search("machine learning", collection_name="arxiv_papers", k=arxiv_sample_size)

        # Convert search results back to chunks
        mixed_chunks = []

        if pg_results:
            for result in pg_results[:pg_sample_size]:
                mixed_chunks.append(result.chunk)

        if arxiv_results:
            for result in arxiv_results[:arxiv_sample_size]:
                mixed_chunks.append(result.chunk)

        # Store in new collection
        if mixed_chunks:
            console.print(f"Storing {len(mixed_chunks)} mixed chunks in collection '{collection_name}'...")
            success = self.storage.store_chunks(mixed_chunks, collection_name)
        else:
            console.print("[yellow]No chunks found for mixed collection[/yellow]")
            success = False

        result = {
            "collection_name": collection_name,
            "pg_chunks": min(pg_sample_size, len(pg_results) if pg_results else 0),
            "arxiv_chunks": min(arxiv_sample_size, len(arxiv_results) if arxiv_results else 0),
            "total_chunks": len(mixed_chunks),
            "success": success,
        }

        if success:
            console.print("[green]✅ Mixed collection created successfully[/green]")
        else:
            console.print("[red]❌ Failed to create mixed collection[/red]")

        return result

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics for all collections"""
        console.print("[blue]Getting collection statistics...[/blue]")

        collections = ["paul_graham_essays", "arxiv_papers", "longmemeval", "mixed_pg_arxiv"]

        stats = {}

        for collection_name in collections:
            try:
                collection = self.storage.chroma_client.get_collection(name=collection_name)
                count = collection.count()
                stats[collection_name] = {"chunk_count": count, "exists": True}
                console.print(f"   {collection_name}: {count} chunks")
            except Exception as e:
                stats[collection_name] = {"chunk_count": 0, "exists": False, "error": str(e)}
                console.print(f"   {collection_name}: not found")

        return stats

    def cleanup_collections(self, collections: list[str] = None) -> dict[str, bool]:
        """Clean up specified collections (or all if none specified)"""
        if collections is None:
            collections = ["paul_graham_essays", "arxiv_papers", "longmemeval", "mixed_pg_arxiv"]

        console.print(f"[yellow]Cleaning up collections: {collections}[/yellow]")

        results = {}
        for collection_name in collections:
            try:
                self.storage.chroma_client.delete_collection(name=collection_name)
                results[collection_name] = True
                console.print(f"   ✅ Deleted: {collection_name}")
            except Exception as e:
                results[collection_name] = False
                console.print(f"   ❌ Failed to delete {collection_name}: {e}")

        return results


def main():
    """Main ingestion function"""
    console.print("🚀 [bold blue]PyData 2025 Context Is King - Data Ingestion[/bold blue]")
    console.print("=" * 60)

    # Initialize ingestion manager
    ingestion_manager = DataIngestionManager()

    # Check existing collections
    console.print("\n📊 Current collection status:")
    stats = ingestion_manager.get_collection_stats()

    # Ingest all datasets
    results = {}

    console.print("\n🔄 Starting data ingestion...")

    # Ingest Paul Graham essays
    results["paul_graham"] = ingestion_manager.ingest_paul_graham_essays()

    # Ingest arXiv papers
    results["arxiv"] = ingestion_manager.ingest_arxiv_papers()

    # Ingest LongMemEval conversations
    results["longmemeval"] = ingestion_manager.ingest_longmemeval()

    # Create mixed collection for cross-domain experiments
    # Note: This will only work after the individual collections are created
    console.print("\n🔀 Creating mixed collection...")
    results["mixed"] = ingestion_manager.create_mixed_collection()

    # Final statistics
    console.print("\n📊 Final collection statistics:")
    final_stats = ingestion_manager.get_collection_stats()

    # Summary
    console.print("\n📋 [bold green]Ingestion Summary[/bold green]")
    for dataset, result in results.items():
        if result.get("success", False):
            console.print(f"   ✅ {dataset}: {result.get('chunks_count', 'N/A')} chunks")
        else:
            console.print(f"   ❌ {dataset}: failed")

    console.print("\n✅ [bold green]Data ingestion completed![/bold green]")
    console.print("\n📁 Available collections:")
    for collection_name, stats in final_stats.items():
        if stats["exists"]:
            console.print(f"   - {collection_name}: {stats['chunk_count']} chunks")


if __name__ == "__main__":
    main()
