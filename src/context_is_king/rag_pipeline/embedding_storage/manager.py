"""
Embedding generation and ChromaDB storage management with GPU acceleration
"""

import time
from pathlib import Path
from typing import Any

import chromadb
import torch
from chromadb import Collection
from rich.console import Console
from rich.progress import Progress
from sentence_transformers import SentenceTransformer

from ..types import DocumentChunk, PipelineConfig

console = Console()


class EmbeddingStorageManager:
    """Manages embedding generation and ChromaDB storage"""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize embedding model with GPU if available
        device = config.embedding_device if torch.cuda.is_available() else "cpu"
        console.print(f"[blue]Initializing embedding model on {device}[/blue]")

        self.embedding_model = SentenceTransformer(config.embedding_model, device=device)

        # Initialize ChromaDB client
        self.chroma_client = self._initialize_chroma_client()

        # Embedding cache for efficiency
        self._embedding_cache = {}

        console.print(f"[green]EmbeddingStorageManager initialized with {config.embedding_model}[/green]")

    def create_collection(self, collection_name: str, metadata: dict[str, Any]) -> Collection:
        """
        Create or get ChromaDB collection for experiment

        Args:
            collection_name: Unique name for the collection
            metadata: Metadata to store with the collection

        Returns:
            ChromaDB collection instance
        """
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=None,  # We handle embeddings manually
            )
            console.print(f"[yellow]Retrieved existing collection: {collection_name}[/yellow]")

        except Exception:
            # Create new collection
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata=metadata,
                embedding_function=None,  # We handle embeddings manually
            )
            console.print(f"[green]Created new collection: {collection_name}[/green]")

        return collection

    def store_chunks(self, chunks: list[DocumentChunk], collection_name: str) -> bool:
        """Store chunks in the specified collection"""
        try:
            # Create or get collection with default metadata
            collection = self.create_collection(collection_name, {"description": f"Collection for {collection_name}"})

            # Batch embed and store
            self.batch_embed_and_store(chunks, collection)

            console.print(f"[green]✅ Stored {len(chunks)} chunks in collection '{collection_name}'[/green]")
            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to store chunks in '{collection_name}': {e}[/red]")
            return False

    def search(self, query: str, collection_name: str, k: int = 10) -> list | None:
        """Search for similar chunks in a collection"""
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            query_embedding = self.embed_query(query)

            results = collection.query(query_embeddings=[query_embedding], n_results=k)

            # Convert results to RetrievalResult objects
            from ..types import RetrievalResult

            retrieval_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc_id, content, metadata, distance) in enumerate(
                    zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                        strict=False,
                    )
                ):
                    # Create DocumentChunk from stored data
                    chunk = DocumentChunk(
                        content=content,
                        doc_id=metadata.get("doc_id", ""),
                        chunk_id=metadata.get("chunk_id", ""),
                        start_char=metadata.get("start_char", 0),
                        end_char=metadata.get("end_char", 0),
                        metadata=metadata,
                    )

                    result = RetrievalResult(
                        chunk=chunk,
                        similarity_score=1.0 - distance,  # Convert distance to similarity
                        rank=i + 1,
                    )
                    retrieval_results.append(result)

            return retrieval_results

        except Exception as e:
            console.print(f"[red]❌ Search failed for collection '{collection_name}': {e}[/red]")
            return None

    def batch_embed_and_store(self, chunks: list[DocumentChunk], collection: Collection) -> None:
        """
        Batch process embeddings and store in ChromaDB

        Args:
            chunks: List of document chunks to embed and store
            collection: ChromaDB collection to store in
        """
        if not chunks:
            console.print("[yellow]No chunks to process[/yellow]")
            return

        console.print(f"[blue]Processing {len(chunks)} chunks in batches of {self.config.embedding_batch_size}[/blue]")

        # Process chunks in batches
        batch_size = self.config.embedding_batch_size
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        with Progress() as progress:
            task = progress.add_task("Processing chunks...", total=len(chunks))

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]

                # Extract texts for embedding
                texts = [chunk.content for chunk in batch_chunks]

                # Generate embeddings for batch
                embeddings = self._generate_embeddings_batch(texts)

                # Prepare data for ChromaDB
                ids = [chunk.full_id for chunk in batch_chunks]
                documents = texts
                metadatas = [self._prepare_chunk_metadata(chunk) for chunk in batch_chunks]

                # Store in ChromaDB
                collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

                progress.update(task, advance=len(batch_chunks))

        console.print(f"[green]Successfully stored {len(chunks)} chunks in collection {collection.name}[/green]")

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get collection statistics for validation"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()

            # Get sample of documents to analyze
            sample_results = collection.peek(limit=10)

            stats = {
                "collection_name": collection_name,
                "document_count": count,
                "embedding_dimension": len(sample_results["embeddings"][0]) if sample_results["embeddings"] else 0,
                "sample_metadata": sample_results["metadatas"][:3] if sample_results["metadatas"] else [],
                "exists": True,
            }

        except Exception as e:
            stats = {"collection_name": collection_name, "error": str(e), "exists": False}

        return stats

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.chroma_client.delete_collection(collection_name)
            console.print(f"[red]Deleted collection: {collection_name}[/red]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to delete collection {collection_name}: {e}[/red]")
            return False

    def list_collections(self) -> list[str]:
        """List all collections"""
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            console.print(f"[red]Failed to list collections: {e}[/red]")
            return []

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query for retrieval"""
        # Check cache first
        if query in self._embedding_cache:
            return self._embedding_cache[query]

        embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]

        # Cache the embedding
        self._embedding_cache[query] = embedding.tolist()

        return embedding.tolist()

    def warm_up_model(self) -> None:
        """Warm up the embedding model with a dummy encoding"""
        console.print("[blue]Warming up embedding model...[/blue]")
        start_time = time.time()

        dummy_texts = ["This is a test sentence for warming up the model."] * 10
        self.embedding_model.encode(dummy_texts, convert_to_tensor=False)

        warmup_time = time.time() - start_time
        console.print(f"[green]Model warmed up in {warmup_time:.2f}s[/green]")

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get embedding model statistics"""
        return {
            "model_name": self.config.embedding_model,
            "device": str(self.embedding_model.device),
            "max_seq_length": getattr(self.embedding_model, "max_seq_length", "unknown"),
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "cache_size": len(self._embedding_cache),
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "gpu_memory_cached": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        }

    def clear_cache(self) -> None:
        """Clear embedding cache"""
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        console.print(f"[yellow]Cleared embedding cache ({cache_size} entries)[/yellow]")

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            console.print("[yellow]Cleared GPU cache[/yellow]")

    def _initialize_chroma_client(self) -> chromadb.Client:
        """Initialize ChromaDB persistent client"""
        db_path = Path(self.config.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(db_path))

        console.print(f"[green]ChromaDB client initialized at {db_path}[/green]")
        return client

    def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            # Encode batch of texts
            embeddings = self.embedding_model.encode(
                texts, batch_size=self.config.embedding_batch_size, convert_to_tensor=False, show_progress_bar=False
            )

            # Convert to list format for ChromaDB
            return embeddings.tolist()

        except Exception as e:
            console.print(f"[red]Failed to generate embeddings: {e}[/red]")
            raise

    def _prepare_chunk_metadata(self, chunk: DocumentChunk) -> dict[str, Any]:
        """Prepare chunk metadata for ChromaDB storage"""
        metadata = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "content_length": len(chunk.content),
            **chunk.metadata,
        }

        # ChromaDB has limitations on metadata types, so clean up
        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned_metadata[key] = value
            elif value is None:
                cleaned_metadata[key] = ""
            else:
                cleaned_metadata[key] = str(value)

        return cleaned_metadata
