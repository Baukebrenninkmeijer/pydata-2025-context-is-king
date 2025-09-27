"""ChromaDB integration for vector storage and retrieval."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chromadb
import polars as pl
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from .config import WikiTextConfig

# Initialize rich console for progress tracking
console = Console()


class ChromaManager:
    """Manage ChromaDB operations for document storage and retrieval."""

    def __init__(self, config: WikiTextConfig):
        self.config = config
        
        # Ensure ChromaDB directory exists
        config.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=str(config.chroma_db_path))
            logger.info(f"ChromaDB client initialized successfully at: {config.chroma_db_path}")
            
            # Test ChromaDB connection by listing collections
            try:
                collections = self.client.list_collections()
                logger.info(f"ChromaDB connection test successful. Found {len(collections)} existing collections.")
            except Exception as test_error:
                logger.warning(f"ChromaDB connection test failed: {test_error}")
                # Don't fail initialization, but log the warning
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client at {config.chroma_db_path}: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

        # Setup logging
        logger.configure(
            handlers=[
                {
                    "sink": "logs/chroma_operations.log",
                    "level": config.log_level,
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                    "rotation": "10 MB",
                }
            ]
        )

    def get_or_create_collection(self, collection_name: str, metadata: dict | None = None) -> chromadb.Collection:
        """Get existing collection or create new one with defensive error handling.

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection

        Returns:
            ChromaDB Collection object
        """
        try:
            # First try to get existing collection
            collection = self.client.get_collection(collection_name)
            logger.info(f"Found existing collection: {collection_name}")
            return collection
        except Exception as e:
            # Check if this is a "Collection does not exist" error (any exception type)
            error_str = str(e).lower()
            if "does not exist" in error_str or "not found" in error_str:
                # Collection doesn't exist, try to create it
                logger.info(f"Collection {collection_name} does not exist, creating new one...")
                try:
                    collection = self.client.create_collection(name=collection_name, metadata=metadata or {})
                    logger.info(f"Successfully created new collection: {collection_name}")
                    return collection
                except Exception as create_error:
                    logger.error(f"Failed to create collection {collection_name}: {create_error}")
                    logger.error(f"ChromaDB path: {self.config.chroma_db_path}")
                    logger.error(f"Collection metadata: {metadata}")
                    raise RuntimeError(f"Unable to create ChromaDB collection '{collection_name}': {create_error}") from create_error
            else:
                # Some other error - provide diagnostics
                logger.error(f"Unexpected error accessing collection {collection_name}: {e}")
                logger.error(f"ChromaDB path: {self.config.chroma_db_path}")
                logger.error(f"ChromaDB client type: {type(self.client)}")
                
                # List existing collections for debugging
                try:
                    existing_collections = [c.name for c in self.client.list_collections()]
                    logger.error(f"Existing collections: {existing_collections}")
                except Exception as list_error:
                    logger.error(f"Could not list collections: {list_error}")
                
                raise RuntimeError(f"Unable to access ChromaDB collection '{collection_name}': {e}. This might indicate a ChromaDB configuration issue.") from e

    def add_documents_in_batches(
        self, collection: chromadb.Collection, documents_df: pl.DataFrame, batch_size: int = 500, max_workers: int = 4
    ) -> dict[str, Any]:
        """Add documents to ChromaDB collection in parallel batches.

        Args:
            collection: ChromaDB collection
            documents_df: DataFrame with documents, embeddings, and metadata
            batch_size: Size of each batch
            max_workers: Number of parallel workers

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Adding {len(documents_df)} documents to collection in batches of {batch_size}")

        # Prepare data
        ids = documents_df.select("unique_id").to_series().to_list()
        documents = documents_df.select("chunked_prompt").to_series().to_list()
        embeddings = documents_df.select("embedding").to_series().to_list()

        # Create metadata
        metadatas = []
        for row in documents_df.to_dicts():
            metadata = {
                "document_url": row.get("document_url", ""),
                "token_count": row.get("token_count", 0),
                "context_length": row.get("context_length", 0),
            }
            metadatas.append(metadata)

        # Split into batches
        total_documents = len(ids)
        num_batches = (total_documents + batch_size - 1) // batch_size

        def add_batch(batch_idx: int) -> dict[str, Any]:
            """Add a single batch of documents."""
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_documents)

            batch_ids = ids[start_idx:end_idx]
            batch_documents = documents[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]

            try:
                collection.add(
                    ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas
                )

                return {"batch_idx": batch_idx, "documents_added": len(batch_ids), "success": True, "error": None}

            except Exception as e:
                logger.error(f"Error adding batch {batch_idx}: {e}")
                return {"batch_idx": batch_idx, "documents_added": 0, "success": False, "error": str(e)}

        # Process batches in parallel with cancellation handling
        successful_batches = 0
        failed_batches = 0
        total_added = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {executor.submit(add_batch, batch_idx): batch_idx for batch_idx in range(num_batches)}

            # Process completed batches with simple logging
            logger.info(f"Processing ChromaDB ingestion: {num_batches} batches")
            try:
                for future in as_completed(future_to_batch):
                    result = future.result()

                    if result["success"]:
                        successful_batches += 1
                        total_added += result["documents_added"]
                    else:
                        failed_batches += 1

                    logger.info(f"ChromaDB batch completed: {successful_batches + failed_batches}/{num_batches}")
            except KeyboardInterrupt:
                logger.info("ChromaDB ingestion interrupted by user - cancelling remaining batches")
                # Cancel remaining futures
                for future in future_to_batch:
                    if not future.done():
                        future.cancel()
                raise

        # Return statistics
        stats = {
            "total_documents": total_documents,
            "documents_added": total_added,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "batch_size": batch_size,
            "collection_name": collection.name,
        }

        logger.info(f"Batch ingestion complete: {stats}")
        return stats

    def query_collection(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 10,
        where: dict | None = None,
        include: list[str] = ["documents", "distances", "metadatas"],
    ) -> dict[str, Any]:
        """Query documents from ChromaDB collection with defensive error handling.

        Args:
            collection_name: Name of collection to query
            query_texts: List of query texts
            n_results: Number of results to return per query
            where: Optional filter conditions
            include: What to include in results

        Returns:
            Query results
        """
        try:
            collection = self.client.get_collection(collection_name)
        except ValueError as e:
            error_msg = f"Collection '{collection_name}' does not exist. Please ensure the ChromaDB ingestion stage has completed successfully."
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unable to access collection '{collection_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        try:
            results = collection.query(query_texts=query_texts, n_results=n_results, where=where, include=include)
            logger.info(f"Queried collection {collection_name} with {len(query_texts)} queries")
            return results
        except Exception as e:
            error_msg = f"Error querying collection '{collection_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get statistics about a collection with defensive error handling.

        Args:
            collection_name: Name of collection

        Returns:
            Collection statistics with error handling
        """
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()

            stats = {"name": collection_name, "document_count": count, "metadata": collection.metadata}

            logger.info(f"Collection {collection_name} stats: {stats}")
            return stats

        except ValueError as e:
            # Collection doesn't exist - this is expected in some scenarios
            error_msg = f"Collection '{collection_name}' does not exist"
            logger.warning(error_msg)
            return {"name": collection_name, "document_count": 0, "exists": False, "error": error_msg}
        except Exception as e:
            # Unexpected error
            error_msg = f"Error accessing collection '{collection_name}': {e}"
            logger.error(error_msg)
            return {"name": collection_name, "document_count": 0, "exists": False, "error": error_msg}

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of collection to check

        Returns:
            True if collection exists, False otherwise
        """
        try:
            self.client.get_collection(collection_name)
            return True
        except ValueError:
            # Collection doesn't exist
            return False
        except Exception as e:
            # Unexpected error - log it but return False for safety
            logger.error(f"Error checking if collection '{collection_name}' exists: {e}")
            return False

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections with their statistics.

        Returns:
            List of collection information
        """
        collections = self.client.list_collections()
        collection_info = []

        for collection in collections:
            try:
                stats = self.get_collection_stats(collection.name)
                collection_info.append(stats)
            except Exception as e:
                logger.error(f"Error getting stats for {collection.name}: {e}")
                collection_info.append({"name": collection.name, "error": str(e)})

        return collection_info

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.

        Args:
            collection_name: Name of collection to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    def backup_collection(self, collection_name: str, output_file: str) -> dict[str, Any]:
        """Backup a collection to a file.

        Args:
            collection_name: Name of collection to backup
            output_file: Path to output file

        Returns:
            Backup statistics
        """
        try:
            collection = self.client.get_collection(collection_name)

            # Get all documents (this could be memory intensive for large collections)
            results = collection.get(include=["documents", "embeddings", "metadatas"])

            # Create backup DataFrame
            backup_df = pl.DataFrame(
                {
                    "id": results["ids"],
                    "document": results["documents"],
                    "embedding": results["embeddings"],
                    "metadata": results["metadatas"],
                }
            )

            # Save to parquet for efficient storage
            backup_df.write_parquet(output_file)

            stats = {
                "collection_name": collection_name,
                "documents_backed_up": len(backup_df),
                "output_file": output_file,
                "success": True,
            }

            logger.info(f"Backed up collection {collection_name}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error backing up collection {collection_name}: {e}")
            return {"collection_name": collection_name, "success": False, "error": str(e)}
