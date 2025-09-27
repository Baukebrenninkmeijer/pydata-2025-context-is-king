# WikiText Query Generation Pipeline

This module provides a complete pipeline for processing local WikiText datasets and generating synthetic queries for research experiments. It extracts the functionality from the Jupyter notebook and provides a clean, modular, production-ready implementation.

## Features

- **Dataset Loading**: Loads locally processed Natural Questions data with sampling support
- **Data Validation**: Validates local data format and filters invalid entries
- **Document Chunking**: Splits documents into manageable chunks with configurable overlap
- **Embedding Generation**: Creates embeddings with batching and rate limiting
- **LLM-based Filtering**: Filters documents using async LLM evaluation with semaphore control
- **ChromaDB Integration**: Stores processed documents in vector database
- **Checkpoint Recovery**: Resume processing from any completed stage
- **Comprehensive Logging**: Detailed logging with rotation and level control

## Architecture

### Core Components

- **`WikiTextProcessor`**: Main pipeline controller
- **`EmbeddingGenerator`**: Handles embedding creation with batching
- **`DocumentFilter`**: LLM-based document filtering with async processing  
- **`ChromaManager`**: Vector database operations
- **`WikiTextConfig`**: Configuration management with validation

### Processing Stages

1. **Data Loading**: Load local processed Natural Questions data
2. **Data Validation**: Validate format and filter invalid entries
3. **Chunking**: Split documents into smaller pieces
4. **Embedding Generation**: Create embeddings with rate limiting
5. **Document Filtering**: Filter documents using LLM evaluation
6. **ChromaDB Ingestion**: Store documents in vector database

## Quick Start

### Basic Usage

```bash
# Full pipeline with local data
python scripts/generate_wikitext_queries.py --local-data-file data/processed/nq_question_answer.parquet

# Process a sample for testing
python scripts/generate_wikitext_queries.py --local-data-file data/processed/nq_question_answer.parquet --sample-size 1000

# Skip certain stages
python scripts/generate_wikitext_queries.py --skip-embedding --skip-filtering

# Resume from checkpoint
python scripts/generate_wikitext_queries.py --resume
```

### Configuration Options

```bash
# Text processing
--chunk-size 2000 --chunk-overlap 300 --max-token-length 2500

# Embedding settings
--embedding-model azure/text-embedding-3-large --embedding-batch-size 50

# Filtering with rate limiting
--filter-model azure/gpt-4o-mini --max-concurrent 50 --tokens-per-minute 500000

# Output control
--data-dir custom_data --output-file results.json --log-level DEBUG
```

## Rate Limiting and Semaphore Control

The pipeline implements sophisticated rate limiting to handle API constraints:

### Embedding Generation
- Configurable batch sizes for memory management
- Automatic retry with exponential backoff
- Progress tracking with checkpoints

### LLM Filtering
- **Semaphore-based concurrency control**: Limits concurrent API calls
- **Token-rate limiting**: Tracks token usage per minute with sliding window
- **Adaptive backoff**: Waits when rate limits are approached
- **Error resilience**: Continues processing despite individual failures

### Example Rate Limiting Configuration

```python
config = WikiTextConfig(
    max_concurrent_requests=50,     # Max parallel requests
    requests_per_minute=500,        # API rate limit
    tokens_per_minute=300_000,      # Token rate limit
    request_timeout=120.0           # Individual request timeout
)
```

## Programming Interface

### Using the Pipeline Programmatically

```python
import asyncio
from context_is_king.query_generation import WikiTextProcessor, WikiTextConfig

# Create configuration
config = WikiTextConfig(
    local_data_file=Path("data/processed/nq_question_answer.parquet"),
    sample_size=5000,
    chunk_size=1900,
    embedding_model="azure/text-embedding-3-small",
    filter_model="azure/gpt-5-nano"
)

# Run pipeline
processor = WikiTextProcessor(config)
results = await processor.process_pipeline()

print(f"Processed {results['final_document_count']} documents")
```

### Stage-by-Stage Processing

```python
# Load and process dataset
df = processor.load_dataset()
df_answers = processor.extract_answers_from_annotations(df)
chunked_df = processor.chunk_documents(df_answers)

# Generate embeddings
embedded_df = processor.embedding_generator.load_or_generate_embeddings(chunked_df)

# Filter documents
filtered_df, metadata = await processor.document_filter.run_filtering_pipeline(chunked_df)

# Store in ChromaDB
collection = processor.chroma_manager.get_or_create_collection("my_collection")
stats = processor.chroma_manager.add_documents_in_batches(collection, embedded_df)
```

## Filtering Results Analysis

The system automatically saves comprehensive filtering results for later inspection and analysis.

### What Gets Saved

When you run filtering, the system automatically creates these files in `logs/filtering_results/`:

1. **`{run_id}_{timestamp}_detailed.json`**: Complete evaluation results with document content
2. **`{run_id}_{timestamp}_summary.md`**: Human-readable summary report  
3. **`{run_id}_{timestamp}_criteria_analysis.json`**: Detailed criteria performance analysis
4. **`{run_id}_{timestamp}_rejected_documents.json`**: Detailed info about rejected documents
5. **`{run_id}_{timestamp}_filtered_documents.csv`**: CSV of documents that passed filtering

### Analyzing Results

Use the `FilteringAnalyzer` class to inspect filtering outcomes:

```python
from src.wikitext_query_generator.filtering_analysis import FilteringAnalyzer, analyze_latest_run, list_all_runs

# Quick analysis of the most recent run
analyze_latest_run()

# List all filtering runs
list_all_runs()

# Detailed analysis
analyzer = FilteringAnalyzer()
analyzer.show_runs_summary()

# Load and analyze a specific run
run_data = analyzer.load_filtering_run("filtering_1234567890")
analyzer.analyze_criteria_performance(run_data)
analyzer.analyze_failure_patterns(run_data)
analyzer.show_rejected_documents_sample(run_data, max_docs=5)

# Compare multiple runs
analyzer.compare_runs(["filtering_1234567890", "filtering_1234567891"])
```

### Understanding Failure Patterns

The analysis helps you understand:
- Which filter criteria are most/least effective
- Common combinations of failed criteria
- Examples of rejected documents and why they failed
- Correlations between different criteria
- Pass rates over time across different runs

## Configuration Reference

### Dataset Configuration
- `local_data_file`: Path to local processed parquet file
- `sample_size`: Optional sample size for testing

### Text Processing
- `chunk_size`: Size of text chunks (default: 1900)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `max_token_length`: Maximum tokens per chunk (default: 2000)

### Embedding Configuration  
- `embedding_model`: Model for embeddings (default: "azure/text-embedding-3-small")
- `embedding_batch_size`: API batch size (default: 100)
- `embedding_max_batch_size`: Memory management batch size (default: 50,000)

### Filtering Configuration
- `filter_model`: Model for filtering (default: "azure/gpt-5-nano")
- `filter_criteria`: List of filtering criteria
- `max_concurrent_requests`: Semaphore limit (default: 50)
- `requests_per_minute`: Rate limit (default: 500)
- `tokens_per_minute`: Token rate limit (default: 300,000)

## Monitoring and Debugging

### Logging
The pipeline creates detailed logs:
- `logs/wikitext_processor.log`: Main pipeline events
- `logs/embedding_generation.log`: Embedding operations
- `logs/document_filtering.log`: Filtering operations
- `logs/chroma_operations.log`: Database operations

### Status Checking
```bash
# Check current status
python scripts/generate_wikitext_queries.py --status

# List ChromaDB collections
python scripts/generate_wikitext_queries.py --list-collections
```

### Progress Tracking
The pipeline provides detailed progress information:
- Stage completion status
- Document counts at each stage
- Processing statistics
- Error summaries

## Error Handling

The pipeline includes robust error handling:

### Automatic Recovery
- **Checkpoint system**: Resume from any completed stage
- **Partial results**: Save intermediate results for recovery
- **Error isolation**: Continue processing despite individual failures

### Common Issues

1. **Rate Limiting**: Reduce `max_concurrent_requests` or `tokens_per_minute`
2. **Memory Issues**: Reduce `embedding_max_batch_size`
3. **API Errors**: Check ORQ_API_KEY and model availability
4. **Disk Space**: Monitor output directory size

## Integration with Research Pipeline

This module integrates with the broader research pipeline:

### Output Formats
- **Parquet files**: For data processing
- **Delta tables**: For incremental updates
- **ChromaDB collections**: For vector search
- **JSON results**: For metadata and statistics

### Compatibility
- Works with existing `context_is_king` package
- Compatible with reranking value experiments
- Supports LongMemEval evaluation framework

## Performance Optimization

### Scaling Recommendations

For large datasets:
```python
config = WikiTextConfig(
    embedding_batch_size=200,           # Larger batches
    embedding_max_batch_size=100_000,   # More memory usage
    max_concurrent_requests=100,        # Higher concurrency
    tokens_per_minute=500_000          # Higher rate limits
)
```

For limited resources:
```python
config = WikiTextConfig(
    embedding_batch_size=50,            # Smaller batches
    embedding_max_batch_size=10_000,    # Less memory usage
    max_concurrent_requests=10,         # Lower concurrency
    tokens_per_minute=100_000          # Conservative rate limits
)
```

## Contributing

When extending the pipeline:

1. **Add new stages**: Extend `PipelineState` with new stages
2. **Custom filters**: Subclass `DocumentFilter` with new criteria
3. **New embeddings**: Extend `EmbeddingGenerator` for new models
4. **Database backends**: Implement new storage managers

### Testing
```bash
# Test with small sample
python scripts/generate_wikitext_queries.py --sample-size 100 --skip-chroma

# Test specific stages
python scripts/generate_wikitext_queries.py --skip-embedding --sync-filtering
```