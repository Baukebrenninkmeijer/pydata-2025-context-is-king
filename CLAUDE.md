# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyData 2025 research project comparing long context window approaches against Retrieval Augmented Generation (RAG) for data grounding. The project implements the Chroma Context Rot study methodology to answer key research questions including:

- When must we switch from context window retrieval to RAG?
- Can we skip RAG and work with full context window when sources fit?
- Is there a context window threshold where performance drops sharply?
- Do models with larger max context windows outperform smaller ones at the same context size?
- How do cost-performance and latency trade-offs change with context size?

## Guidelines

- Always use azure/openai models via the ORQ API proxy.
- For an OPENAI_API_KEY, always use the ORQ API key. "ORQ_API_KEY".
- for embedding, use `openai/text-embedding-3-small` for Azure OpenAI and `text-embedding-3-small` for OpenAI.
- Follow the unified model interface in `context_is_king.models.ModelInterface`.
- Use azure/gpt-4.1-mini for most operations.
- **IMPORTANT**: For LLM filtering tasks, use `azure/gpt-4.1-mini` instead of `azure/gpt-5-nano`. GPT-5-nano has significant rate limiting that causes filtering failures.

## Development Commands

### Experiments

```bash
# Context advantage experiment (main research experiment)
python src/context_is_king/experiments/context_advantage/context_advantage_experiment.py --help

# Quick needle-in-haystack test
python src/context_is_king/experiments/context_advantage/context_advantage_experiment.py --experiment-type needle --quick-test

# Resume interrupted experiment
python src/context_is_king/experiments/context_advantage/context_advantage_experiment.py --resume-from <experiment_id>

# Context window scaling experiment
python src/context_is_king/experiments/context_window_scaling/run_experiment.py --help

# Quick scaling test
python src/context_is_king/experiments/context_window_scaling/run_experiment.py --quick-test

# Reranking value experiment
python src/context_is_king/experiments/reranking_value/run_experiment.py --help
```

### Data Preparation

Use the installed CLI entry points (see `pyproject.toml` `[project.scripts]`):

```bash
ck-download-datasets          # Paul Graham essays, arXiv papers, Chroma needles
ck-ingest-chroma              # Ingest LongMemEval into ChromaDB
ck-ingest-data                # RAG pipeline data ingestion
ck-generate-queries-v2        # WikiText query pipeline (current version)
ck-extract-longmemeval-ground-truth
```

## Code Architecture

### Package Structure

The main package `context_is_king` is organized into:

- **`models/`** - Unified model interfaces (ModelInterface, ModelConfig, QueryResult)
- **`evaluation/`** - Evaluation metrics (NeedleEvaluator, LongMemEvalEvaluator, ResultsAggregator)  
- **`experiments/`** - Shared experiment utilities (ContextWindowExperiment)
- **`rag_pipeline/`** - RAG pipeline implementation
- **`utils/`** - Common utilities (ready for future expansion)

### Experiment Architecture

Two main experimental tracks:

#### Context Advantage Experiment (`experiments/context_advantage/`)

**Research Question**: Do models with larger maximum context windows outperform smaller-window models at the same context size?

- **Main Controller**: `context_advantage_experiment.py` - Coordinates both needle-in-haystack and LongMemEval experiments
- **Needle Generation**: `needle_haystack/needle_generator.py` - Creates synthetic test cases
- **Haystack Building**: `needle_haystack/haystack_builder.py` - Assembles documents to target token counts  
- **LongMemEval Analysis**: `longmemeval_analysis/context_grouper.py` - Processes real-world evaluation data
- **Fixed Q&A Sets**: `data/context_advantage/fixed_qas/` - Pre-generated question-answer pairs for consistent evaluation

**Test Scenario Types**:

- **Direct Retrieval**: Simple fact extraction (e.g., "What is the founding year of Zenith Dynamics?")
- **Cross-Reference**: Connect information from multiple document sections
- **Synthesis**: Combine information requiring reasoning across sources
- **Domain Transfer**: Apply concepts across domains (e.g., biological concepts to cybersecurity)

**Key Features**:

- Checkpoint/recovery system for long-running experiments
- Progress tracking with tqdm integration
- Graceful shutdown handling (SIGINT/SIGTERM)
- Results saved as both JSON checkpoints and structured data
- Fixed synthetic Q&A sets ensure reproducible comparisons

#### Context Window Scaling (`experiments/context_window_scaling/`)

**Research Question**: How does LLM performance scale with context window size?

- **Main Module**: `scaling.py` (moved to `context_is_king.experiments.scaling`)
- **CLI Interface**: `run_experiment.py`
- **Interactive**: `context_window_experiment.ipynb`

### Data Flow Architecture

1. **Data Ingestion**: Raw datasets → ChromaDB + processed formats
2. **Experiment Setup**: Configuration → Haystack/Needle generation → Model queries
3. **Evaluation**: Query results → Metrics calculation → Statistical analysis  
4. **Checkpointing**: Intermediate results → JSON checkpoints for recovery

### Model Interface Design

Unified interface supports multiple providers via ORQ API:

- Rate limiting with exponential backoff
- Token counting with tiktoken
- Cost estimation and tracking
- Async/sync operation modes

## Key Configuration Files

### Environment Variables

Required for experiments:

```bash
ORQ_API_KEY=your_orq_api_key_here
```

### pyproject.toml

- Package configuration with comprehensive dependencies
- Ruff linting configuration (line length 120, extensive rule set)
- MyPy type checking configuration
- Use UV for dependency management.

### Standards

- Always use python 3.10+ native typehints. So always use dict instead of typing.Dict, list instead of typing.List. And use x | y instead of typing.Union[x, y].

### Experiment Configuration

Experiments use dataclass-based configuration:

- `ExperimentConfig` for context advantage experiments
- Model selection, context sizes, iterations, output paths
- Checkpoint and recovery settings

## Data Dependencies

### External Datasets

- Paul Graham essays (for content diversity)
- ArXiv papers (technical domain content)  
- LongMemEval dataset (real-world evaluation cases)

### Generated Data

- Synthetic needles (questions/facts for insertion)
- Haystacks (concatenated documents at target token counts)
- ChromaDB vector embeddings
- Fixed Q&A sets (40 total: 10 direct, 10 cross-reference, 10 synthesis, 10 domain transfer)

## Common Development Patterns

### Adding New Models

1. Add ModelConfig to `MODELS` dict in `ModelInterface`
2. Ensure API compatibility through ORQ proxy
3. Test with connection test method

### Adding New Experiments

1. Import from main package: `from context_is_king.models import ModelInterface`
2. Use shared evaluation metrics: `from context_is_king.evaluation import NeedleEvaluator`
3. Follow checkpoint/recovery pattern for long-running experiments
4. Use fixed Q&A sets from `data/context_advantage/fixed_qas/` for consistent evaluation

### Extending Evaluation Metrics

1. Add new evaluator classes to `context_is_king.evaluation`
2. Implement standard evaluation interface pattern
3. Update `__init__.py` exports

## Experimental Design Principles

The codebase emphasizes reproducibility and controlled evaluation:

### Fixed vs Dynamic Content

- **Fixed Q&A Sets**: Pre-generated synthetic questions ensure consistent evaluation across runs
- **Dynamic Haystacks**: Documents are concatenated to reach precise token targets with controlled composition
- **Position Tracking**: Needles inserted at strategic positions (beginning, early-middle, center, late-middle, end)

### Evaluation Complexity Levels

- **Simple**: Direct fact retrieval with clear answers
- **Complex**: Multi-hop reasoning, cross-domain knowledge transfer, synthesis tasks

### Reproducibility Features

- Checkpoint recovery for interrupted long-running experiments
- Fixed random seeds and deterministic document ordering
- Unified interfaces for comparing different context window approaches across multiple LLM providers

## Utility Scripts

The project includes several utility scripts for data processing and experiment management:

### Data Processing

- **`concatenate_wikiqa_parquets.py`** - Concatenates all WikiQA experiment parquet files into a single consolidated file
  - Handles schema normalization for inconsistent column types
  - Converts `stage_2_output` from String to List format for consistency
  - Casts retrieval benchmark columns from Int64 to Float64
  - Saves output as both Parquet and Delta Lake formats

### Checkpoint System

- **`improved_checkpoint_system.py`** - Enhanced checkpoint manager preventing data overwrites
  - Generates unique experiment IDs with timestamps to prevent conflicts
  - Implements atomic file operations for safe checkpoint saves
  - Provides experiment recovery and resumption capabilities
  - Validates experiment completion before finalization

- **`notebook_checkpoint_integration.py`** - Notebook integration for the improved checkpoint system
  - Provides `run_single_experiment_improved()` and `run_full_experiment_improved()` functions
  - Handles parallel experiment execution with semaphore-based rate limiting
  - Integrates with existing notebook variables (chroma_client, tokenizer, etc.)

### Logging System

- **`loguru_disk_setup.py`** - Comprehensive disk-based logging for experiments
  - Creates multiple log files: main, structured JSON, errors-only, checkpoints
  - Implements log rotation and retention policies
  - Provides experiment phase logging with timing
  - Includes notebook integration with `setup_notebook_logging()`

### Usage Examples

```python
# Setup logging for experiments
from loguru_disk_setup import setup_notebook_logging, ExperimentPhaseLogger
log_info = setup_notebook_logging()

# Use improved checkpoint system in notebook
from notebook_checkpoint_integration import run_single_experiment_improved
results = await run_single_experiment_improved(
    qa_pairs=filtering_results,
    cleaned_df=html_cleaned_df,
    k=20, retrieval_kind='enhanced_rag', rerank=True,
    n_questions=200
)

# Phase-based logging
with ExperimentPhaseLogger("Data Loading"):
    # your experiment code
    pass
```

These utilities ensure data integrity, prevent experimental overwrites, and provide comprehensive tracking for long-running experiments.
