# Context Window Scaling Experiment

This experiment measures how LLM call duration scales with context window size across multiple models and providers.

## Overview

- **Goal**: Understand performance characteristics of different LLMs as context size increases
- **Models**: 6 models via ORQ proxy (OpenAI GPT, Google Gemini, Anthropic Claude)
- **Token Sizes**: Powers of 10 from 10¹ to 10⁶ tokens
- **Methodology**: Controlled timing with tiktoken validation and retry logic

## Files

- `context_scaling.py` - Main experiment module with reusable classes
- `run_experiment.py` - Command-line interface for running experiments  
- `context_window_experiment.ipynb` - Interactive Jupyter notebook
- `README.md` - This documentation

## Quick Start

### 1. Prerequisites

```bash
# Ensure you have the required dependencies
pip install openai pandas matplotlib seaborn tiktoken

# Set your ORQ API key
export ORQ_API_KEY="your_orq_api_key_here"
```

### 2. Command Line Usage

```bash
# Quick test (recommended first run)
python run_experiment.py --quick-test

# Full experiment (all models, all sizes, ~108 API calls)
python run_experiment.py --full-experiment

# Custom experiment
python run_experiment.py --models gpt-5-mini claude-haiku --tokens 1000 10000 --iterations 5

# Show help
python run_experiment.py --help
```

### 3. Python Module Usage

```python
from context_scaling import ContextWindowExperiment, ExperimentAnalyzer

# Initialize experiment
experiment = ContextWindowExperiment(data_dir="../../data")

# Run quick test
results = experiment.run_experiment(
    models={"gpt-5-mini": experiment.MODELS["gpt-5-mini"]},
    token_sizes=[10, 100, 1000],
    iterations_per_test=2
)

# Analyze results
analyzer = ExperimentAnalyzer(results)
analyzer.print_summary()
analyzer.create_visualizations()
analyzer.save_results()
```

### 4. Jupyter Notebook

Open `context_window_experiment.ipynb` for an interactive experience with:
- Step-by-step experiment setup
- Visualization tools
- Results analysis
- Utility functions

## Experiment Configuration

### Models Tested

| Model Name | ORQ Proxy ID | Token Limit |
|------------|--------------|-------------|
| GPT-5 Mini | `azure/gpt-5-mini` | 1,000,000 |
| GPT-5 Nano | `azure/gpt-5-nano` | 272,000 |
| GPT-4.1 Mini | `azure/gpt-4.1-mini` | 1,000,000 |
| GPT-4.1 Nano | `azure/gpt-4.1-nano` | 1,000,000 |
| Gemini 2.5 Flash | `google-ai/gemini-2.5-flash` | 1,000,000 |
| Gemini 2.0 Flash Lite | `google-ai/gemini-2.0-flash-lite-001` | 1,000,000 |
| Claude Sonnet 4 | `azure/claude-sonnet-4-20250514` | 1,000,000 |
| Claude Haiku | `anthropic/claude-3-haiku-20240307` | 200,000 |

**Note**: Token sizes are automatically filtered based on model limits. For example, GPT-5 Nano will skip 500K and 1M token tests, and Claude Haiku will skip 500K and 1M token tests.

### Token Sizes

- 10 tokens (10¹)
- 100 tokens (10²)  
- 1,000 tokens (10³)
- 10,000 tokens (10⁴)
- 100,000 tokens (10⁵)
- 1,000,000 tokens (10⁶)

### Data Sources

Text content is generated from:
- Paul Graham essays (from `data/paul_graham_essays/`)
- ArXiv papers (from `data/arxiv_papers/`)
- Content is shuffled and combined to create exact token counts

## Results and Analysis

### Metrics Collected

- **Duration**: Total response time (seconds)
- **Throughput**: Input tokens processed per second
- **Success Rate**: Percentage of successful API calls
- **Error Messages**: Detailed failure reasons

### Analysis Features

- Performance scaling patterns
- Model comparison visualizations
- Statistical summaries
- Success rate analysis
- Log-log correlation analysis

### Output Files

Results are saved to `../../data/results/`:
- `experiment_TIMESTAMP.csv` - Tabular results
- `experiment_TIMESTAMP.json` - Complete results with metadata

## Example Commands

```bash
# Test only Claude models
python run_experiment.py --models claude-haiku claude-sonnet-4 --tokens 100 1000 10000

# Quick comparison of two models
python run_experiment.py --models gpt-5-mini gemini-2.5-flash --tokens 1000 10000 --iterations 3

# Test large contexts only
python run_experiment.py --tokens 100000 1000000 --iterations 1

# Save to custom location
python run_experiment.py --quick-test --output-dir ./my_results --filename my_experiment
```

## Expected Results

### Scaling Patterns
- **Linear scaling**: Most models show linear increase in processing time
- **Context limits**: Some models may fail at very large context sizes  
- **Provider differences**: Significant variance between OpenAI, Google, and Anthropic

### Performance Characteristics
- **Speed**: Varies dramatically between models
- **Reliability**: Success rates may decrease with context size
- **Throughput**: Tokens/second typically decreases with larger contexts

### New Features

#### 🛡️ Rate Limit Handling
- **Automatic 429 Error Parsing**: Extracts retry delays from error messages
- **Smart Retry Logic**: Waits exactly as long as the API requests
- **Example**: "retry after 44 seconds" → waits 44 seconds before retry

#### 📏 Token Limit Filtering  
- **Model-Specific Limits**: Each model has defined token limits
- **Automatic Filtering**: Skips token sizes that exceed model capacity
- **Clear Warnings**: Shows which token sizes are skipped and why

#### 📊 Enhanced Model Information
```python
experiment = ContextWindowExperiment()
experiment.display_model_info()  # Shows all models and limits
```

## Troubleshooting

### Common Issues

**API Key Issues**
```bash
# Verify your API key is set
echo $ORQ_API_KEY

# Set it if missing
export ORQ_API_KEY="your_key_here"
```

**Import Errors**
```python
# If running from different directory, add to path
import sys
sys.path.append('path/to/context_window_scaling')
```

**Data Directory Issues**
```python
# Explicitly set data directory
experiment = ContextWindowExperiment(data_dir="/full/path/to/data")
```

**Rate Limiting**
- The experiment includes automatic delays between calls  
- 429 errors are automatically parsed and proper retry delays applied
- If you still hit rate limits, reduce iterations or increase base delays

**Memory Issues**
- Very large contexts (1M tokens) may cause memory problems
- Consider testing smaller contexts first

## Architecture

### Class Structure

```
ContextWindowExperiment
├── __init__(data_dir, orq_api_key)
├── prepare_text_data() -> List[str]
├── create_text_with_exact_tokens(target_tokens) -> str
├── time_llm_call(model_name, model_id, context_text) -> ExperimentResult
├── run_single_experiment(model, token_count) -> ExperimentResult
└── run_experiment(models, token_sizes, iterations) -> List[ExperimentResult]

ExperimentAnalyzer
├── __init__(results)
├── print_summary()
├── create_visualizations()
└── save_results() -> Tuple[Path, Path]
```

### Data Flow

1. **Text Preparation**: Load and shuffle content from multiple sources
2. **Token Validation**: Use tiktoken to ensure exact token counts
3. **API Timing**: Measure total request duration including network latency
4. **Result Collection**: Store all metrics and error information
5. **Analysis**: Generate statistics, visualizations, and insights
6. **Persistence**: Save results for future analysis

## Contributing

To extend this experiment:

1. **Add new models**: Update `MODELS` dict in `ContextWindowExperiment`
2. **New metrics**: Extend `ExperimentResult` dataclass
3. **Different prompts**: Modify `time_llm_call` method
4. **Analysis features**: Add methods to `ExperimentAnalyzer`

## License

This experiment is part of the PyData 2025 "Context is King" project.