# Reranking Value Experiment

**Research Question**: Does reranking still make sense in a post-RAG world?

## Overview

This experiment investigates the value of reranking in modern retrieval systems by comparing three distinct approaches across different question complexities using LongMemEval dataset.

## Experimental Approaches

### 1. Full Context Retrieval
- Loads complete relevant documents into context window
- Uses existing context chopping to stay within model limits
- No retrieval step - direct question answering

### 2. Enhanced RAG without Reranking  
- **Query Enhancement**: LLM-based query rewriting for better coverage
- **Dual Retrieval**: Search with both original and rewritten queries
- **Result Fusion**: Reciprocal Rank Fusion (RRF) of results
- **Semantic Ranking**: Uses original similarity scores only

### 3. Enhanced RAG with Reranking
- Same as approach 2, but adds:
- **Cross-Encoder Reranking**: Re-scores fused results with cross-encoder
- **Final Ranking**: Uses reranking scores for context assembly

## Evaluation Framework

### LongMemEval Integration
- **Question Types**: factual, reasoning, synthesis, multi-hop
- **Context Groups**: Short (0-25K), Medium (25K-75K), Long (75K-150K)  
- **Ground Truth**: Uses existing LongMemEval expected answers

### Enhanced LLM Judge
- **Structured Output**: True/False correctness + confidence + detailed reasoning
- **Metrics**: answer_completeness, context_utilization per approach
- **Question-Type Aware**: Different evaluation criteria by complexity

### Context Tracking
- **Token Measurement**: Precise context size before LLM calls
- **Composition Analysis**: Query + context + system prompt breakdown  
- **Efficiency Metrics**: Context utilization ratios across approaches

## Key Components

### Reusable Components (in main package)
- `context_is_king.rag_pipeline.query_enhancement.QueryRewriter` - Uses OpenAI SDK via ORQ proxy
- `context_is_king.rag_pipeline.query_enhancement.DualRetrieval` - Enhanced retrieval with query fusion
- `context_is_king.evaluation.JudgeEvaluator` - LLM judge via ORQ proxy
- `context_is_king.evaluation.ContextTracker` - Token counting and analysis

### Experiment-Specific Components
- `reranking_experiment.py`: Main controller coordinating all approaches
- `run_experiment.py`: CLI interface with configuration options
- `analysis/results_analyzer.py`: Statistical analysis and comparison
- `analysis/cost_analyzer.py`: Token cost and efficiency analysis

## Usage

### Prerequisites
```bash
# Ensure your .env file in the root directory contains:
# ORQ_API_KEY=your_orq_api_key_here
# (The experiment will automatically load it)

# Install the package in development mode
pip install -e .
```

### Quick Test
```bash
python experiments/reranking_value/run_experiment.py --quick-test
```

### Full Experiment
```bash  
python experiments/reranking_value/run_experiment.py \
  --longmemeval-path data/LongMemEval/ \
  --output-dir results/reranking_value/ \
  --models openai/gpt-4-turbo anthropic/claude-3-sonnet-20240229 \
  --context-groups short medium long
```

### Resume Interrupted Experiment
```bash
python experiments/reranking_value/run_experiment.py --resume-from experiment_id_12345
```

## Expected Outputs

### Research Insights
1. **Reranking Value Quantification**: When reranking adds measurable benefit
2. **Context Size Thresholds**: Where full context becomes competitive  
3. **Query Enhancement Impact**: Value of dual retrieval strategies
4. **Question Complexity Analysis**: Optimal approach per LongMemEval type

### Practical Guidelines
1. **Decision Framework**: How to choose between approaches
2. **Cost-Performance Trade-offs**: Token efficiency vs accuracy gains
3. **Implementation Recommendations**: When to invest in reranking infrastructure

## Directory Structure

```
experiments/reranking_value/
├── README.md                       # This file
├── __init__.py                     # Package initialization
├── reranking_experiment.py         # Main experiment controller
├── run_experiment.py              # CLI interface
├── analysis/
│   ├── __init__.py
│   ├── results_analyzer.py        # Statistical analysis
│   └── cost_analyzer.py           # Cost-benefit analysis  
├── data/
│   ├── longmemeval_selected.json  # Curated test questions
│   └── ground_truth_mapping.json  # Expected answers
└── results/                       # Experiment outputs
```

## Analysis Framework

### Performance Metrics
- **Accuracy**: Correctness by approach and question type
- **Confidence**: Judge confidence in evaluations  
- **Context Efficiency**: Relevant content utilization ratios
- **Cost Analysis**: Token usage and computational overhead

### Statistical Tests
- **Significance Testing**: Pairwise approach comparisons
- **Effect Size**: Practical significance of differences
- **Question Type Stratification**: Analysis by complexity level
- **Context Size Scaling**: Performance across context groups

## Research Contributions

This experiment provides:
1. **Empirical Evidence**: Data-driven answer to reranking value question
2. **Reusable Components**: Enhanced RAG pipeline for future research
3. **Evaluation Framework**: Structured LLM judge for RAG assessment
4. **Practical Guidelines**: Decision support for RAG system design