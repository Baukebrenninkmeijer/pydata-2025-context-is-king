# Context Window Advantage Experiment

## Research Question
**Do models with larger maximum context windows outperform models with smaller maximum context windows when operating at the same context window size?**

For example: Does Claude Sonnet 4 (1M token limit) outperform Claude Haiku 3.5 (200K token limit) when both process exactly 100K tokens?

## Two-Pronged Experimental Design

### Prong 1: Synthetic Needle-in-Haystack Experiments
**Controlled evaluation with synthetic test cases**

#### Dataset Construction Strategy
- **Haystack Assembly**: Concatenate documents to reach exact token targets
  - Paul Graham essays + ArXiv papers for content diversity
  - Shuffle order to prevent position bias
  - Target sizes: 10K, 50K, 100K, 150K tokens

- **Needle Insertion Patterns**:
  - **Same-domain**: PG needles in PG-heavy haystacks
  - **Cross-domain**: PG needles in ArXiv-heavy haystacks  
  - **Mixed-domain**: Balanced content with varied needle types

- **Positioning Strategy**: 
  - Beginning (0-10%), Early-middle (25-35%), Center (45-55%), Late-middle (65-75%), End (90-100%)
  - Multiple needles per haystack for multi-hop reasoning tests

#### Test Scenario Types
1. **Direct Retrieval**: Find explicit facts
2. **Cross-Reference**: Connect facts from different sections
3. **Synthesis**: Combine information requiring reasoning
4. **Domain Transfer**: Answer questions requiring cross-domain knowledge

### Prong 2: LongMemEval Context-Length Analysis  
**Real-world validation using existing challenging dataset**

#### Methodology
- **Dataset Segmentation**: Group LongMemEval questions by optimal context length requirements
- **Context Length Bins**: 
  - Short (0-25K tokens)
  - Medium (25K-75K tokens) 
  - Long (75K-150K tokens)
  - Extended (150K+ tokens where applicable)

- **Performance Analysis**: 
  - Compare high-capacity vs low-capacity models within each context bin
  - Identify if certain context lengths show systematic advantages
  - Analyze question complexity interaction with context size

#### Benefits
- **Authentic Difficulty**: Real challenging questions vs synthetic needles
- **Diverse Task Types**: Multiple reasoning patterns already validated
- **Existing Baselines**: Can compare against published results

## Directory Structure

```
experiments/context_advantage/
├── README.md                          # This plan document
├── context_advantage_experiment.py    # Main experiment coordinator
├── needle_haystack/
│   ├── needle_generator.py           # Create synthetic test cases
│   ├── haystack_builder.py           # Construct document assemblies  
│   └── evaluator.py                  # Needle retrieval evaluation
├── longmemeval_analysis/
│   ├── context_grouper.py            # Group questions by context needs
│   ├── performance_analyzer.py       # Compare model performance by group
│   └── question_complexity.py        # Analyze difficulty interactions
├── shared/
│   ├── model_interface.py            # Unified model API
│   ├── metrics.py                    # Common evaluation metrics
│   └── visualization.py              # Results plotting
├── data/
│   ├── needles/                      # Synthetic test questions
│   ├── haystacks/                    # Constructed document assemblies
│   ├── longmemeval_grouped/          # Context-length grouped questions
│   └── raw/                          # Source documents
└── results/
    ├── needle_experiments/           # Synthetic results
    ├── longmemeval_analysis/         # Real-world validation
    └── combined_analysis/            # Cross-validation findings
```

## Model Comparison Strategy

### Same-Tier Context Window Comparisons

#### GPT-4 Family: Full Models
- **GPT-4** (8K tokens) → **GPT-4 Turbo** (128K tokens)
- **GPT-4o** (128K tokens) → **GPT-4.1** (1M tokens) *when available*

#### GPT-4 Family: Mini Models  
- **GPT-4o Mini** (128K tokens) → **GPT-4.1 Mini** (1M tokens) *when available*

#### Claude Family: Sonnet Models
- **Claude Sonnet 3.5** (200K tokens) → **Claude Sonnet 4** (1M tokens)
- **Claude Sonnet 3.7** (500K tokens) → **Claude Sonnet 4** (1M tokens) *when available*

#### Claude Family: Haiku Models
- **Claude Haiku 3.5** (200K tokens) → **Claude Haiku 4** (1M tokens) *when available*

### Test Context Sizes (Within Smaller Model's Capacity)
- **GPT-4 vs GPT-4 Turbo**: 2K, 4K, 6K, 8K tokens
- **GPT-4o vs GPT-4.1**: 25K, 50K, 100K, 128K tokens  
- **Claude Sonnet 3.5 vs 4**: 50K, 100K, 150K, 200K tokens

## Key Metrics Framework

### Needle-in-Haystack Metrics
- **Retrieval Accuracy**: % correct needle extraction
- **Position Sensitivity**: Performance variance by needle location  
- **Multi-hop Success**: Ability to connect distributed information
- **Domain Transfer**: Cross-domain reasoning capability

### LongMemEval Metrics
- **Context-Length Performance**: Accuracy by context size group
- **Question Complexity Interaction**: How difficulty scales with context
- **Consistency**: Performance stability within context groups
- **Efficiency**: Performance per token utilized

## Experimental Controls

### Content Controls
- **Identical Base Corpus**: Same source documents across all tests
- **Balanced Domain Distribution**: Equal PG/ArXiv representation in haystacks
- **Consistent Tokenization**: Same encoding across all models
- **Shuffling Variants**: Test both ordered and randomized content

### Procedural Controls  
- **Multiple Iterations**: 3-5 runs per scenario for statistical significance
- **Temperature = 0**: Eliminate sampling randomness
- **Standardized Prompting**: Identical prompts across all models
- **Token Budget Matching**: Exact context size control

## Key Hypotheses

### H1: Context Window Efficiency Hypothesis
Models trained on larger contexts are more efficient at processing smaller contexts due to better attention mechanisms and position encoding.

### H2: Attention Degradation Hypothesis  
Models with smaller max context windows maintain better attention quality within their operating range.

### H3: Training Distribution Hypothesis
Models see different context length distributions during training, affecting performance at specific sizes.

## Expected Insights

### Cross-Validation Benefits
- **Synthetic → Real-World**: Validate controlled findings on authentic tasks
- **Real-World → Synthetic**: Identify patterns to explore in controlled settings
- **Convergent Evidence**: Stronger conclusions from multiple methodologies

### Research Contributions
- **Model Selection Guidance**: When to choose high vs medium capacity models
- **Training Insights**: How context capacity affects sub-capacity performance  
- **Architecture Understanding**: Attention mechanism efficiency patterns
- **Practical Applications**: Context window utilization strategies

## Implementation Timeline

### Phase 1: Infrastructure (Week 1)
- Set up experiment framework
- Implement model interfaces
- Create dataset construction pipelines

### Phase 2: Needle Experiments (Week 2)  
- Generate synthetic test cases
- Run controlled needle-in-haystack experiments
- Initial performance analysis

### Phase 3: LongMemEval Analysis (Week 3)
- Group questions by context requirements
- Run real-world validation experiments
- Cross-validate findings

### Phase 4: Analysis & Documentation (Week 4)
- Combined statistical analysis
- Visualization and reporting
- Research paper preparation

## Quick Start

### Prerequisites
```bash
# Ensure you have the required dependencies
pip install openai pandas matplotlib seaborn tiktoken numpy scipy

# Set your ORQ API key
export ORQ_API_KEY="your_orq_api_key_here"
```

### Running the Experiments

1. **Generate Haystacks**:
   ```bash
   python needle_haystack/haystack_builder.py --target-sizes 10000 50000 100000
   ```

2. **Create Synthetic Needles**:
   ```bash
   python needle_haystack/needle_generator.py --num-needles 50 --domains pg,arxiv
   ```

3. **Run Needle Experiments**:
   ```bash
   python context_advantage_experiment.py --experiment-type needle --context-sizes 10000,50000,100000
   ```

4. **Analyze LongMemEval**:
   ```bash
   python longmemeval_analysis/context_grouper.py --input-path ../data/longmemeval/
   ```

5. **Run LongMemEval Experiments**:
   ```bash
   python context_advantage_experiment.py --experiment-type longmemeval
   ```

6. **Generate Analysis**:
   ```bash
   python shared/visualization.py --results-dir results/ --output-dir results/combined_analysis/
   ```

This experiment provides both experimental control and real-world validation, addressing the core research question from multiple complementary angles.

## Commands
```python scripts/longmemeval_ground_truth_extraction.py --query-strategy question --primary-metric token_containment --max-chunks 10 --force-reingest --embedding-model sentence-transformer --limit-ingestion 100 --limit-extraction 100```