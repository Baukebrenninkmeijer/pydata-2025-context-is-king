# ORQ Proxy Integration

The reranking value experiment has been updated to properly use the ORQ proxy with the OpenAI SDK, following the existing codebase patterns.

## Key Changes Made

### 1. QueryRewriter Updated
- **Before**: Used custom `GenerationInterface` 
- **After**: Uses `ModelInterface` with OpenAI SDK + ORQ proxy
- **Benefits**: Consistent API access, rate limiting, unified error handling

```python
# Updated usage
from context_is_king.models import ModelInterface
from context_is_king.rag_pipeline.query_enhancement import QueryRewriter

model_interface = ModelInterface()  # Uses ORQ proxy automatically
query_rewriter = QueryRewriter(model_interface)
```

### 2. JudgeEvaluator Integration  
- Already correctly configured to use `ModelInterface`
- Uses OpenAI SDK with ORQ proxy for LLM-as-a-judge evaluation
- Maintains structured True/False + confidence + reasoning output

### 3. Experiment Controller
- **Unified Model Interface**: Single `ModelInterface` instance shared across components
- **ORQ Proxy**: All LLM calls route through `https://api.orq.ai/v2/proxy`
- **Model Names**: Uses ORQ format (e.g., `openai/gpt-4-turbo`, `anthropic/claude-3-sonnet-20240229`)

### 4. DualRetrieval Enhancement
- Automatically initializes `QueryRewriter` with proper model interface
- Maintains all fusion strategies (RRF, concatenation, interleaving)
- Seamlessly integrates with existing RAG pipeline

## ORQ Proxy Configuration

### Environment Setup
```bash
export ORQ_API_KEY="your_orq_api_key_here"
```

### Model Access
The system automatically uses:
- **Base URL**: `https://api.orq.ai/v2/proxy`
- **Rate Limiting**: 5 retries with exponential backoff
- **Error Handling**: Unified error parsing and retry logic
- **Token Counting**: GPT-4 encoding for consistent measurement

### Available Models
The experiment can use any model available through ORQ:
- `openai/gpt-4-turbo`
- `openai/gpt-4`  
- `anthropic/claude-3-sonnet-20240229`
- `anthropic/claude-3-haiku-20240307`
- And many others through the ORQ model catalog

## Benefits of ORQ Integration

### 1. **Unified Access**
- Single API key for multiple providers
- Consistent interface across OpenAI, Anthropic, Google, etc.
- No need for separate API configurations

### 2. **Rate Limiting & Reliability**
- Built-in rate limiting with intelligent backoff
- Automatic retry logic for transient failures
- Error parsing for different provider formats

### 3. **Cost Tracking**
- Centralized usage tracking through ORQ dashboard
- Cost analysis across different model providers
- Token usage monitoring

### 4. **Model Flexibility**
- Easy switching between models from different providers
- A/B testing across providers
- Fallback model support

## Usage Examples

### Basic Query Rewriting
```python
from context_is_king.models import ModelInterface
from context_is_king.rag_pipeline.query_enhancement import QueryRewriter

model_interface = ModelInterface()
rewriter = QueryRewriter(model_interface)

result = rewriter.rewrite_query(
    "What are the benefits of machine learning?",
    model="openai/gpt-4-turbo"  # ORQ model name
)
```

### Judge Evaluation
```python
from context_is_king.evaluation import JudgeEvaluator

judge = JudgeEvaluator(model_interface, judge_model="openai/gpt-4")

evaluation = judge.evaluate_answer(
    question="What is the capital of France?",
    expected_answer="Paris", 
    model_answer="The capital of France is Paris.",
    approach_used="enhanced_rag_rerank"
)
```

### Full Experiment
```bash
python run_experiment.py \
  --models openai/gpt-4-turbo anthropic/claude-3-sonnet-20240229 \
  --judge-model openai/gpt-4 \
  --context-groups short medium long
```

## Testing
All components have been tested and verified to work with the ORQ proxy:
- ✅ Component imports
- ✅ Model interface initialization  
- ✅ Query rewriting functionality
- ✅ Judge evaluation system
- ✅ Context tracking
- ✅ Analysis frameworks

The experiment is now ready for use with proper ORQ integration!