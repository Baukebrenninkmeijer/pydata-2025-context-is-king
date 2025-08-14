# **Context vs RAG Experimental Implementation Guide**
*Complete specification for coding assistant implementation*

## **Project Overview**

### **Research Objective**
Compare long context window approaches against Retrieval Augmented Generation (RAG) for data grounding using identical datasets and evaluation methodologies to the Chroma Context Rot study. Answer 6 key research questions about when to use RAG vs long context.

### **Research Questions**
1. **Q1**: When must we switch from context window retrieval to RAG?
2. **Q2**: Can we skip RAG and immediately work with full context window when resources fit?
3. **Q3**: Is there a context window threshold where we observe a sharp dropoff in performance?
4. **Q4**: Does reranking still make sense in a post-RAG world?
5. **Q5**: If small context is still needed, how can we best utilize the effective context window?
6. **Q6**: With growing context windows, how does RAG performance change when we include more retrieved chunks?

### Important sources
Research follows the chroma contextrot research: https://research.trychroma.com/context-rot. Read this to understand what they did. 

---

## **Technical Specifications**

### **Technology Stack**
```python
TECH_STACK = {
    'vectorstore': 'chromadb',                    # Local persistent vector database
    'tabular_data': 'excel + polars',           # .xlsx for manual editing, polars for operations
    'embeddings': 'all-MiniLM-L6-v2',          # Local SentenceTransformer model
    'llm_generation': 'moonshotai/kimi-k2-instruct',  # NVIDIA API
    'llm_judge': 'moonshotai/kimi-k2-instruct',       # Same model for consistency
    'optimization': 'batching for all operations',     # Speed/efficiency
    'hardware': 'RTX 3080 (10GB) + MacBook (18GB)',
    'budget': '$0 (free NVIDIA API)',
    'rate_limit': '40 requests/minute'
}
```

### **Core Dependencies**
```python
# requirements.txt
chromadb>=0.4.0
polars>=0.20.0
sentence-transformers>=2.2.0
openpyxl>=3.1.0
pandas>=2.0.0  # For Excel writer compatibility
numpy>=1.24.0
torch>=2.0.0
requests>=2.28.0
pathlib
datetime
json
time
collections
```

### **Hardware Configuration**
- **RTX 3080**: Local embedding generation, batching for GPU efficiency
- **MacBook 18GB**: Data processing, API coordination, result storage
- **Storage**: ~50GB for datasets, embeddings, results

---

## **Data Sources (All Verified Available)**

### **Primary Datasets**
```python
DATASETS = {
    'chroma_context_rot': {
        'source': 'https://github.com/chroma-core/context-rot',
        'google_drive': 'https://drive.google.com/drive/folders/1FuOysriSotnYasJUbZJzn31SWt85_3yf',
        'contents': 'needles, distractors, evaluation data',
        'format': 'JSON, various'
    },
    'paul_graham_essays': {
        'source': 'https://huggingface.co/datasets/chromadb/paul_graham_essay',
        'alternative': 'https://github.com/ofou/graham-essays',
        'contents': 'Essay collection for haystack',
        'format': 'JSON/Markdown'
    },
    'arxiv_papers': {
        'source': 'https://www.kaggle.com/datasets/Cornell-University/arxiv',
        'subset': 'Information retrieval papers (1000 max for budget)',
        'contents': 'Academic papers for haystack',
        'format': 'JSON'
    },
    'longmemeval': {
        'source': 'https://github.com/xiaowu0162/LongMemEval',
        'huggingface': 'Alternative download source',
        'contents': 'Conversational QA dataset',
        'format': 'JSON'
    }
}
```

---

## **Implementation Architecture**

### **Phase 1: Environment Setup (3-4 days)**

#### **Directory Structure**
```
context-vs-rag-experiments/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Chunked, cleaned data
│   └── embeddings/             # Generated embeddings
├── src/
│   ├── setup/                  # Environment setup
│   ├── models/                 # Model interfaces
│   ├── experiments/            # Experiment implementations
│   ├── analysis/               # Analysis scripts  
│   └── utils/                  # Helper functions
├── experiment_results/
│   ├── calibration/            # Judge calibration data
│   ├── checkpoints/            # Experiment checkpoints
│   └── final_reports/          # Analysis outputs
├── chroma_db/                  # ChromaDB persistent storage
└── configs/                    # Configuration files
```

#### **Core Classes to Implement**

**1. BatchedEmbeddingGenerator**
```python
class BatchedEmbeddingGenerator:
    """Local embedding generation with RTX 3080 optimization"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        """
        Generate embeddings in batches for GPU efficiency
        
        Args:
            texts: List of text strings
            batch_size: Batch size for GPU processing (optimize for 10GB VRAM)
            
        Returns:
            List of embedding arrays
        """
        # IMPLEMENTATION NEEDED: Batch processing with progress tracking
        pass
    
    def save_embeddings(self, embeddings: List[np.ndarray], metadata: List[dict], filename: str):
        """Save embeddings with metadata for reuse"""
        # IMPLEMENTATION NEEDED: Efficient storage format
        pass
```

**2. RateLimitedNvidiaClient**
```python
class RateLimitedNvidiaClient:
    """NVIDIA API client with 40 requests/minute rate limiting"""
    
    def __init__(self, api_key: str, model: str = "moonshotai/kimi-k2-instruct"):
        self.api_key = api_key
        self.model = model
        self.max_rpm = 40
        self.request_times = deque()
        
    def make_request(self, messages: List[dict], **kwargs) -> dict:
        """Single API request with rate limiting"""
        # IMPLEMENTATION NEEDED: Rate limiting logic, error handling
        pass
    
    def batch_requests(self, requests: List[dict], batch_size: int = 10) -> List[dict]:
        """Process multiple requests with rate limiting"""
        # IMPLEMENTATION NEEDED: Batch processing with progress tracking
        pass
```

**3. ChromaDBManager**
```python
class ChromaDBManager:
    """ChromaDB interface for vector storage and retrieval"""
    
    def __init__(self, persist_directory: str = './chroma_db'):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}
        
    def create_collection(self, name: str, metadata: dict = None) -> chromadb.Collection:
        """Create or get collection with specified configuration"""
        # IMPLEMENTATION NEEDED: Collection management
        pass
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     embeddings: List[np.ndarray], metadata: List[dict]):
        """Add documents with embeddings to collection"""
        # IMPLEMENTATION NEEDED: Batch insertion, deduplication
        pass
    
    def query_collection(self, collection_name: str, query_embedding: np.ndarray, 
                        k: int = 10) -> dict:
        """Query collection for similar documents"""
        # IMPLEMENTATION NEEDED: Similarity search
        pass
```

**4. ExperimentPersistence**
```python
class ExperimentPersistence:
    """Handle Excel-based data persistence for easy manual editing"""
    
    def __init__(self, base_dir: str = './experiment_results'):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def save_calibration_data(self, samples: List[dict], filename: str):
        """Save calibration samples to Excel for manual labeling"""
        # IMPLEMENTATION NEEDED: DataFrame creation, Excel export
        pass
    
    def load_human_labels(self, filename: str) -> pl.DataFrame:
        """Load human-labeled data from Excel"""
        # IMPLEMENTATION NEEDED: Excel import, validation
        pass
    
    def save_experiment_results(self, experiment_name: str, results: List[dict]):
        """Save experiment results with checkpointing"""
        # IMPLEMENTATION NEEDED: Polars DataFrame operations, Excel export
        pass
    
    def load_checkpoint(self, experiment_name: str) -> List[dict]:
        """Resume from saved checkpoint"""
        # IMPLEMENTATION NEEDED: Checkpoint loading logic
        pass
```

#### **TASK REFINEMENT NEEDED**
- **NVIDIA API Integration**: Need to research exact API endpoints and authentication for `moonshotai/kimi-k2-instruct`
- **ChromaDB Configuration**: Determine optimal settings for cosine similarity, HNSW parameters
- **Batch Size Optimization**: Test optimal batch sizes for RTX 3080 memory constraints

---

### **Phase 2: Judge Implementation (Using Chroma's Exact Methodology - 0 calibration calls needed!)**

#### **Adopt Chroma's Pre-Validated Judge System**
Instead of manual calibration, use Chroma's proven judge implementation directly from their repository: `https://github.com/chroma-core/context-rot`

**Implementation Requirements:**
```python
class ChromaCompatibleJudge:
    """Implement Chroma's exact judge methodology with NVIDIA API backend"""
    
    def __init__(self, nvidia_api_key: str):
        # Adapt Chroma's LLMJudge to use NVIDIA API instead of OpenAI
        self.nvidia_client = RateLimitedNvidiaClient(nvidia_api_key)
        
        # NIAH Extension Judge Prompt (from evaluate_niah_extension.py)
        self.niah_prompt = """
        Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
        You must only respond with "true" or "false".
        If the response is partially incorrect, such as a typo, respond with "false".
        If the repsonse contains a snippet of text or additional supporting information, while still maintaining the correct answer without changing the meaning, respond with "true".
        If the response starts with anything like "here is the most relevant information in the documents: ", respond with "true". This is fine as long as the following content aligns with the correct answer.

        Question: {question}

        CORRECT answer: {correct_answer}

        Response to judge: {output}

        Instructions: Respond with only "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false".
        """
        
        # LongMemEval Judge Prompt (from evaluate_longmemeval.py)  
        self.longmemeval_prompt = """
        Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
        In some cases, 0 and "I do not have an answer" are considered to be both correct. 
        If both responses say that there is no answer, this should be judged as true.
        If the correct answer contains an answer, but the response abstains from answering, this should be judged as false.

        Question: {question}

        CORRECT answer: {correct_answer}

        Response to judge: {output}

        Instructions: Respond with only "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false".
        """
    
    def evaluate_niah_experiments(self, input_path: str, output_path: str) -> None:
        """Use NIAH-specific judge prompt for needle experiments"""
        # IMPLEMENTATION NEEDED: Adapt Chroma's evaluate() method for NVIDIA API
        pass
        
    def evaluate_longmemeval_experiments(self, input_path: str, output_path: str) -> None:
        """Use LongMemEval-specific judge prompt for conversational QA"""
        # IMPLEMENTATION NEEDED: Adapt Chroma's evaluate() method for NVIDIA API  
        pass
```

#### **Key Advantages of This Approach**
- **Zero calibration cost**: No manual labeling or API calls needed for judge setup
- **Perfect methodology alignment**: Identical evaluation criteria to Chroma study
- **Pre-validated prompts**: Chroma already tested and validated these judge prompts
- **Direct result comparability**: Can directly compare findings to Chroma's results

#### **TASK REFINEMENT NEEDED**
- **NVIDIA API Integration**: Adapt Chroma's OpenAI-based LLMJudge to work with NVIDIA API
- **Rate Limiting**: Implement 40 req/min limiting within judge evaluation
- **Prompt Selection**: Use appropriate prompt (NIAH vs LongMemEval) for each experiment type

---

### **Phase 3: Core Experiments (2 weeks, ~4,000 API calls)**
*Note: Gained 500+ extra API calls by eliminating manual judge calibration*

#### **Experiment Implementations Required**

**1. Needle-Question Similarity Experiment**
```python
def run_needle_similarity_experiment():
    """
    Test how RAG performance varies with needle-question semantic similarity
    
    Data: 8 needles with similarity scores 0.445-0.829 (from Chroma)
    Haystacks: Paul Graham essays, arXiv papers
    
    Metrics to collect:
    - Retrieval recall (needle found in retrieved chunks)
    - Generation accuracy (judge evaluation)
    - Context tokens used
    - API calls consumed
    """
    # IMPLEMENTATION NEEDED: Complete experiment logic
    pass
```

**2. Distractor Impact Experiment**
```python
def run_distractor_impact_experiment():
    """
    Test how distractors in RAG corpus affect performance
    
    Setup: High-similarity needle + 0, 1, or 4 distractors
    Measure: How distractors interfere with retrieval and generation
    """
    # IMPLEMENTATION NEEDED: Distractor insertion logic
    pass
```

**3. Cross-Domain Experiment**
```python
def run_cross_domain_experiment():
    """
    Test needle-haystack similarity effects (PG needles in arXiv corpus, etc.)
    """
    # IMPLEMENTATION NEEDED: Cross-domain testing logic
    pass
```

**4. Structure Impact Experiment**
```python
def run_structure_impact_experiment():
    """
    Test original vs shuffled document structure in RAG chunks
    """
    # IMPLEMENTATION NEEDED: Document shuffling logic
    pass
```

**5. LongMemEval Experiment**
```python
def run_longmemeval_experiment():
    """
    Compare RAG retrieval vs full/focused context for conversational QA
    """
    # IMPLEMENTATION NEEDED: Conversation history processing
    pass
```

#### **Two RAG Configurations**
```python
class RAGConfiguration:
    """Base RAG system with configurable components"""
    
    def __init__(self, use_reranker: bool = True):
        self.use_reranker = use_reranker
        self.retrieval_k = 10
        self.chunk_size = 1000
        self.chunk_overlap = 100
        
    def process_query(self, query: str, collection_name: str) -> dict:
        """
        Full RAG pipeline:
        1. Query embedding
        2. Vector similarity search
        3. Optional reranking
        4. Context formatting
        5. LLM generation
        """
        # IMPLEMENTATION NEEDED: Complete RAG pipeline
        pass

# Config A: With reranker
rag_with_reranker = RAGConfiguration(use_reranker=True)

# Config B: Without reranker  
rag_without_reranker = RAGConfiguration(use_reranker=False)
```

#### **TASK REFINEMENT NEEDED**
- **Chunking Strategy**: Implement RecursiveCharacterTextSplitter with 200 character chunks and 40 cahracters overlap.
- **Reranker Implementation**: Research and implement cross-encoder reranking. This can run either locally or on the GPU.
- **Context Window Management**: How to handle different context sizes for comparison

---

### **Phase 4: Analysis & Results (3-4 days, ~500 API calls)**

#### **Analysis Framework**
```python
class ExperimentAnalysis:
    """Analyze results to answer research questions"""
    
    def analyze_research_questions(self, all_results: dict) -> dict:
        """
        Map experimental results to research questions
        
        Q1: When to switch context window to RAG?
        Q2: Can we skip RAG when resources fit?
        Q3: Context window threshold for dropoff?
        Q4: Does reranking still make sense?
        Q5: How to best utilize context window?
        Q6: RAG performance with more chunks?
        """
        # IMPLEMENTATION NEEDED: Statistical analysis for each question
        pass
    
    def find_crossover_points(self, results: pl.DataFrame) -> dict:
        """Identify performance crossover points between RAG and long context"""
        # IMPLEMENTATION NEEDED: Crossover analysis
        pass
    
    def optimal_configuration_analysis(self, results: pl.DataFrame) -> dict:
        """Find Pareto-optimal RAG configurations"""
        # IMPLEMENTATION NEEDED: Multi-objective optimization
        pass
```

#### **Report Generation**
```python
def generate_final_report(analysis_results: dict):
    """
    Create comprehensive Excel report with:
    - Executive summary
    - Research question answers
    - Optimal configurations
    - Statistical significance tests
    - Visualizations (charts)
    """
    # IMPLEMENTATION NEEDED: Multi-sheet Excel report generation
    pass
```

#### **TASK REFINEMENT NEEDED**
- **Statistical Methods**: Determine appropriate significance tests for non-parametric data
- **Visualization**: Create informative charts for decision-making
- **Practical Recommendations**: Translate statistical findings into actionable insights

---

## **Implementation Priorities**

### **Critical Path (Must Implement First)**
1. **Environment Setup**: ChromaDB, NVIDIA API, local embeddings
2. **Judge Calibration**: Human labeling + iterative prompt improvement
3. **Basic RAG Pipeline**: Embedding → retrieval → generation
4. **Experiment Framework**: Batching, checkpoints, persistence

### **Secondary Implementation**
1. **All 5 experiments**: Following Chroma methodology exactly
2. **Analysis pipeline**: Statistical analysis, report generation
3. **Optimization**: Batch processing, error handling

### **Optional (Budget Permitting)**
1. **Reranker implementation**: Cross-encoder for Config A
2. **Advanced visualizations**: Interactive charts, dashboards
3. **Extended analysis**: Additional statistical tests

---

## **Risk Factors & Mitigation**

### **Technical Risks**
1. **Rate Limiting**: NVIDIA API 40 req/min → Implement careful batching
2. **GPU Memory**: RTX 3080 10GB → Optimize batch sizes
3. **Judge Alignment**: Target >95% → Allocate time for iteration

### **Budget Risks**
1. **API Overuse**: Monitor call counts carefully
2. **Storage**: Large embeddings → Implement compression
3. **Time Overrun**: Checkpoint everything for resumption

### **Quality Risks**
1. **Sample Size**: Reduced vs Chroma → Focus on high-impact experiments
2. **Model Differences**: Kimi vs GPT-4 → Document limitations
3. **Human Bias**: Manual labeling → Use clear criteria

---

## **Success Criteria**

### **Technical Milestones**
- [ ] Judge calibrated to >95% human alignment
- [ ] All 5 experiments executed successfully
- [ ] Results comparable to Chroma methodology
- [ ] Complete analysis of all 6 research questions

### **Research Outcomes**
- [ ] Clear decision boundaries for RAG vs long context
- [ ] Optimal RAG configurations identified
- [ ] Statistical significance for key findings
- [ ] Actionable recommendations for practitioners

### **Deliverables**
- [ ] Complete codebase with documentation
- [ ] Excel-based results and analysis
- [ ] Comprehensive final report
- [ ] Reproducibility package

---

## **Next Steps for Implementation**

1. **Start with Phase 1**: Set up environment, download datasets
2. **Implement core classes**: Focus on batching and persistence first
3. **Calibrate judge**: This is critical - allocate sufficient time
4. **Build incrementally**: One experiment at a time with checkpoints
5. **Monitor resources**: Track API calls and GPU usage carefully

This document provides complete specifications for implementing the experimental plan. The coding assistant should focus on the critical path first, implementing robust batching and persistence systems before moving to the experimental logic.