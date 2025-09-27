"""
LongMemEval Dataset Module

Handles LongMemEval dataset processing, including:
- Conversation chunking with overlapping windows
- ChromaDB ingestion for similarity search
- Ground truth chunk relevance extraction using focused vs full conversations
- Content overlap metrics (Jaccard similarity, ROUGE-L, etc.)

Dataset Information:
- Source: LongMemEval conversations dataset  
- Format: CSV files with focused (relevant) and full (with distractors) conversations
- Usage: Long-context retrieval evaluation and ground truth generation

Components:
- GroundTruthExtractor: Main extraction orchestrator
- ConversationChunker: Chunks conversations with configurable overlap
- ContentOverlapMetrics: Calculates overlap between focused and chunk content
"""

from .ground_truth_extraction import (
    GroundTruthExtractor,
    ConversationChunker, 
    ContentOverlapMetrics,
    LongMemEvalPromptCleaner
)

__all__ = [
    'GroundTruthExtractor',
    'ConversationChunker',
    'ContentOverlapMetrics',
    'LongMemEvalPromptCleaner'
]