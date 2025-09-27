"""
Reranking Value Experiment

Research question: "Does reranking still make sense in a post-RAG world?"

This experiment compares three approaches:
1. Full Context Retrieval - Load complete documents into context
2. Enhanced RAG without Reranking - Query rewriting + dual retrieval + semantic ranking  
3. Enhanced RAG with Reranking - Query rewriting + dual retrieval + cross-encoder reranking

The experiment uses LongMemEval questions across different complexity levels
and provides comprehensive evaluation using LLM-as-a-judge with structured output.
"""

__version__ = "1.0.0"
__author__ = "PyData 2025 Context is King Project"