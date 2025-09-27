"""
Context assembly engine with multi-modal strategies and token management
"""

from typing import Optional, Any, Union, List, Dict
from pydantic import BaseModel, Field

import tiktoken
from rich.console import Console

from ..types import Document, DocumentChunk, RetrievalResult, PipelineConfig

console = Console()


class AssembledContext(BaseModel):
    """Container for assembled context with metadata"""
    context: str
    context_type: str  # 'rag', 'full', 'focused', 'conversation'
    token_count: int
    chunks_used: list[str]  # chunk IDs used
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextAssembly:
    """Assembles retrieved chunks into formatted context with token management"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Context templates
        self.templates = {
            "rag": """Based on the following relevant information, please answer the question.

Relevant Information:
{context}

Question: {question}

Please provide a clear and accurate answer based on the information provided.""",

            "full": """Based on the following document, please answer the question.

Document:
{context}

Question: {question}

Please provide a clear and accurate answer based on the document.""",

            "focused": """Based on the following recent information, please answer the question.

Recent Information:
{context}

Question: {question}

Please provide a clear and accurate answer based on the recent information.""",

            "conversation": """Based on the following conversation history, please answer the question.

Conversation History:
{context}

Current Question: {question}

Please provide a clear and accurate answer based on the conversation context."""
        }
        
        console.print(f"[green]ContextAssembly initialized with max context length: {config.max_context_length}[/green]")
    
    def assemble_rag_context(self, 
                           retrieved_chunks: list[RetrievalResult], 
                           query: str,
                           format_template: str = None,
                           max_chunks: int = None) -> AssembledContext:
        """
        Assemble RAG context from retrieved chunks
        
        Args:
            retrieved_chunks: List of retrieved results
            query: Original query
            format_template: Custom template (defaults to 'rag')
            max_chunks: Maximum number of chunks to include
            
        Returns:
            AssembledContext with formatted context
        """
        template = format_template or "rag"
        
        if not retrieved_chunks:
            return AssembledContext(
                context="No relevant information found.",
                context_type="rag",
                token_count=self._count_tokens("No relevant information found."),
                chunks_used=[],
                metadata={"query": query, "template": template}
            )
        
        # Sort chunks by rank and optionally limit
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.rank)
        if max_chunks:
            sorted_chunks = sorted_chunks[:max_chunks]
        
        # Build context from chunks
        context_parts = []
        chunk_ids = []
        
        for i, result in enumerate(sorted_chunks, 1):
            chunk_text = result.chunk.content.strip()
            
            # Add chunk with source attribution
            chunk_section = f"[{i}] {chunk_text}"
            if result.chunk.metadata.get('doc_id'):
                chunk_section += f"\n   (Source: {result.chunk.metadata['doc_id']})"
            
            context_parts.append(chunk_section)
            chunk_ids.append(result.chunk.full_id)
        
        raw_context = "\n\n".join(context_parts)
        
        # Apply token management
        managed_context, truncated = self._manage_context_window(
            raw_context, query, template
        )
        
        # Format with template
        formatted_context = self.templates[template].format(
            context=managed_context,
            question=query
        )
        
        return AssembledContext(
            context=formatted_context,
            context_type="rag",
            token_count=self._count_tokens(formatted_context),
            chunks_used=chunk_ids,
            truncated=truncated,
            metadata={
                "query": query,
                "template": template,
                "chunks_count": len(sorted_chunks),
                "avg_similarity": sum(r.similarity_score for r in sorted_chunks) / len(sorted_chunks),
                "reranked": any(r.reranked for r in sorted_chunks)
            }
        )
    
    def assemble_full_context(self, 
                            document: Document, 
                            query: str,
                            format_template: str = None) -> AssembledContext:
        """
        Assemble full document context
        
        Args:
            document: Source document
            query: Query/question
            format_template: Template to use (defaults to 'full')
            
        Returns:
            AssembledContext with full document
        """
        template = format_template or "full"
        
        # Apply token management
        managed_content, truncated = self._manage_context_window(
            document.content, query, template
        )
        
        # Format with template
        formatted_context = self.templates[template].format(
            context=managed_content,
            question=query
        )
        
        return AssembledContext(
            context=formatted_context,
            context_type="full",
            token_count=self._count_tokens(formatted_context),
            chunks_used=[f"full_doc_{document.doc_id}"],
            truncated=truncated,
            metadata={
                "query": query,
                "template": template,
                "doc_id": document.doc_id,
                "doc_type": document.doc_type,
                "original_length": len(document.content)
            }
        )
    
    def assemble_focused_context(self, 
                               content_items: List[Union[str, DocumentChunk]],
                               query: str,
                               focus_strategy: str = "recent",
                               max_items: int = None) -> AssembledContext:
        """
        Assemble focused context from recent or relevant content
        
        Args:
            content_items: List of strings or DocumentChunks
            query: Query/question
            focus_strategy: 'recent', 'relevant', or 'mixed'
            max_items: Maximum items to include
            
        Returns:
            AssembledContext with focused content
        """
        if not content_items:
            return AssembledContext(
                context="No recent information available.",
                context_type="focused",
                token_count=self._count_tokens("No recent information available."),
                chunks_used=[],
                metadata={"query": query, "focus_strategy": focus_strategy}
            )
        
        # Apply focusing strategy
        focused_items = self._apply_focus_strategy(
            content_items, query, focus_strategy, max_items
        )
        
        # Build context
        context_parts = []
        item_ids = []
        
        for i, item in enumerate(focused_items, 1):
            if isinstance(item, str):
                context_parts.append(f"[{i}] {item}")
                item_ids.append(f"text_item_{i}")
            else:  # DocumentChunk
                context_parts.append(f"[{i}] {item.content}")
                item_ids.append(item.full_id)
        
        raw_context = "\n\n".join(context_parts)
        
        # Apply token management
        managed_context, truncated = self._manage_context_window(
            raw_context, query, "focused"
        )
        
        # Format with template
        formatted_context = self.templates["focused"].format(
            context=managed_context,
            question=query
        )
        
        return AssembledContext(
            context=formatted_context,
            context_type="focused",
            token_count=self._count_tokens(formatted_context),
            chunks_used=item_ids,
            truncated=truncated,
            metadata={
                "query": query,
                "focus_strategy": focus_strategy,
                "items_count": len(focused_items),
                "original_items_count": len(content_items)
            }
        )
    
    def assemble_conversation_context(self, 
                                    conversation_turns: List[Dict[str, str]],
                                    current_query: str,
                                    max_turns: int = None) -> AssembledContext:
        """
        Assemble conversation context from conversation history
        
        Args:
            conversation_turns: List of conversation turns with 'speaker' and 'content'
            current_query: Current question
            max_turns: Maximum number of turns to include
            
        Returns:
            AssembledContext with conversation history
        """
        if not conversation_turns:
            return AssembledContext(
                context="No conversation history available.",
                context_type="conversation",
                token_count=self._count_tokens("No conversation history available."),
                chunks_used=[],
                metadata={"query": current_query}
            )
        
        # Limit turns if specified
        if max_turns:
            conversation_turns = conversation_turns[-max_turns:]
        
        # Build conversation context
        context_parts = []
        for i, turn in enumerate(conversation_turns):
            speaker = turn.get('speaker', f'Speaker_{i}')
            content = turn.get('content', '').strip()
            context_parts.append(f"{speaker}: {content}")
        
        raw_context = "\n".join(context_parts)
        
        # Apply token management
        managed_context, truncated = self._manage_context_window(
            raw_context, current_query, "conversation"
        )
        
        # Format with template
        formatted_context = self.templates["conversation"].format(
            context=managed_context,
            question=current_query
        )
        
        return AssembledContext(
            context=formatted_context,
            context_type="conversation",
            token_count=self._count_tokens(formatted_context),
            chunks_used=[f"turn_{i}" for i in range(len(conversation_turns))],
            truncated=truncated,
            metadata={
                "query": current_query,
                "turns_count": len(conversation_turns),
                "original_turns_count": len(conversation_turns)
            }
        )
    
    def calculate_context_stats(self, context: str) -> Dict[str, Any]:
        """Calculate comprehensive context statistics"""
        token_count = self._count_tokens(context)
        
        stats = {
            "character_count": len(context),
            "token_count": token_count,
            "word_count": len(context.split()),
            "line_count": len(context.split('\n')),
            "paragraph_count": len([p for p in context.split('\n\n') if p.strip()]),
            "utilization_ratio": token_count / self.config.max_context_length,
            "fits_in_context": token_count <= self.config.max_context_length
        }
        
        return stats
    
    def compare_context_strategies(self, 
                                 contexts: List[AssembledContext],
                                 query: str) -> Dict[str, Any]:
        """Compare multiple context assembly strategies"""
        if not contexts:
            return {"error": "No contexts to compare"}
        
        comparison = {
            "query": query,
            "strategies_compared": len(contexts),
            "contexts": []
        }
        
        for context in contexts:
            strategy_stats = {
                "context_type": context.context_type,
                "token_count": context.token_count,
                "chunks_count": len(context.chunks_used),
                "truncated": context.truncated,
                "utilization": context.token_count / self.config.max_context_length,
                "metadata": context.metadata
            }
            comparison["contexts"].append(strategy_stats)
        
        # Find optimal strategy (highest utilization without truncation)
        non_truncated = [c for c in comparison["contexts"] if not c["truncated"]]
        if non_truncated:
            optimal = max(non_truncated, key=lambda x: x["utilization"])
            comparison["recommended_strategy"] = optimal["context_type"]
        else:
            comparison["recommended_strategy"] = "all_truncated"
        
        return comparison
    
    def _manage_context_window(self, 
                             content: str, 
                             query: str, 
                             template: str) -> tuple[str, bool]:
        """Ensure context fits within token limits"""
        # Calculate template overhead
        template_text = self.templates[template].format(context="", question=query)
        template_tokens = self._count_tokens(template_text)
        
        # Available tokens for content
        available_tokens = self.config.max_context_length - template_tokens - 100  # Buffer
        
        content_tokens = self._count_tokens(content)
        
        if content_tokens <= available_tokens:
            return content, False
        
        # Need to truncate - preserve most relevant content
        console.print(f"[yellow]Context too long ({content_tokens} tokens), truncating to {available_tokens}[/yellow]")
        
        truncated_content = self._intelligent_truncate(content, available_tokens)
        return truncated_content, True
    
    def _intelligent_truncate(self, content: str, max_tokens: int) -> str:
        """Intelligently truncate content to fit token limit"""
        if max_tokens <= 0:
            return ""
        
        # Try to preserve paragraph boundaries
        paragraphs = content.split('\n\n')
        result_paragraphs = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            para_tokens = self._count_tokens(paragraph)
            
            if current_tokens + para_tokens <= max_tokens:
                result_paragraphs.append(paragraph)
                current_tokens += para_tokens
            else:
                # Try to fit partial paragraph
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:  # Only if we have reasonable space
                    # Truncate at sentence boundary if possible
                    sentences = paragraph.split('. ')
                    partial_para = ""
                    
                    for sentence in sentences:
                        sentence_with_period = sentence + ". "
                        sentence_tokens = self._count_tokens(sentence_with_period)
                        
                        if self._count_tokens(partial_para + sentence_with_period) <= remaining_tokens:
                            partial_para += sentence_with_period
                        else:
                            break
                    
                    if partial_para.strip():
                        result_paragraphs.append(partial_para.strip())
                break
        
        truncated = '\n\n'.join(result_paragraphs)
        
        # Add truncation indicator
        if truncated and len(truncated) < len(content):
            truncated += "\n\n[... content truncated ...]"
        
        return truncated
    
    def _apply_focus_strategy(self, 
                            items: List[Union[str, DocumentChunk]],
                            query: str,
                            strategy: str,
                            max_items: int = None) -> List[Union[str, DocumentChunk]]:
        """Apply focusing strategy to select most relevant items"""
        if strategy == "recent":
            # Take most recent items
            focused = items[-max_items:] if max_items else items
            
        elif strategy == "relevant":
            # Simple relevance scoring based on query term overlap
            scored_items = []
            query_terms = set(query.lower().split())
            
            for item in items:
                content = item.content if isinstance(item, DocumentChunk) else str(item)
                content_terms = set(content.lower().split())
                overlap = len(query_terms.intersection(content_terms))
                scored_items.append((overlap, item))
            
            # Sort by relevance score and take top items
            scored_items.sort(key=lambda x: x[0], reverse=True)
            focused = [item for _, item in scored_items[:max_items]] if max_items else [item for _, item in scored_items]
            
        elif strategy == "mixed":
            # Mix of recent and relevant
            half = (max_items // 2) if max_items else (len(items) // 2)
            recent = self._apply_focus_strategy(items, query, "recent", half)
            relevant = self._apply_focus_strategy(items, query, "relevant", max_items - len(recent) if max_items else None)
            
            # Combine and deduplicate
            seen = set()
            focused = []
            for item in recent + relevant:
                item_key = item.full_id if isinstance(item, DocumentChunk) else str(item)
                if item_key not in seen:
                    focused.append(item)
                    seen.add(item_key)
                    
        else:
            # Default to recent
            focused = items[-max_items:] if max_items else items
        
        return focused
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to approximate counting
            return len(text.split()) * 1.3  # Rough approximation
    
    def get_template(self, template_name: str) -> str:
        """Get context template by name"""
        return self.templates.get(template_name, self.templates["rag"])
    
    def add_custom_template(self, name: str, template: str) -> None:
        """Add a custom context template"""
        self.templates[name] = template
        console.print(f"[green]Added custom template: {name}[/green]")