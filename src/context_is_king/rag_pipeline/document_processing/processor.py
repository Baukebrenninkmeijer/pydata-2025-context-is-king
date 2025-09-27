"""
Document processing pipeline with chunking, shuffling, and distractor injection
"""

import json
import random
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import track

from ..types import Document, DocumentChunk, PipelineConfig

console = Console()


class DocumentProcessor:
    """Handles document ingestion, chunking, and preprocessing"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_document(self, file_path: str) -> Document:
        """Load a single document from file"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with open(path, encoding="utf-8") as f:
            content = f.read().strip()

        # Determine document type from path
        doc_type = self._infer_doc_type(path)

        return Document(
            content=content,
            doc_id=path.stem,
            source=str(path),
            doc_type=doc_type,
            metadata={"file_size": len(content), "file_path": str(path), "original_structure": True},
        )

    def load_documents(self, file_paths: list[str]) -> list[Document]:
        """Load multiple documents"""
        documents = []
        for path in track(file_paths, description="Loading documents..."):
            try:
                doc = self.load_document(path)
                documents.append(doc)
            except Exception as e:
                console.print(f"[red]Failed to load {path}: {e}[/red]")

        console.print(f"[green]Loaded {len(documents)} documents[/green]")
        return documents

    def process_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """Process documents into chunks with metadata"""
        all_chunks = []

        for doc in track(documents, description="Processing documents into chunks..."):
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)

        console.print(f"[green]Created {len(all_chunks)} chunks from {len(documents)} documents[/green]")
        return all_chunks

    def shuffle_document(self, document: Document, intensity: str = "moderate") -> Document:
        """
        Shuffle document structure for structure impact experiments

        Args:
            document: Source document to shuffle
            intensity: 'light', 'moderate', or 'heavy'

        Returns:
            New document with shuffled structure
        """
        paragraphs = self._extract_paragraphs(document.content)

        if intensity == "light":
            # Shuffle within sections, preserve section order
            shuffled_paragraphs = self._light_shuffle(paragraphs)
        elif intensity == "moderate":
            # Shuffle paragraphs completely, keep section headers
            shuffled_paragraphs = self._moderate_shuffle(paragraphs)
        elif intensity == "heavy":
            # Shuffle everything including breaking section boundaries
            shuffled_paragraphs = self._heavy_shuffle(paragraphs)
        else:
            raise ValueError(f"Unknown shuffle intensity: {intensity}")

        shuffled_content = "\n\n".join(shuffled_paragraphs)

        # Create new document with shuffled content
        shuffled_doc = Document(
            content=shuffled_content,
            doc_id=f"{document.doc_id}_shuffled_{intensity}",
            source=document.source,
            doc_type=document.doc_type,
            metadata={
                **document.metadata,
                "shuffled": True,
                "shuffle_intensity": intensity,
                "original_doc_id": document.doc_id,
                "original_structure": False,
            },
        )

        return shuffled_doc

    def inject_distractors(self, document: Document, distractors: list[str], count: int) -> Document:
        """
        Inject distractors for distractor impact experiments

        Args:
            document: Base document
            distractors: List of distractor texts
            count: Number of distractors to inject (0, 1, or 4)

        Returns:
            New document with distractors injected
        """
        if count == 0:
            return document

        if count > len(distractors):
            raise ValueError(f"Requested {count} distractors but only {len(distractors)} available")

        # Select random distractors
        selected_distractors = random.sample(distractors, count)

        # Insert distractors at random positions in the document
        paragraphs = self._extract_paragraphs(document.content)

        # Choose random insertion points
        insertion_points = sorted(random.sample(range(len(paragraphs) + 1), count))

        # Insert distractors from back to front to maintain indices
        for i, (insertion_point, distractor) in enumerate(
            zip(reversed(insertion_points), reversed(selected_distractors), strict=False)
        ):
            paragraphs.insert(insertion_point, f"[DISTRACTOR_{count - i}] {distractor}")

        distractor_content = "\n\n".join(paragraphs)

        # Create new document with distractors
        distractor_doc = Document(
            content=distractor_content,
            doc_id=f"{document.doc_id}_distractors_{count}",
            source=document.source,
            doc_type=document.doc_type,
            metadata={
                **document.metadata,
                "has_distractors": True,
                "distractor_count": count,
                "distractor_positions": insertion_points,
                "original_doc_id": document.doc_id,
            },
        )

        return distractor_doc

    def load_chroma_needles(self, needles_file: str) -> list[dict[str, Any]]:
        """Load needles from Chroma Context Rot dataset"""
        with open(needles_file, encoding="utf-8") as f:
            needles_data = json.load(f)

        console.print(f"[green]Loaded {len(needles_data)} needles from Chroma dataset[/green]")
        return needles_data

    def load_chroma_distractors(self, distractors_file: str) -> list[str]:
        """Load distractors from Chroma Context Rot dataset"""
        with open(distractors_file, encoding="utf-8") as f:
            distractors_data = json.load(f)

        # Extract distractor texts
        distractors = []
        for key in sorted(distractors_data.keys()):
            distractor_text = distractors_data[key]["rewrite_for_analysis"]
            distractors.append(distractor_text)

        console.print(f"[green]Loaded {len(distractors)} distractors from Chroma dataset[/green]")
        return distractors

    def save_processed_documents(self, documents: list[Document], output_dir: str) -> None:
        """Save processed documents to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            file_path = output_path / f"{doc.doc_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc.model_dump(), f, indent=2, ensure_ascii=False)

        console.print(f"[green]Saved {len(documents)} processed documents to {output_dir}[/green]")

    def _chunk_document(self, document: Document) -> list[DocumentChunk]:
        """Split document into chunks"""
        texts = self.text_splitter.split_text(document.content)
        chunks = []

        current_pos = 0
        for i, text in enumerate(texts):
            # Find the start position of this chunk in the original content
            start_char = document.content.find(text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(text)

            chunk = DocumentChunk(
                content=text,
                doc_id=document.doc_id,
                chunk_id=f"chunk_{i:04d}",
                start_char=start_char,
                end_char=end_char,
                metadata={**document.metadata, "chunk_index": i, "chunk_length": len(text)},
            )
            chunks.append(chunk)
            current_pos = end_char

        return chunks

    def _extract_paragraphs(self, content: str) -> list[str]:
        """Extract paragraphs from document content"""
        # Split on double newlines and clean up
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return paragraphs

    def _light_shuffle(self, paragraphs: list[str]) -> list[str]:
        """Light shuffling: shuffle within sections, preserve section order"""
        # Simple implementation: shuffle consecutive groups of 3-5 paragraphs
        shuffled = []
        i = 0
        while i < len(paragraphs):
            group_size = min(random.randint(3, 5), len(paragraphs) - i)
            group = paragraphs[i : i + group_size]
            random.shuffle(group)
            shuffled.extend(group)
            i += group_size
        return shuffled

    def _moderate_shuffle(self, paragraphs: list[str]) -> list[str]:
        """Moderate shuffling: shuffle paragraphs completely, keep section headers"""
        # Identify potential section headers (short lines, all caps, etc.)
        headers = []
        content_paras = []

        for para in paragraphs:
            if len(para) < 50 and (para.isupper() or para.startswith("#") or para.endswith(":")):
                headers.append((len(content_paras), para))
            else:
                content_paras.append(para)

        # Shuffle content paragraphs
        random.shuffle(content_paras)

        # Reinsert headers at appropriate positions
        result = content_paras.copy()
        for pos, header in reversed(headers):
            if pos < len(result):
                result.insert(pos, header)
            else:
                result.append(header)

        return result

    def _heavy_shuffle(self, paragraphs: list[str]) -> list[str]:
        """Heavy shuffling: shuffle everything including breaking section boundaries"""
        shuffled = paragraphs.copy()
        random.shuffle(shuffled)
        return shuffled

    def _infer_doc_type(self, path: Path) -> str:
        """Infer document type from file path"""
        path_str = str(path).lower()

        if "paul_graham" in path_str or "pg_" in path_str:
            return "paul_graham"
        if "arxiv" in path_str:
            return "arxiv"
        if "conversation" in path_str or "longmemeval" in path_str:
            return "conversation"
        return "unknown"
