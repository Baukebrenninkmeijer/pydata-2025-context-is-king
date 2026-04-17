#!/usr/bin/env python3
"""
Haystack Builder for Context Window Advantage Experiments

This module builds haystacks (large text contexts) by concatenating documents
to reach specific token targets. Supports different composition strategies:
- PG-heavy: Mostly Paul Graham essays with some ArXiv papers
- ArXiv-heavy: Mostly ArXiv papers with some PG essays
- Mixed: Balanced combination of both domains

Usage:
    python haystack_builder.py --target-sizes 10000 50000 100000
    python haystack_builder.py --composition mixed --target-sizes 25000 --shuffle
"""

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import tiktoken


@dataclass
class Document:
    """Represents a source document with metadata."""

    content: str
    filename: str
    domain: str  # 'pg' or 'arxiv'
    token_count: int


@dataclass
class Haystack:
    """Represents a constructed haystack with metadata."""

    content: str
    token_count: int
    composition: str  # 'pg_heavy', 'arxiv_heavy', 'mixed'
    target_size: int
    documents_used: list[str]
    shuffled: bool
    needle_positions: list[int]  # Will be populated by needle_generator


class HaystackBuilder:
    """Builds haystacks by concatenating documents to reach token targets."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize the haystack builder."""
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            data_dir = project_root / "data"

        self.data_dir = Path(data_dir)
        self.encoding = tiktoken.encoding_for_model("gpt-4o")

        # Load source documents
        self.pg_documents = self._load_pg_documents()
        self.arxiv_documents = self._load_arxiv_documents()

        print(f"📚 Loaded {len(self.pg_documents)} PG essays, {len(self.arxiv_documents)} ArXiv papers")

    def _load_pg_documents(self) -> list[Document]:
        """Load Paul Graham essays."""
        documents = []
        pg_path = self.data_dir / "paul_graham_essays"

        if not pg_path.exists():
            print(f"⚠️  Paul Graham essays not found at {pg_path}")
            return documents

        for txt_file in pg_path.glob("*.txt"):
            try:
                with open(txt_file, encoding="utf-8") as f:
                    content = f.read()

                token_count = len(self.encoding.encode(content))
                documents.append(
                    Document(content=content, filename=txt_file.name, domain="pg", token_count=token_count)
                )
            except Exception as e:
                print(f"⚠️  Error reading {txt_file}: {e}")

        return documents

    def _load_arxiv_documents(self) -> list[Document]:
        """Load ArXiv papers."""
        documents = []
        arxiv_path = self.data_dir / "arxiv_papers"

        if not arxiv_path.exists():
            print(f"⚠️  ArXiv papers not found at {arxiv_path}")
            return documents

        for txt_file in arxiv_path.glob("*.txt"):
            try:
                with open(txt_file, encoding="utf-8") as f:
                    content = f.read()

                token_count = len(self.encoding.encode(content))
                documents.append(
                    Document(content=content, filename=txt_file.name, domain="arxiv", token_count=token_count)
                )
            except Exception as e:
                print(f"⚠️  Error reading {txt_file}: {e}")

        return documents

    def build_haystack(
        self, target_size: int, composition: str = "mixed", shuffle: bool = False, buffer_tokens: int = 500
    ) -> Haystack:
        """
        Build a haystack with target token count.

        Args:
            target_size: Target token count
            composition: 'pg_heavy', 'arxiv_heavy', or 'mixed'
            shuffle: Whether to shuffle sentence order
            buffer_tokens: Allowed deviation from target size

        Returns:
            Haystack object with content and metadata
        """
        print(f"🏗️  Building {composition} haystack targeting {target_size:,} tokens...")

        # Select documents based on composition strategy
        selected_docs = self._select_documents(target_size, composition)

        if not selected_docs:
            raise ValueError(f"No documents available for composition: {composition}")

        # Concatenate documents
        content_parts = []
        total_tokens = 0
        documents_used = []

        for doc in selected_docs:
            # Check if adding this document would exceed our target
            if total_tokens + doc.token_count > target_size + buffer_tokens:
                # Try to find a smaller document
                remaining_tokens = target_size - total_tokens
                # print(f'{remaining_tokens=}')
                available_ratio = remaining_tokens / doc.token_count
                # print(f'{available_ratio=}')
                updated_doc  = copy.deepcopy(doc)
                # print(f'{len(doc.content)}')
                updated_doc.content = updated_doc.content[:int(len(updated_doc.content)*available_ratio)]
                # print(f'{len(updated_doc.content)}')
                content_parts.append(updated_doc.content)
                total_tokens += updated_doc.token_count
                documents_used.append(updated_doc.filename)
            else:
                content_parts.append(doc.content)
                total_tokens += doc.token_count
                documents_used.append(doc.filename)

            # Stop if we've reached our target
            if total_tokens >= target_size - buffer_tokens:
                break

        # Join content with document separators
        full_content = "\n\n---\n\n".join(content_parts)

        # Apply shuffling if requested
        if shuffle:
            full_content = self._shuffle_content(full_content)

        # Final token count verification
        actual_tokens = len(self.encoding.encode(full_content))

        print(f"✅ Built haystack: {actual_tokens:,} tokens ({len(content_parts)} documents)")

        return Haystack(
            content=full_content,
            token_count=actual_tokens,
            composition=composition,
            target_size=target_size,
            documents_used=documents_used,
            shuffled=shuffle,
            needle_positions=[],  # Will be populated later
        )

    def _select_documents(self, target_size: int, composition: str) -> list[Document]:
        """Select documents based on composition strategy."""
        if composition == "pg_heavy":
            # 70% PG, 30% ArXiv
            pg_target = int(target_size * 0.7)
            arxiv_target = target_size - pg_target
            selected = self._select_by_tokens(self.pg_documents, pg_target) + self._select_by_tokens(
                self.arxiv_documents, arxiv_target
            )
        elif composition == "arxiv_heavy":
            # 70% ArXiv, 30% PG
            arxiv_target = int(target_size * 0.7)
            pg_target = target_size - arxiv_target
            selected = self._select_by_tokens(self.arxiv_documents, arxiv_target) + self._select_by_tokens(
                self.pg_documents, pg_target
            )
        elif composition == "mixed":
            # 50% each
            half_target = target_size // 2
            selected = self._select_by_tokens(self.pg_documents, half_target) + self._select_by_tokens(
                self.arxiv_documents, half_target
            )
        else:
            raise ValueError(f"Unknown composition: {composition}")

        # Shuffle the document order
        random.shuffle(selected)
        return selected

    def _select_by_tokens(self, documents: list[Document], target_tokens: int) -> list[Document]:
        """Select documents to approximately reach target token count."""
        if not documents:
            return []

        # Sort by token count to use greedy selection
        sorted_docs = sorted(documents, key=lambda d: d.token_count)
        selected = []
        current_tokens = 0

        for doc in sorted_docs:
            while current_tokens < target_tokens:
            # if current_tokens + doc.token_count <= target_tokens * 1.2:  # 20% buffer
                selected.append(doc)
                current_tokens += doc.token_count

                if current_tokens >= target_tokens:
                    break

        return selected

    def _find_document_by_size(
        self, documents: list[Document], target_tokens: int, buffer_tokens: int
    ) -> Document | None:
        """Find a document that fits within the token budget."""
        candidates = [doc for doc in documents if doc.token_count <= target_tokens + buffer_tokens]

        if not candidates:
            return None

        # Return the document closest to the target size
        return min(candidates, key=lambda d: abs(d.token_count - target_tokens))

    def _shuffle_content(self, content: str) -> str:
        """Shuffle sentences while preserving document boundaries."""
        # Split on document separators
        documents = content.split("\n\n---\n\n")
        shuffled_docs = []

        for doc in documents:
            # Split into sentences and shuffle
            sentences = [s.strip() for s in doc.split(".") if s.strip()]
            random.shuffle(sentences)
            shuffled_docs.append(". ".join(sentences) + ".")

        return "\n\n---\n\n".join(shuffled_docs)

    def save_haystack(self, haystack: Haystack, output_path: Path) -> None:
        """Save haystack to disk with metadata."""
        # Save the content
        content_path = output_path / f"haystack_{haystack.target_size}_{haystack.composition}.txt"
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(haystack.content)

        # Save metadata
        metadata_path = output_path / f"haystack_{haystack.target_size}_{haystack.composition}.json"
        metadata = {
            "target_size": haystack.target_size,
            "actual_tokens": haystack.token_count,
            "composition": haystack.composition,
            "documents_used": haystack.documents_used,
            "shuffled": haystack.shuffled,
            "needle_positions": haystack.needle_positions,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Saved haystack to {content_path}")

    def build_multiple_haystacks(
        self,
        target_sizes: list[int],
        compositions: list[str] | None = None,
        shuffle: bool = False,
        output_dir: Path | None = None,
    ) -> list[Haystack]:
        """Build multiple haystacks with different configurations."""
        if compositions is None:
            compositions = ["pg_heavy", "arxiv_heavy", "mixed"]

        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            output_dir = project_root / "data" / "context_advantage" / "haystacks"

        output_dir.mkdir(parents=True, exist_ok=True)
        haystacks = []

        for target_size in target_sizes:
            for composition in compositions:
                try:
                    haystack = self.build_haystack(target_size=target_size, composition=composition, shuffle=shuffle)

                    self.save_haystack(haystack, output_dir)
                    haystacks.append(haystack)

                except Exception as e:
                    print(f"❌ Failed to build {composition} haystack for {target_size} tokens: {e}")

        return haystacks


def main():
    """Command line interface for building haystacks."""
    parser = argparse.ArgumentParser(description="Build haystacks for context window experiments")
    parser.add_argument(
        "--target-sizes", type=int, nargs="+", default=[10000, 50000, 100000], help="Target token sizes for haystacks"
    )
    parser.add_argument(
        "--compositions",
        nargs="+",
        default=["pg_heavy", "arxiv_heavy", "mixed"],
        choices=["pg_heavy", "arxiv_heavy", "mixed"],
        help="Composition strategies",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sentence order within documents")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: ../data/haystacks/)")
    parser.add_argument("--data-dir", type=Path, help="Data directory containing source documents")

    args = parser.parse_args()

    # Initialize builder
    builder = HaystackBuilder(data_dir=args.data_dir)

    # Build haystacks
    print(f"🏗️  Building haystacks for sizes: {args.target_sizes}")
    print(f"📝 Compositions: {args.compositions}")
    print(f"🔀 Shuffle: {args.shuffle}")

    haystacks = builder.build_multiple_haystacks(
        target_sizes=args.target_sizes, compositions=args.compositions, shuffle=args.shuffle, output_dir=args.output_dir
    )

    print(f"\n✅ Built {len(haystacks)} haystacks successfully!")

    # Summary statistics
    for composition in args.compositions:
        comp_haystacks = [h for h in haystacks if h.composition == composition]
        if comp_haystacks:
            avg_accuracy = sum(abs(h.token_count - h.target_size) / h.target_size for h in comp_haystacks) / len(
                comp_haystacks
            )
            print(f"📊 {composition}: {len(comp_haystacks)} haystacks, avg accuracy: {(1 - avg_accuracy) * 100:.1f}%")


if __name__ == "__main__":
    main()
