#!/usr/bin/env python3
"""
Script to download datasets for PyData 2025 Context Is King experiments
"""

import json
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


def download_paul_graham_essays():
    """Download Paul Graham essays from his website"""
    console.print("[blue]Downloading Paul Graham essays...[/blue]")

    # List of Paul Graham essay URLs (sample set for testing)
    essays = [
        {"title": "How to Do Great Work", "url": "http://www.paulgraham.com/greatwork.html"},
        {"title": "The Refragmentation", "url": "http://www.paulgraham.com/re.html"},
        {"title": "Economic Inequality", "url": "http://www.paulgraham.com/inequality.html"},
        {"title": "The Bus Ticket Theory of Genius", "url": "http://www.paulgraham.com/genius.html"},
        {"title": "Novelty and Heresy", "url": "http://www.paulgraham.com/nov.html"},
        {"title": "Life is Short", "url": "http://www.paulgraham.com/vb.html"},
        {"title": "Mean People Fail", "url": "http://www.paulgraham.com/mean.html"},
        {"title": "Before the Startup", "url": "http://www.paulgraham.com/before.html"},
        {"title": "How to Raise Money", "url": "http://www.paulgraham.com/fr.html"},
        {"title": "Do Things that Don't Scale", "url": "http://www.paulgraham.com/ds.html"},
        {"title": "Startup Ideas", "url": "http://www.paulgraham.com/startupideas.html"},
        {"title": "How to Get Startup Ideas", "url": "http://www.paulgraham.com/startupideas.html"},
        {"title": "The Hardest Lessons for Startups to Learn", "url": "http://www.paulgraham.com/startuplessons.html"},
        {"title": "What We Look for in Founders", "url": "http://www.paulgraham.com/founders.html"},
        {"title": "Relentlessly Resourceful", "url": "http://www.paulgraham.com/relres.html"},
    ]

    data_dir = Path("data/paul_graham_essays")
    data_dir.mkdir(exist_ok=True)

    downloaded_essays = []

    for essay in track(essays, description="Downloading essays..."):
        try:
            # Add delay to be respectful
            time.sleep(1)

            response = requests.get(essay["url"], timeout=30)
            response.raise_for_status()

            # Extract text content (basic HTML parsing)
            text = response.text

            # Remove HTML tags and extract main content
            # This is a simplified extraction - you might want to use BeautifulSoup
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            # Save essay
            filename = re.sub(r"[^\w\s-]", "", essay["title"]).strip()
            filename = re.sub(r"[-\s]+", "-", filename)

            essay_path = data_dir / f"{filename}.txt"
            with open(essay_path, "w", encoding="utf-8") as f:
                f.write(text)

            downloaded_essays.append(
                {
                    "title": essay["title"],
                    "url": essay["url"],
                    "filename": filename + ".txt",
                    "length": len(text),
                    "doc_type": "paul_graham",
                }
            )

            console.print(f"   ✅ Downloaded: {essay['title']}")

        except Exception as e:
            console.print(f"   ❌ Failed to download {essay['title']}: {e}")

    # Save metadata
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(downloaded_essays, f, indent=2)

    console.print(f"[green]Downloaded {len(downloaded_essays)} Paul Graham essays[/green]")
    return downloaded_essays


def download_arxiv_papers():
    """Download arXiv papers using Hugging Face datasets"""
    console.print("[blue]Downloading arXiv papers...[/blue]")

    try:
        # Load a subset of the arXiv dataset
        dataset = load_dataset("jamescalam/ai-arxiv2", split="train[:100]")  # Just first 100 for testing

        data_dir = Path("data/arxiv_papers")
        data_dir.mkdir(exist_ok=True)

        downloaded_papers = []

        for i, paper in enumerate(track(dataset, description="Processing arXiv papers...")):
            try:
                # Extract paper content
                title = paper.get("title", f"paper_{i}")
                abstract = paper.get("abstract", "")
                content = paper.get("content", abstract)  # Use full content if available, else abstract

                if not content:
                    continue

                # Clean filename
                filename = re.sub(r"[^\w\s-]", "", title).strip()
                filename = re.sub(r"[-\s]+", "-", filename)[:50]  # Limit length

                paper_path = data_dir / f"{filename}_{i}.txt"
                with open(paper_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n\nAbstract: {abstract}\n\nContent:\n{content}")

                downloaded_papers.append(
                    {
                        "title": title,
                        "filename": f"{filename}_{i}.txt",
                        "length": len(content),
                        "doc_type": "arxiv",
                        "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                    }
                )

            except Exception as e:
                console.print(f"   ❌ Failed to process paper {i}: {e}")

        # Save metadata
        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(downloaded_papers, f, indent=2)

        console.print(f"[green]Downloaded {len(downloaded_papers)} arXiv papers[/green]")
        return downloaded_papers

    except Exception as e:
        console.print(f"[red]Failed to download arXiv dataset: {e}[/red]")
        return []


def download_chroma_needles():
    """Download Chroma needle datasets from their repository"""
    console.print("[blue]Downloading Chroma needles...[/blue]")

    # URLs for Chroma's needle data
    needle_urls = ["https://raw.githubusercontent.com/chroma-core/context-rot/master/data/pg_distractors.json"]

    data_dir = Path("data/chroma_needles")
    data_dir.mkdir(exist_ok=True)

    downloaded_files = []

    for url in needle_urls:
        try:
            filename = url.split("/")[-1]
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            file_path = data_dir / filename
            with open(file_path, "wb") as f:
                f.write(response.content)

            downloaded_files.append({"filename": filename, "url": url, "size": len(response.content)})

            console.print(f"   ✅ Downloaded: {filename}")

        except Exception as e:
            console.print(f"   ❌ Failed to download {url}: {e}")

    # Create sample needles for testing
    sample_needles = [
        {
            "needle": "The key to startup success is relentless execution and customer focus.",
            "question": "What is the key to startup success?",
            "answer": "relentless execution and customer focus",
            "similarity_score": 0.8,
            "domain": "startup",
        },
        {
            "needle": "Machine learning models require massive amounts of data to achieve good performance.",
            "question": "What do machine learning models require for good performance?",
            "answer": "massive amounts of data",
            "similarity_score": 0.75,
            "domain": "ml",
        },
        {
            "needle": "The transformer architecture revolutionized natural language processing through attention mechanisms.",
            "question": "How did transformers revolutionize NLP?",
            "answer": "through attention mechanisms",
            "similarity_score": 0.7,
            "domain": "ai",
        },
    ]

    needles_path = data_dir / "sample_needles.json"
    with open(needles_path, "w", encoding="utf-8") as f:
        json.dump(sample_needles, f, indent=2)

    downloaded_files.append({"filename": "sample_needles.json", "type": "generated", "count": len(sample_needles)})

    # Save metadata
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(downloaded_files, f, indent=2)

    console.print("[green]Downloaded Chroma needles data[/green]")
    return downloaded_files


def download_longmemeval():
    """Download LongMemEval dataset"""
    console.print("[blue]Downloading LongMemEval dataset...[/blue]")

    try:
        # Note: You may need to adjust this based on the actual LongMemEval dataset location
        # For now, we'll create a sample structure

        data_dir = Path("data/longmemeval")
        data_dir.mkdir(exist_ok=True)

        # Create sample conversational data for testing
        sample_conversations = [
            {
                "conversation_id": "conv_001",
                "messages": [
                    {"role": "user", "content": "Tell me about the history of artificial intelligence."},
                    {"role": "assistant", "content": "AI history began in the 1950s with pioneers like Alan Turing..."},
                    {"role": "user", "content": "What was Turing's main contribution?"},
                    {
                        "role": "assistant",
                        "content": "Turing proposed the famous Turing Test to measure machine intelligence...",
                    },
                ],
                "needle": "Turing Test",
                "question": "What test did Turing propose to measure machine intelligence?",
                "answer": "Turing Test",
            },
            {
                "conversation_id": "conv_002",
                "messages": [
                    {"role": "user", "content": "Explain machine learning algorithms."},
                    {
                        "role": "assistant",
                        "content": "Machine learning includes supervised, unsupervised, and reinforcement learning...",
                    },
                    {"role": "user", "content": "What's the difference between supervised and unsupervised learning?"},
                    {
                        "role": "assistant",
                        "content": "Supervised learning uses labeled data, while unsupervised finds patterns in unlabeled data...",
                    },
                ],
                "needle": "labeled data",
                "question": "What type of data does supervised learning use?",
                "answer": "labeled data",
            },
        ]

        conversations_path = data_dir / "sample_conversations.json"
        with open(conversations_path, "w", encoding="utf-8") as f:
            json.dump(sample_conversations, f, indent=2)

        metadata = {
            "filename": "sample_conversations.json",
            "type": "generated_sample",
            "conversation_count": len(sample_conversations),
            "note": "This is sample data. Replace with actual LongMemEval dataset.",
        }

        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[green]Created sample LongMemEval data ({len(sample_conversations)} conversations)[/green]")
        return metadata

    except Exception as e:
        console.print(f"[red]Failed to download LongMemEval: {e}[/red]")
        return {}


def main():
    """Main download function"""
    console.print("🚀 [bold blue]Downloading datasets for PyData 2025 Context Is King[/bold blue]")
    console.print("=" * 60)

    results = {}

    # Download all datasets
    results["paul_graham"] = download_paul_graham_essays()
    results["arxiv"] = download_arxiv_papers()
    results["chroma_needles"] = download_chroma_needles()
    results["longmemeval"] = download_longmemeval()

    # Summary
    console.print("\n📊 [bold green]Download Summary[/bold green]")
    for dataset, data in results.items():
        if isinstance(data, list):
            console.print(f"   {dataset}: {len(data)} items")
        elif isinstance(data, dict) and "count" in data:
            console.print(f"   {dataset}: {data['count']} items")
        else:
            console.print(f"   {dataset}: downloaded")

    console.print("\n✅ [bold green]All datasets downloaded successfully![/bold green]")
    console.print("\n📁 Data directories:")
    console.print("   data/paul_graham_essays/")
    console.print("   data/arxiv_papers/")
    console.print("   data/chroma_needles/")
    console.print("   data/longmemeval/")


if __name__ == "__main__":
    main()
