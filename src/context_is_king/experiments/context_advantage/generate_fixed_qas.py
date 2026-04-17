#!/usr/bin/env python3
"""
Generate Fixed Question-Answer Sets for Context Advantage Experiments

This script creates a fixed dataset of questions and answers to replace random
generation in needle-in-haystack experiments. This improves consistency and
transparency by fixing another variable in the experimental setup.

For each question type (direct, cross_reference, synthesis, domain_transfer),
we generate 10 carefully crafted Q&As that can be reused across experiments.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class QuestionAnswer:
    """Represents a question-answer pair with metadata."""

    question: str
    answer: str
    question_type: str
    domain: str  # 'pg', 'arxiv', or 'synthetic'
    complexity: str  # 'simple', 'medium', 'complex'
    id: str


class FixedQAGenerator:
    """Generates fixed question-answer sets for needle-in-haystack experiments."""

    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            output_dir = project_root / "data" / "context_advantage" / "fixed_qas"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_qa_sets(self) -> Dict[str, List[QuestionAnswer]]:
        """Generate all Q&A sets for each question type."""
        qa_sets = {}

        question_types = ["direct", "cross_reference", "synthesis", "domain_transfer"]

        for qtype in question_types:
            print(f"📝 Generating {qtype} questions...")
            qa_sets[qtype] = self._generate_qa_set(qtype, count=10)

        return qa_sets

    def _generate_qa_set(self, question_type: str, count: int = 10) -> List[QuestionAnswer]:
        """Generate a set of Q&As for a specific question type."""
        qas = []

        if question_type == "direct":
            qas.extend(self._generate_direct_qas(count))
        elif question_type == "cross_reference":
            qas.extend(self._generate_cross_reference_qas(count))
        elif question_type == "synthesis":
            qas.extend(self._generate_synthesis_qas(count))
        elif question_type == "domain_transfer":
            qas.extend(self._generate_domain_transfer_qas(count))

        return qas

    def _generate_direct_qas(self, count: int) -> List[QuestionAnswer]:
        """Generate direct factual questions."""
        templates = [
            {
                "question": "What is the founding year of {company}?",
                "answer": "{company} was founded in {year}.",
                "domain": "synthetic",
                "complexity": "simple",
                "facts": [
                    {"company": "Zenith Dynamics", "year": "1987"},
                    {"company": "Aurora Technologies", "year": "2003"},
                    {"company": "Pinnacle Systems", "year": "1994"},
                    {"company": "Quantum Solutions", "year": "2001"},
                    {"company": "Meridian Corp", "year": "1999"},
                ],
            },
            {
                "question": "Who is the CEO of {company}?",
                "answer": "The CEO of {company} is {ceo_name}.",
                "domain": "synthetic",
                "complexity": "simple",
                "facts": [
                    {"company": "TechFlow Industries", "ceo_name": "Sarah Chen"},
                    {"company": "DataStream Corp", "ceo_name": "Michael Rodriguez"},
                    {"company": "CloudBridge Solutions", "ceo_name": "Jennifer Kim"},
                    {"company": "NeuralPath Systems", "ceo_name": "David Thompson"},
                    {"company": "Velocity Enterprises", "ceo_name": "Lisa Wang"},
                ],
            },
        ]

        qas = []
        qa_id = 1

        for template in templates:
            for fact in template["facts"]:
                if len(qas) >= count:
                    break

                question = template["question"].format(**fact)
                answer = template["answer"].format(**fact)

                qas.append(
                    QuestionAnswer(
                        question=question,
                        answer=answer,
                        question_type="direct",
                        domain=template["domain"],
                        complexity=template["complexity"],
                        id=f"direct_{qa_id:02d}",
                    )
                )
                qa_id += 1

        return qas[:count]

    def _generate_cross_reference_qas(self, count: int) -> List[QuestionAnswer]:
        """Generate questions requiring cross-referencing multiple facts."""
        templates = [
            {
                "question": "Which company founded in {year} has its headquarters in {city}?",
                "answer": "{company} was founded in {year} and has its headquarters in {city}.",
                "domain": "synthetic",
                "complexity": "medium",
                "facts": [
                    {"company": "Nexus Innovations", "year": "2005", "city": "Seattle"},
                    {"company": "Prism Technologies", "year": "1998", "city": "Austin"},
                    {"company": "Apex Solutions", "year": "2010", "city": "Denver"},
                    {"company": "Vector Systems", "year": "2002", "city": "Portland"},
                    {"company": "Fusion Dynamics", "year": "2007", "city": "Phoenix"},
                ],
            },
            {
                "question": "What product launched by {company} in {year} generated ${revenue}M in revenue?",
                "answer": "{company} launched {product} in {year}, which generated ${revenue}M in revenue.",
                "domain": "synthetic",
                "complexity": "medium",
                "facts": [
                    {"company": "Digital Horizons", "product": "CloudSync Pro", "year": "2019", "revenue": "45"},
                    {"company": "Smart Systems", "product": "DataMiner Elite", "year": "2020", "revenue": "72"},
                    {"company": "Innovate Labs", "product": "AI Assistant Plus", "year": "2021", "revenue": "38"},
                    {"company": "Future Tech", "product": "SecureVault", "year": "2018", "revenue": "91"},
                    {"company": "Rapid Solutions", "product": "FlowOptimizer", "year": "2022", "revenue": "56"},
                ],
            },
        ]

        qas = []
        qa_id = 1

        for template in templates:
            for fact in template["facts"]:
                if len(qas) >= count:
                    break

                question = template["question"].format(**fact)
                answer = template["answer"].format(**fact)

                qas.append(
                    QuestionAnswer(
                        question=question,
                        answer=answer,
                        question_type="cross_reference",
                        domain=template["domain"],
                        complexity=template["complexity"],
                        id=f"cross_ref_{qa_id:02d}",
                    )
                )
                qa_id += 1

        return qas[:count]

    def _generate_synthesis_qas(self, count: int) -> List[QuestionAnswer]:
        """Generate questions requiring synthesis of multiple pieces of information."""
        templates = [
            {
                "question": "Based on the market performance data, which sector showed the highest growth rate between {start_year} and {end_year}?",
                "answer": "The {sector} sector showed the highest growth rate between {start_year} and {end_year}, with a {growth_rate}% increase.",
                "domain": "synthetic",
                "complexity": "complex",
                "facts": [
                    {"sector": "renewable energy", "start_year": "2018", "end_year": "2023", "growth_rate": "247"},
                    {"sector": "biotechnology", "start_year": "2019", "end_year": "2023", "growth_rate": "189"},
                    {
                        "sector": "artificial intelligence",
                        "start_year": "2020",
                        "end_year": "2023",
                        "growth_rate": "312",
                    },
                    {"sector": "quantum computing", "start_year": "2018", "end_year": "2023", "growth_rate": "156"},
                    {"sector": "space technology", "start_year": "2019", "end_year": "2023", "growth_rate": "203"},
                ],
            },
            {
                "question": "What can be concluded about {company}'s market strategy based on their {metric1} of {value1} and {metric2} of {value2}?",
                "answer": "Based on {company}'s {metric1} of {value1} and {metric2} of {value2}, the company is pursuing a {strategy} strategy focused on {focus_area}.",
                "domain": "synthetic",
                "complexity": "complex",
                "facts": [
                    {
                        "company": "GlobalTech",
                        "metric1": "R&D investment",
                        "value1": "22% of revenue",
                        "metric2": "patent filings",
                        "value2": "340 per year",
                        "strategy": "innovation-first",
                        "focus_area": "technological advancement",
                    },
                    {
                        "company": "MarketLeader",
                        "metric1": "marketing spend",
                        "value1": "$890M annually",
                        "metric2": "customer acquisition cost",
                        "value2": "$45",
                        "strategy": "growth-oriented",
                        "focus_area": "market expansion",
                    },
                    {
                        "company": "EfficiencyCorp",
                        "metric1": "operational efficiency",
                        "value1": "94%",
                        "metric2": "cost reduction",
                        "value2": "15% annually",
                        "strategy": "efficiency-driven",
                        "focus_area": "operational excellence",
                    },
                    {
                        "company": "CustomerFirst",
                        "metric1": "customer satisfaction",
                        "value1": "4.8/5.0",
                        "metric2": "retention rate",
                        "value2": "96%",
                        "strategy": "customer-centric",
                        "focus_area": "user experience",
                    },
                    {
                        "company": "SustainableFuture",
                        "metric1": "carbon neutrality",
                        "value1": "achieved in 2023",
                        "metric2": "renewable energy use",
                        "value2": "87%",
                        "strategy": "sustainability-focused",
                        "focus_area": "environmental impact",
                    },
                ],
            },
        ]

        qas = []
        qa_id = 1

        for template in templates:
            for fact in template["facts"]:
                if len(qas) >= count:
                    break

                question = template["question"].format(**fact)
                answer = template["answer"].format(**fact)

                qas.append(
                    QuestionAnswer(
                        question=question,
                        answer=answer,
                        question_type="synthesis",
                        domain=template["domain"],
                        complexity=template["complexity"],
                        id=f"synthesis_{qa_id:02d}",
                    )
                )
                qa_id += 1

        return qas[:count]

    def _generate_domain_transfer_qas(self, count: int) -> List[QuestionAnswer]:
        """Generate questions that transfer concepts between domains."""
        templates = [
            {
                "question": "How does the concept of {concept} from {source_domain} apply to {target_domain}?",
                "answer": "The concept of {concept} from {source_domain} applies to {target_domain} through {application}. This demonstrates {principle}.",
                "domain": "synthetic",
                "complexity": "complex",
                "facts": [
                    {
                        "concept": "natural selection",
                        "source_domain": "biology",
                        "target_domain": "algorithm optimization",
                        "application": "genetic algorithms that evolve solutions over generations",
                        "principle": "survival of the fittest in computational problem-solving",
                    },
                    {
                        "concept": "network effects",
                        "source_domain": "economics",
                        "target_domain": "social media platforms",
                        "application": "user value increasing exponentially with platform adoption",
                        "principle": "positive feedback loops in digital ecosystems",
                    },
                    {
                        "concept": "entropy",
                        "source_domain": "thermodynamics",
                        "target_domain": "information theory",
                        "application": "measuring information content and compression efficiency",
                        "principle": "disorder quantification across physical and digital systems",
                    },
                    {
                        "concept": "resonance frequency",
                        "source_domain": "physics",
                        "target_domain": "organizational behavior",
                        "application": "optimal communication cadence that maximizes team performance",
                        "principle": "harmonic alignment between system components",
                    },
                    {
                        "concept": "immune system",
                        "source_domain": "biology",
                        "target_domain": "cybersecurity",
                        "application": "adaptive threat detection that learns from previous attacks",
                        "principle": "distributed defense mechanisms with memory",
                    },
                ],
            },
            {
                "question": "What parallels exist between {system1} in {domain1} and {system2} in {domain2}?",
                "answer": "Both {system1} in {domain1} and {system2} in {domain2} exhibit {shared_property} and demonstrate {common_principle}. This suggests {insight}.",
                "domain": "synthetic",
                "complexity": "complex",
                "facts": [
                    {
                        "system1": "neural networks",
                        "domain1": "neuroscience",
                        "system2": "deep learning",
                        "domain2": "artificial intelligence",
                        "shared_property": "layered processing and weighted connections",
                        "common_principle": "parallel information processing",
                        "insight": "biological inspiration can guide artificial system design",
                    },
                    {
                        "system1": "market dynamics",
                        "domain1": "economics",
                        "system2": "ecosystem balance",
                        "domain2": "ecology",
                        "shared_property": "resource allocation and competition",
                        "common_principle": "equilibrium through feedback mechanisms",
                        "insight": "natural and artificial systems follow similar optimization principles",
                    },
                    {
                        "system1": "cellular metabolism",
                        "domain1": "biology",
                        "system2": "factory production",
                        "domain2": "manufacturing",
                        "shared_property": "input transformation and waste management",
                        "common_principle": "efficient resource utilization",
                        "insight": "biological processes inspire industrial optimization",
                    },
                    {
                        "system1": "swarm behavior",
                        "domain1": "animal behavior",
                        "system2": "distributed computing",
                        "domain2": "computer science",
                        "shared_property": "emergent intelligence from simple rules",
                        "common_principle": "collective problem-solving",
                        "insight": "decentralized coordination can solve complex problems",
                    },
                    {
                        "system1": "adaptive immunity",
                        "domain1": "immunology",
                        "system2": "machine learning",
                        "domain2": "data science",
                        "shared_property": "pattern recognition and memory formation",
                        "common_principle": "learning from experience",
                        "insight": "biological learning mechanisms inform artificial intelligence",
                    },
                ],
            },
        ]

        qas = []
        qa_id = 1

        for template in templates:
            for fact in template["facts"]:
                if len(qas) >= count:
                    break

                question = template["question"].format(**fact)
                answer = template["answer"].format(**fact)

                qas.append(
                    QuestionAnswer(
                        question=question,
                        answer=answer,
                        question_type="domain_transfer",
                        domain=template["domain"],
                        complexity=template["complexity"],
                        id=f"domain_transfer_{qa_id:02d}",
                    )
                )
                qa_id += 1

        return qas[:count]

    def save_qa_sets(self, qa_sets: Dict[str, List[QuestionAnswer]]) -> None:
        """Save Q&A sets to JSON files."""
        for question_type, qas in qa_sets.items():
            output_file = self.output_dir / f"{question_type}_qas.json"

            # Convert to dict format for JSON serialization
            qa_data = {
                "question_type": question_type,
                "count": len(qas),
                "questions_answers": [asdict(qa) for qa in qas],
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved {len(qas)} {question_type} Q&As to {output_file}")

    def validate_qa_sets(self, qa_sets: Dict[str, List[QuestionAnswer]]) -> bool:
        """Validate that all Q&A sets meet requirements."""
        all_valid = True

        for question_type, qas in qa_sets.items():
            print(f"🔍 Validating {question_type} Q&As...")

            if len(qas) != 10:
                print(f"❌ {question_type}: Expected 10 Q&As, got {len(qas)}")
                all_valid = False
                continue

            # Check for unique IDs
            ids = [qa.id for qa in qas]
            if len(set(ids)) != len(ids):
                print(f"❌ {question_type}: Duplicate IDs found")
                all_valid = False

            # Check for non-empty questions and answers
            for qa in qas:
                if not qa.question.strip() or not qa.answer.strip():
                    print(f"❌ {question_type}: Empty question or answer in {qa.id}")
                    all_valid = False

                if qa.question_type != question_type:
                    print(f"❌ {question_type}: Mismatched question type in {qa.id}")
                    all_valid = False

            if all_valid:
                print(f"✅ {question_type}: All validations passed")

        return all_valid


def main():
    """Generate and save fixed Q&A sets."""
    print("🚀 Generating fixed question-answer sets for needle-in-haystack experiments...")

    generator = FixedQAGenerator()

    # Generate all Q&A sets
    qa_sets = generator.generate_all_qa_sets()

    # Validate the sets
    if not generator.validate_qa_sets(qa_sets):
        print("❌ Validation failed. Please fix issues before proceeding.")
        return

    # Save to JSON files
    generator.save_qa_sets(qa_sets)

    # Summary
    total_qas = sum(len(qas) for qas in qa_sets.values())
    print(f"\n✅ Successfully generated {total_qas} Q&As across {len(qa_sets)} question types")
    print(f"📁 Files saved to: {generator.output_dir}")

    # Show sample from each type
    print("\n📋 Sample questions:")
    for question_type, qas in qa_sets.items():
        sample_qa = qas[0]
        print(f"\n{question_type.upper()}:")
        print(f"  Q: {sample_qa.question}")
        print(f"  A: {sample_qa.answer}")


if __name__ == "__main__":
    main()
