#!/usr/bin/env python3
"""
Needle Generator for Context Window Advantage Experiments

This module generates synthetic needles (facts, questions, test cases) that can be
inserted into haystacks for controlled evaluation. Supports different needle types:
- Direct retrieval: Simple fact extraction
- Cross-reference: Connect information from multiple needles
- Synthesis: Combine information requiring reasoning
- Domain transfer: Cross-domain knowledge application

Usage:
    python needle_generator.py --num-needles 50 --domains pg,arxiv
    python needle_generator.py --needle-types direct,synthesis --output-dir data/needles/
"""

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken


@dataclass
class NeedleFact:
    """A synthetic fact to be inserted as a needle."""

    id: str
    content: str
    domain: str  # 'pg', 'arxiv', 'synthetic'
    fact_type: str  # 'person', 'date', 'concept', 'number', 'location'
    key_entities: list[str]
    token_count: int


@dataclass
class TestQuestion:
    """A question that tests retrieval of needle facts."""

    id: str
    question: str
    expected_answer: str
    question_type: str  # 'direct', 'cross_reference', 'synthesis', 'domain_transfer'
    required_needles: list[str]  # IDs of needles needed to answer
    domain: str
    difficulty: str  # 'easy', 'medium', 'hard'
    token_count: int


@dataclass
class NeedleSet:
    """A complete set of needles and questions for testing."""

    id: str
    needles: list[NeedleFact]
    questions: list[TestQuestion]
    total_needles: int
    total_questions: int
    domains: list[str]
    question_types: list[str]


@dataclass
class FixedQuestionAnswer:
    """Represents a fixed question-answer pair loaded from JSON."""

    question: str
    answer: str
    question_type: str
    domain: str
    complexity: str
    id: str


class NeedleGenerator:
    """Generates synthetic needles and test questions for haystack experiments."""

    def __init__(self, use_fixed_qas: bool = True, fixed_qa_dir: Path | None = None):
        """Initialize the needle generator.

        Args:
            use_fixed_qas: Whether to use fixed Q&As instead of random generation
            fixed_qa_dir: Directory containing fixed Q&A JSON files
        """
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        self.needle_id_counter = 0
        self.question_id_counter = 0
        self.use_fixed_qas = use_fixed_qas

        # Always load templates for needle generation (needed for facts)
        self.pg_templates = self._load_pg_templates()
        self.arxiv_templates = self._load_arxiv_templates()
        self.synthetic_templates = self._load_synthetic_templates()

        if self.use_fixed_qas:
            if fixed_qa_dir is None:
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent
                fixed_qa_dir = project_root / "data" / "context_advantage" / "fixed_qas"
            self.fixed_qa_dir = Path(fixed_qa_dir)
            self.fixed_qas = self._load_fixed_qas()
            print(f"🧵 Initialized Needle Generator with fixed Q&As from {self.fixed_qa_dir}")
        else:
            self.fixed_qas = {}
            print("🧵 Initialized Needle Generator with random generation")

    def _load_fixed_qas(self) -> dict[str, list[FixedQuestionAnswer]]:
        """Load fixed question-answer sets from JSON files."""
        fixed_qas = {}
        question_types = ["direct", "cross_reference", "synthesis", "domain_transfer"]

        for qtype in question_types:
            qa_file = self.fixed_qa_dir / f"{qtype}_qas.json"
            if not qa_file.exists():
                print(f"⚠️  Fixed Q&A file not found: {qa_file}")
                fixed_qas[qtype] = []
                continue

            try:
                with open(qa_file, encoding="utf-8") as f:
                    data = json.load(f)

                qas = []
                for qa_data in data["questions_answers"]:
                    qas.append(FixedQuestionAnswer(**qa_data))

                fixed_qas[qtype] = qas
                print(f"📚 Loaded {len(qas)} fixed {qtype} Q&As")

            except Exception as e:
                print(f"❌ Error loading {qa_file}: {e}")
                fixed_qas[qtype] = []

        return fixed_qas

    def _load_pg_templates(self) -> dict[str, list[str]]:
        """Load Paul Graham domain templates."""
        return {
            "person": [
                "Paul Graham worked with {name} at {company} in {year}.",
                "In {year}, {name} became a key advisor to {company}.",
                "{name} was instrumental in developing {concept} at {company}.",
                "The partnership between {name} and {company} lasted {duration} years.",
            ],
            "date": [
                "Y Combinator was founded in {year} by Paul Graham and others.",
                "The first Y Combinator batch started in {season} {year}.",
                "In {year}, Y Combinator moved to {location}.",
                "{event} happened on {date} at Y Combinator.",
            ],
            "concept": [
                "Paul Graham defined {concept} as '{definition}'.",
                "The key to {concept} is understanding {principle}.",
                "{concept} requires three components: {comp1}, {comp2}, and {comp3}.",
                "According to Graham, {concept} is most important for {context}.",
            ],
            "number": [
                "Y Combinator funded {number} startups in its first batch.",
                "The average valuation of YC companies reached ${amount} million.",
                "{company} raised ${amount} million in Series {round}.",
                "{metric} increased by {percentage}% over {timeframe}.",
            ],
            "location": [
                "Paul Graham lived in {city} while writing {essay}.",
                "The Y Combinator office was located at {address} in {city}.",
                "{event} took place in {city} in {year}.",
                "Graham frequently visited {location} for {purpose}.",
            ],
        }

    def _load_arxiv_templates(self) -> dict[str, list[str]]:
        """Load ArXiv domain templates."""
        return {
            "person": [
                "Dr. {name} published groundbreaking research on {topic} in {year}.",
                "The collaboration between {name1} and {name2} led to {discovery}.",
                "{name} received the {award} for work in {field}.",
                "Professor {name} established the {theory} framework.",
            ],
            "date": [
                "The {algorithm} was first proposed in {year}.",
                "Experiments were conducted from {start_date} to {end_date}.",
                "The dataset was collected in {month} {year}.",
                "{breakthrough} was achieved on {date}.",
            ],
            "concept": [
                "The {algorithm} achieves {metric} of {value} on {dataset}.",
                "{method} outperforms {baseline} by {improvement}%.",
                "The key innovation in {approach} is {mechanism}.",
                "{technique} relies on the principle of {theory}.",
            ],
            "number": [
                "The model achieved {accuracy}% accuracy on {dataset}.",
                "Training required {hours} hours on {gpu_count} GPUs.",
                "The parameter count reached {params} million.",
                "Inference latency was reduced to {time} milliseconds.",
            ],
            "location": [
                "Experiments were conducted at {institution} in {city}.",
                "The data was collected from {location} over {duration}.",
                "{conference} was held in {city} in {year}.",
                "The research team is based at {university}.",
            ],
        }

    def _load_synthetic_templates(self) -> dict[str, list[str]]:
        """Load synthetic cross-domain templates."""
        return {
            "connection": [
                "The startup {company} used {technique} from {field} to solve {problem}.",
                "Paul Graham's essay on {topic} influenced the development of {algorithm}.",
                "The {principle} from entrepreneurship applies directly to {technical_area}.",
                "{researcher} adapted {startup_concept} for {academic_application}.",
            ],
            "comparison": [
                "Unlike traditional {approach1}, the new {approach2} achieves {benefit}.",
                "Both {startup} and {research_group} faced similar challenges with {problem}.",
                "The {business_metric} mirrors the {technical_metric} in surprising ways.",
                "{concept1} and {concept2} share the fundamental principle of {principle}.",
            ],
            "transfer": [
                "The lessons from {domain1} can be applied to {domain2} by {method}.",
                "{insight} from startup culture revolutionized {technical_field}.",
                "Academic research on {topic} validated {startup_hypothesis}.",
                "{entrepreneur} and {scientist} reached similar conclusions about {concept}.",
            ],
        }

    def generate_needle_fact(self, domain: str, fact_type: str = None, custom_content: str = None) -> NeedleFact:
        """Generate a single needle fact."""
        needle_id = f"needle_{self.needle_id_counter:04d}"
        self.needle_id_counter += 1

        if custom_content:
            content = custom_content
            # Extract entities from custom content (simplified)
            key_entities = [word.strip(".,!?") for word in content.split() if len(word) > 3 and word[0].isupper()][:3]
        else:
            if fact_type is None:
                fact_type = random.choice(list(self._get_templates(domain).keys()))

            content, key_entities = self._generate_fact_content(domain, fact_type)

        token_count = len(self.encoding.encode(content))

        return NeedleFact(
            id=needle_id,
            content=content,
            domain=domain,
            fact_type=fact_type or "custom",
            key_entities=key_entities,
            token_count=token_count,
        )

    def _get_templates(self, domain: str) -> dict[str, list[str]]:
        """Get templates for a specific domain."""
        if domain == "pg":
            return self.pg_templates
        if domain == "arxiv":
            return self.arxiv_templates
        if domain == "synthetic":
            return self.synthetic_templates
        raise ValueError(f"Unknown domain: {domain}")

    def _generate_fact_content(self, domain: str, fact_type: str) -> tuple[str, list[str]]:
        """Generate content for a fact based on domain and type."""
        templates = self._get_templates(domain)
        template = random.choice(templates[fact_type])

        # Generate random values based on placeholders
        entities = []
        content = template

        # Replace common placeholders with generated values
        replacements = self._generate_replacements(domain, fact_type)

        for placeholder, value in replacements.items():
            if f"{{{placeholder}}}" in content:
                content = content.replace(f"{{{placeholder}}}", value)
                if isinstance(value, str) and len(value) > 2:
                    entities.append(value)

        return content, entities[:3]  # Keep top 3 entities

    def _generate_replacements(self, domain: str, fact_type: str) -> dict[str, str]:
        """Generate replacement values for template placeholders."""
        replacements = {}

        # Common names and entities
        names = [
            "Sarah Chen",
            "Michael Rodriguez",
            "David Kim",
            "Lisa Zhang",
            "Alex Thompson",
            "Maria Garcia",
            "James Wilson",
            "Elena Petrov",
            "Ahmed Hassan",
            "Sophie Martin",
        ]

        companies = [
            "TechCorp",
            "DataFlow",
            "InnovateLab",
            "NextGen Systems",
            "CloudTech",
            "AI Dynamics",
            "QuantumSoft",
            "BioTech Solutions",
            "RoboVision",
            "CyberCore",
        ]

        concepts_pg = [
            "product-market fit",
            "lean startup",
            "user acquisition",
            "pivot strategy",
            "network effects",
            "viral growth",
            "bootstrapping",
            "venture funding",
        ]

        concepts_arxiv = [
            "attention mechanism",
            "gradient descent",
            "neural architecture",
            "reinforcement learning",
            "transfer learning",
            "generative models",
            "optimization algorithm",
            "feature extraction",
        ]

        cities = [
            "San Francisco",
            "Boston",
            "Austin",
            "Seattle",
            "New York",
            "London",
            "Berlin",
            "Tokyo",
            "Singapore",
            "Toronto",
        ]

        # Generate specific replacements
        replacements.update(
            {
                "name": random.choice(names),
                "name1": random.choice(names),
                "name2": random.choice(names),
                "company": random.choice(companies),
                "city": random.choice(cities),
                "location": random.choice(cities),
                "year": str(random.randint(2010, 2024)),
                "date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(2020, 2024)}",
                "number": str(random.randint(10, 500)),
                "amount": str(random.randint(1, 100)),
                "percentage": str(random.randint(5, 95)),
                "accuracy": str(random.randint(85, 99)),
                "hours": str(random.randint(1, 72)),
                "params": str(random.randint(1, 175)),
                "time": str(random.randint(1, 1000)),
            }
        )

        # Domain-specific concepts
        if domain == "pg":
            replacements.update(
                {
                    "concept": random.choice(concepts_pg),
                    "principle": random.choice(concepts_pg),
                }
            )
        elif domain == "arxiv":
            replacements.update(
                {
                    "concept": random.choice(concepts_arxiv),
                    "algorithm": random.choice(concepts_arxiv),
                    "technique": random.choice(concepts_arxiv),
                }
            )

        return replacements

    def generate_test_question(
        self, needles: list[NeedleFact], question_type: str = None, difficulty: str = "medium"
    ) -> TestQuestion:
        """Generate a test question that requires specific needles to answer."""
        question_id = f"question_{self.question_id_counter:04d}"
        self.question_id_counter += 1

        if question_type is None:
            question_type = random.choice(["direct", "cross_reference", "synthesis", "domain_transfer"])

        question, answer, required_needles, domain = self._generate_question_content(needles, question_type, difficulty)

        token_count = len(self.encoding.encode(question))

        return TestQuestion(
            id=question_id,
            question=question,
            expected_answer=answer,
            question_type=question_type,
            required_needles=required_needles,
            domain=domain,
            difficulty=difficulty,
            token_count=token_count,
        )

    def _generate_question_content(
        self, needles: list[NeedleFact], question_type: str, difficulty: str
    ) -> tuple[str, str, list[str], str]:
        """Generate question content based on available needles."""

        if self.use_fixed_qas and question_type in self.fixed_qas and self.fixed_qas[question_type]:
            # Use fixed Q&A instead of random generation
            fixed_qa = random.choice(self.fixed_qas[question_type])

            # For fixed Q&As, we still need to associate with needles for positioning
            # Select random needles to serve as the "carriers" of this information
            if question_type == "direct":
                required_needles = [random.choice(needles).id] if needles else []
            elif question_type == "cross_reference":
                num_needles = min(2, len(needles))
                required_needles = [n.id for n in random.sample(needles, num_needles)] if needles else []
            elif question_type == "synthesis":
                num_needles = min(3, len(needles))
                required_needles = [n.id for n in random.sample(needles, num_needles)] if needles else []
            elif question_type == "domain_transfer":
                num_needles = min(2, len(needles))
                required_needles = [n.id for n in random.sample(needles, num_needles)] if needles else []
            else:
                required_needles = [random.choice(needles).id] if needles else []

            return fixed_qa.question, fixed_qa.answer, required_needles, fixed_qa.domain

        # Fallback to original random generation logic
        if question_type == "direct":
            # Single needle, direct fact retrieval
            needle = random.choice(needles)
            entity = random.choice(needle.key_entities) if needle.key_entities else "unknown"

            question = f"What information is provided about {entity}?"
            answer = needle.content
            required_needles = [needle.id]
            domain = needle.domain

        elif question_type == "cross_reference":
            # Multiple needles from same domain
            domain_needles = [n for n in needles if len(needles) > 1]
            if len(domain_needles) < 2:
                # Fallback to direct if not enough needles
                return self._generate_question_content(needles, "direct", difficulty)

            selected = random.sample(domain_needles, 2)
            entities = []
            for needle in selected:
                if needle.key_entities:
                    entities.extend(needle.key_entities[:1])

            question = f"How are {' and '.join(entities[:2])} connected based on the provided information?"
            answer = f"Based on the provided facts: {selected[0].content} {selected[1].content}"
            required_needles = [n.id for n in selected]
            domain = selected[0].domain

        elif question_type == "synthesis":
            # Combine information from multiple needles
            if len(needles) < 2:
                return self._generate_question_content(needles, "direct", difficulty)

            selected = random.sample(needles, min(3, len(needles)))
            question = "What patterns or insights can be derived from combining the following pieces of information?"
            answer = "Combined insights: " + " ".join([n.content for n in selected])
            required_needles = [n.id for n in selected]
            domain = "mixed"

        elif question_type == "domain_transfer":
            # Cross-domain reasoning
            pg_needles = [n for n in needles if n.domain == "pg"]
            arxiv_needles = [n for n in needles if n.domain == "arxiv"]

            if not pg_needles or not arxiv_needles:
                return self._generate_question_content(needles, "synthesis", difficulty)

            pg_needle = random.choice(pg_needles)
            arxiv_needle = random.choice(arxiv_needles)

            question = "How might concepts from entrepreneurship relate to technical research based on the provided information?"
            answer = f"Connection between domains: {pg_needle.content} relates to {arxiv_needle.content}"
            required_needles = [pg_needle.id, arxiv_needle.id]
            domain = "cross_domain"

        else:
            raise ValueError(f"Unknown question type: {question_type}")

        return question, answer, required_needles, domain

    def generate_needle_set(
        self,
        num_needles: int,
        domains: list[str] | None = None,
        question_types: list[str] | None = None,
        questions_per_needle: int = 2,
    ) -> NeedleSet:
        """Generate a complete set of needles and questions."""

        if domains is None:
            domains = ["pg", "arxiv", "synthetic"]

        if question_types is None:
            question_types = ["direct", "cross_reference", "synthesis", "domain_transfer"]

        set_id = str(uuid.uuid4())[:8]

        print(f"🎯 Generating needle set {set_id} with {num_needles} needles...")

        # Generate needles across domains
        needles = []
        for i in range(num_needles):
            domain = random.choice(domains)
            needle = self.generate_needle_fact(domain)
            needles.append(needle)

        # Generate questions
        questions = []
        for _ in range(num_needles * questions_per_needle):
            question_type = random.choice(question_types)
            question = self.generate_test_question(needles, question_type)
            questions.append(question)

        print(f"✅ Generated {len(needles)} needles and {len(questions)} questions")

        return NeedleSet(
            id=set_id,
            needles=needles,
            questions=questions,
            total_needles=len(needles),
            total_questions=len(questions),
            domains=domains,
            question_types=question_types,
        )

    def save_needle_set(self, needle_set: NeedleSet, output_path: Path) -> None:
        """Save needle set to disk."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_path / f"needle_set_{needle_set.id}.json"
        with open(json_path, "w") as f:
            json.dump(asdict(needle_set), f, indent=2)

        # Save summary
        summary_path = output_path / f"needle_set_{needle_set.id}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Needle Set Summary: {needle_set.id}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Needles: {needle_set.total_needles}\n")
            f.write(f"Total Questions: {needle_set.total_questions}\n")
            f.write(f"Domains: {', '.join(needle_set.domains)}\n")
            f.write(f"Question Types: {', '.join(needle_set.question_types)}\n\n")

            f.write("Needle Preview:\n")
            f.write("-" * 20 + "\n")
            for needle in needle_set.needles[:3]:
                f.write(f"ID: {needle.id}\n")
                f.write(f"Domain: {needle.domain}\n")
                f.write(f"Content: {needle.content}\n")
                f.write(f"Entities: {', '.join(needle.key_entities)}\n\n")

            f.write("Question Preview:\n")
            f.write("-" * 20 + "\n")
            for question in needle_set.questions[:3]:
                f.write(f"ID: {question.id}\n")
                f.write(f"Type: {question.question_type}\n")
                f.write(f"Question: {question.question}\n")
                f.write(f"Required Needles: {', '.join(question.required_needles)}\n\n")

        print(f"💾 Saved needle set to {json_path}")


def main():
    """Command line interface for generating needles."""
    parser = argparse.ArgumentParser(description="Generate synthetic needles for context experiments")
    parser.add_argument("--num-needles", type=int, default=50, help="Number of needles to generate")
    parser.add_argument("--domains", default="pg,arxiv,synthetic", help="Comma-separated list of domains")
    parser.add_argument(
        "--question-types",
        default="direct,cross_reference,synthesis,domain_transfer",
        help="Comma-separated list of question types",
    )
    parser.add_argument("--questions-per-needle", type=int, default=2, help="Number of questions per needle")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: ../data/needles/)")
    parser.add_argument("--num-sets", type=int, default=1, help="Number of needle sets to generate")

    args = parser.parse_args()

    # Parse domains and question types
    domains = args.domains.split(",")
    question_types = args.question_types.split(",")

    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        args.output_dir = project_root / "data" / "context_advantage" / "needles"

    # Initialize generator
    generator = NeedleGenerator()

    # Generate needle sets
    print(f"🎯 Generating {args.num_sets} needle set(s)...")
    print("📝 Parameters:")
    print(f"   - Needles per set: {args.num_needles}")
    print(f"   - Domains: {domains}")
    print(f"   - Question types: {question_types}")
    print(f"   - Questions per needle: {args.questions_per_needle}")
    print(f"   - Output: {args.output_dir}")
    print("-" * 50)

    for i in range(args.num_sets):
        needle_set = generator.generate_needle_set(
            num_needles=args.num_needles,
            domains=domains,
            question_types=question_types,
            questions_per_needle=args.questions_per_needle,
        )

        generator.save_needle_set(needle_set, args.output_dir)

    print(f"\n✅ Generated {args.num_sets} needle set(s) successfully!")


if __name__ == "__main__":
    main()
