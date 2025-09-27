"""
Experiment controller for orchestrating complete RAG pipeline experiments
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import asdict
from itertools import product

from rich.console import Console
from rich.progress import track
from rich.table import Table

from ..types import (
    Document, DocumentChunk, RetrievalResult, GenerationResult, 
    JudgeResult, ExperimentConfig, ExperimentResults, PipelineConfig
)
from ..document_processing.processor import DocumentProcessor
from ..embedding_storage.manager import EmbeddingStorageManager
from ..retrieval.engine import RetrievalEngine
from ..reranking.module import RerankerModule
from ..context_assembly.engine import ContextAssembly
from ..generation.interface import GenerationInterface

console = Console()


class ExperimentController:
    """Orchestrates experiments and manages configurations"""
    
    def __init__(self, pipeline_config: PipelineConfig):
        self.config = pipeline_config
        
        # Initialize all pipeline components
        console.print("[blue]Initializing RAG pipeline components...[/blue]")
        
        self.doc_processor = DocumentProcessor(pipeline_config)
        self.embedding_manager = EmbeddingStorageManager(pipeline_config)
        self.retrieval_engine = RetrievalEngine(self.embedding_manager, pipeline_config)
        self.reranker = RerankerModule(pipeline_config)
        self.context_assembly = ContextAssembly(pipeline_config)
        self.generation = GenerationInterface(pipeline_config)
        
        # Warm up models for better performance
        self._warm_up_models()
        
        console.print("[green]ExperimentController initialized successfully[/green]")
    
    def run_experiment(self, experiment_config: ExperimentConfig) -> ExperimentResults:
        """
        Run complete experiment with given configuration
        
        Args:
            experiment_config: Configuration for the experiment
            
        Returns:
            Complete experiment results
        """
        console.print(f"[bold blue]Starting experiment: {experiment_config.experiment_id}[/bold blue]")
        start_time = time.time()
        
        # Initialize results container
        results = ExperimentResults(
            experiment_id=experiment_config.experiment_id,
            config=experiment_config,
            metadata={"start_time": start_time}
        )
        
        try:
            # Step 1: Load and process documents
            console.print("[blue]Step 1: Loading and processing documents[/blue]")
            processed_data = self._prepare_experiment_data(experiment_config)
            
            # Step 2: Create embeddings and collections
            console.print("[blue]Step 2: Creating embeddings and collections[/blue]")
            collections = self._create_experiment_collections(processed_data, experiment_config)
            
            # Step 3: Generate all experiment variations
            console.print("[blue]Step 3: Generating experiment variations[/blue]")
            variations = self._generate_experiment_variations(experiment_config)
            
            console.print(f"[blue]Running {len(variations)} experiment variations[/blue]")
            
            # Step 4: Execute all variations
            for i, variation in enumerate(track(variations, description="Running variations...")):
                try:
                    console.print(f"[blue]Running variation {i+1}/{len(variations)}: {variation['name']}[/blue]")
                    
                    # Run generation for this variation
                    generation_results = self._run_generation_variation(
                        variation, collections, processed_data
                    )
                    results.generation_results.extend(generation_results)
                    
                    # Run evaluation for this variation
                    judge_results = self._run_evaluation_variation(
                        generation_results, variation, experiment_config
                    )
                    results.judge_results.extend(judge_results)
                    
                    # Save intermediate results if configured
                    if experiment_config.save_intermediate:
                        self._save_intermediate_results(results, experiment_config, i)
                    
                except Exception as e:
                    console.print(f"[red]Variation {i+1} failed: {e}[/red]")
                    continue
            
            # Step 5: Calculate summary statistics
            console.print("[blue]Step 5: Calculating summary statistics[/blue]")
            results.summary_stats = self._calculate_summary_statistics(results)
            
            # Step 6: Calculate retrieval metrics if ground truth available
            if hasattr(processed_data, 'ground_truth'):
                results.retrieval_metrics = self._calculate_experiment_retrieval_metrics(
                    results, processed_data.ground_truth
                )
            
            # Finalize results
            total_time = time.time() - start_time
            results.metadata.update({
                "end_time": time.time(),
                "total_duration_seconds": total_time,
                "variations_completed": len([r for r in results.generation_results if r.response]),
                "evaluation_success_rate": len([j for j in results.judge_results if j.judge_response != "ERROR"]) / len(results.judge_results) if results.judge_results else 0
            })
            
            console.print(f"[green]Experiment completed in {total_time:.1f}s[/green]")
            
            return results
            
        except Exception as e:
            console.print(f"[red]Experiment failed: {e}[/red]")
            results.metadata["error"] = str(e)
            return results
    
    def run_comparative_experiment(self, base_config: ExperimentConfig, 
                                 variations: List[Dict[str, Any]]) -> Dict[str, ExperimentResults]:
        """
        Run comparative experiments with different configurations
        
        Args:
            base_config: Base experiment configuration
            variations: List of configuration variations to test
            
        Returns:
            Dictionary mapping variation names to results
        """
        console.print(f"[bold blue]Running comparative experiment with {len(variations)} variations[/bold blue]")
        
        comparative_results = {}
        
        for i, variation in enumerate(variations):
            variation_name = variation.get("name", f"variation_{i+1}")
            console.print(f"[blue]Running comparative variation: {variation_name}[/blue]")
            
            # Create modified config
            modified_config = self._apply_config_variation(base_config, variation)
            modified_config.experiment_id = f"{base_config.experiment_id}_{variation_name}"
            
            # Run experiment
            result = self.run_experiment(modified_config)
            comparative_results[variation_name] = result
        
        # Generate comparison analysis
        comparison_analysis = self._analyze_comparative_results(comparative_results)
        
        return {
            **comparative_results,
            "_comparison_analysis": comparison_analysis
        }
    
    def save_experiment_results(self, results: ExperimentResults, output_path: str) -> None:
        """Save results with full experiment metadata"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = self._make_serializable(asdict(results))
        
        # Save main results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save detailed generation results
        gen_results_file = output_file.parent / f"{output_file.stem}_generation_details.json"
        generation_details = [asdict(r) for r in results.generation_results]
        with open(gen_results_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_serializable(generation_details), f, indent=2, ensure_ascii=False)
        
        # Save detailed judge results
        judge_results_file = output_file.parent / f"{output_file.stem}_judge_details.json"
        judge_details = [asdict(r) for r in results.judge_results]
        with open(judge_results_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_serializable(judge_details), f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        self._generate_summary_report(results, output_file.parent / f"{output_file.stem}_summary.txt")
        
        console.print(f"[green]Results saved to {output_path}[/green]")
    
    def validate_experiment_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Validate experiment configuration"""
        validation = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check required fields
        if not config.experiment_id:
            validation["errors"].append("experiment_id is required")
        
        if not config.experiment_type:
            validation["errors"].append("experiment_type is required")
        
        # Check data sources
        if not config.needles and config.experiment_type != "longmemeval":
            validation["warnings"].append("No needles specified")
        
        if not config.haystacks and config.experiment_type != "longmemeval":
            validation["warnings"].append("No haystacks specified")
        
        # Check variations
        if not any([config.retrieval_k, config.reranking, config.context_types]):
            validation["warnings"].append("No experimental variations specified")
        
        # Check output directory
        if config.results_dir:
            try:
                Path(config.results_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation["errors"].append(f"Cannot create results directory: {e}")
        
        # Check API budget estimation
        estimated_calls = self._estimate_api_calls(config)
        if estimated_calls > 5000:
            validation["warnings"].append(f"Estimated {estimated_calls} API calls - this may be expensive")
        
        validation["valid"] = len(validation["errors"]) == 0
        validation["estimated_api_calls"] = estimated_calls
        
        return validation
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        status = {
            "pipeline_config": asdict(self.config),
            "component_status": {},
            "system_status": {}
        }
        
        # Test component status
        try:
            status["component_status"]["embedding_storage"] = self.embedding_manager.get_embedding_stats()
            status["component_status"]["reranker"] = self.reranker.get_model_stats()
            status["component_status"]["generation"] = self.generation.get_generation_stats()
        except Exception as e:
            status["component_status"]["error"] = str(e)
        
        # Test API connections
        try:
            status["system_status"]["api_connections"] = self.generation.test_connections()
        except Exception as e:
            status["system_status"]["connection_error"] = str(e)
        
        # List available collections
        try:
            status["system_status"]["available_collections"] = self.embedding_manager.list_collections()
        except Exception as e:
            status["system_status"]["collections_error"] = str(e)
        
        return status
    
    def _warm_up_models(self):
        """Warm up all models for better performance"""
        console.print("[blue]Warming up models...[/blue]")
        
        try:
            self.embedding_manager.warm_up_model()
            self.reranker.warm_up_model()
        except Exception as e:
            console.print(f"[yellow]Model warmup warning: {e}[/yellow]")
    
    def _prepare_experiment_data(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Load and prepare all data for experiment"""
        data = {
            "needles": [],
            "documents": [],
            "distractors": [],
            "conversations": []
        }
        
        # Load needles
        for needle_file in config.needles:
            if Path(needle_file).suffix == '.json':
                needles = self.doc_processor.load_chroma_needles(needle_file)
                data["needles"].extend(needles)
            else:
                # Assume it's a document file
                doc = self.doc_processor.load_document(needle_file)
                data["documents"].append(doc)
        
        # Load haystacks
        for haystack in config.haystacks:
            if Path(haystack).is_dir():
                # Load all documents from directory
                doc_files = list(Path(haystack).glob("*.txt")) + list(Path(haystack).glob("*.md"))
                docs = self.doc_processor.load_documents([str(f) for f in doc_files])
                data["documents"].extend(docs)
            else:
                # Single document
                doc = self.doc_processor.load_document(haystack)
                data["documents"].append(doc)
        
        # Load distractors if specified
        for distractor_file in config.distractors:
            distractors = self.doc_processor.load_chroma_distractors(distractor_file)
            data["distractors"].extend(distractors)
        
        return data
    
    def _create_experiment_collections(self, data: Dict[str, Any], 
                                     config: ExperimentConfig) -> Dict[str, Any]:
        """Create ChromaDB collections for experiment"""
        collections = {}
        
        # Process documents based on experiment requirements
        for doc in track(data["documents"], description="Processing documents..."):
            
            # Create variations based on experiment type
            if config.experiment_type == "structure_impact":
                # Create shuffled versions
                for intensity in config.shuffle_types:
                    shuffled_doc = self.doc_processor.shuffle_document(doc, intensity)
                    collection_name = f"{config.experiment_id}_{doc.doc_id}_{intensity}"
                    
                    chunks = self.doc_processor.process_documents([shuffled_doc])
                    collection = self.embedding_manager.create_collection(
                        collection_name,
                        {"experiment_id": config.experiment_id, "doc_id": doc.doc_id, "shuffle": intensity}
                    )
                    self.embedding_manager.batch_embed_and_store(chunks, collection)
                    collections[collection_name] = collection
            
            elif config.experiment_type == "distractor_impact":
                # Create distractor variations
                for distractor_count in config.distractor_counts:
                    if distractor_count == 0:
                        processed_doc = doc
                    else:
                        processed_doc = self.doc_processor.inject_distractors(
                            doc, data["distractors"], distractor_count
                        )
                    
                    collection_name = f"{config.experiment_id}_{doc.doc_id}_dist_{distractor_count}"
                    
                    chunks = self.doc_processor.process_documents([processed_doc])
                    collection = self.embedding_manager.create_collection(
                        collection_name,
                        {"experiment_id": config.experiment_id, "doc_id": doc.doc_id, "distractors": distractor_count}
                    )
                    self.embedding_manager.batch_embed_and_store(chunks, collection)
                    collections[collection_name] = collection
            
            else:
                # Standard processing
                collection_name = f"{config.experiment_id}_{doc.doc_id}"
                
                chunks = self.doc_processor.process_documents([doc])
                collection = self.embedding_manager.create_collection(
                    collection_name,
                    {"experiment_id": config.experiment_id, "doc_id": doc.doc_id}
                )
                self.embedding_manager.batch_embed_and_store(chunks, collection)
                collections[collection_name] = collection
        
        return collections
    
    def _generate_experiment_variations(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate all experiment variations from config"""
        variations = []
        
        # Create all combinations of parameters
        param_combinations = list(product(
            config.retrieval_k,
            config.reranking,
            config.context_types,
            config.context_window_sizes
        ))
        
        for i, (k, rerank, context_type, context_size) in enumerate(param_combinations):
            variation = {
                "name": f"k{k}_rerank{rerank}_{context_type}_ctx{context_size}",
                "retrieval_k": k,
                "reranking": rerank,
                "context_type": context_type,
                "context_window_size": context_size,
                "variation_id": i
            }
            variations.append(variation)
        
        return variations
    
    def _run_generation_variation(self, variation: Dict[str, Any], 
                                collections: Dict[str, Any],
                                data: Dict[str, Any]) -> List[GenerationResult]:
        """Run generation for a specific variation"""
        results = []
        
        # For each needle/query
        for needle in data["needles"]:
            query = needle.get("question", "")
            if not query:
                continue
            
            # For each collection/document
            for collection_name, collection in collections.items():
                try:
                    if variation["context_type"] == "rag":
                        # RAG-based generation
                        retrieved = self.retrieval_engine.retrieve_chunks(
                            query, collection, variation["retrieval_k"]
                        )
                        
                        if variation["reranking"] and retrieved:
                            retrieved = self.reranker.rerank_results(query, retrieved)
                        
                        context = self.context_assembly.assemble_rag_context(
                            retrieved, query
                        )
                        
                    elif variation["context_type"] == "full":
                        # Full document context
                        # Find original document for this collection
                        doc_id = collection_name.split('_')[-1]  # Extract doc_id
                        doc = next((d for d in data["documents"] if d.doc_id == doc_id), None)
                        
                        if doc:
                            context = self.context_assembly.assemble_full_context(doc, query)
                        else:
                            continue
                    
                    else:
                        # Other context types (focused, conversation)
                        continue  # Skip for now
                    
                    # Generate response
                    gen_result = self.generation.generate_response(
                        context.context, query,
                        max_tokens=variation.get("max_tokens", 1000)
                    )
                    
                    # Add variation metadata
                    gen_result.metadata.update({
                        "variation": variation,
                        "collection_name": collection_name,
                        "needle_id": needle.get("id", ""),
                        "context_stats": self.context_assembly.calculate_context_stats(context.context)
                    })
                    
                    results.append(gen_result)
                    
                except Exception as e:
                    console.print(f"[red]Generation failed for {collection_name}: {e}[/red]")
                    continue
        
        return results
    
    def _run_evaluation_variation(self, generation_results: List[GenerationResult],
                                variation: Dict[str, Any],
                                config: ExperimentConfig) -> List[JudgeResult]:
        """Run evaluation for generation results"""
        if not generation_results:
            return []
        
        evaluations = []
        for gen_result in generation_results:
            # Find correct answer from needle data
            needle_id = gen_result.metadata.get("needle_id", "")
            # This would need to be matched with actual needle data
            correct_answer = "TODO: Extract from needle data"  # Placeholder
            
            evaluation = {
                "response": gen_result.response,
                "question": gen_result.query,
                "correct_answer": correct_answer,
                "experiment_type": config.experiment_type
            }
            evaluations.append(evaluation)
        
        # Batch evaluate
        judge_results = self.generation.batch_evaluate(evaluations)
        
        # Add variation metadata to judge results
        for judge_result in judge_results:
            judge_result.metadata.update({"variation": variation})
        
        return judge_results
    
    def _calculate_summary_statistics(self, results: ExperimentResults) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        if not results.judge_results:
            return {"error": "No evaluation results available"}
        
        # Basic statistics
        total_evaluations = len(results.judge_results)
        correct_responses = len([j for j in results.judge_results if j.is_correct])
        success_rate = correct_responses / total_evaluations if total_evaluations > 0 else 0
        
        # Performance by variation
        variation_stats = {}
        for judge_result in results.judge_results:
            variation_name = judge_result.metadata.get("variation", {}).get("name", "unknown")
            if variation_name not in variation_stats:
                variation_stats[variation_name] = {"correct": 0, "total": 0}
            
            variation_stats[variation_name]["total"] += 1
            if judge_result.is_correct:
                variation_stats[variation_name]["correct"] += 1
        
        # Calculate success rates by variation
        for var_name, stats in variation_stats.items():
            stats["success_rate"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        return {
            "total_evaluations": total_evaluations,
            "correct_responses": correct_responses,
            "overall_success_rate": success_rate,
            "variation_performance": variation_stats,
            "best_variation": max(variation_stats.items(), key=lambda x: x[1]["success_rate"])[0] if variation_stats else None,
            "worst_variation": min(variation_stats.items(), key=lambda x: x[1]["success_rate"])[0] if variation_stats else None
        }
    
    def _save_intermediate_results(self, results: ExperimentResults, 
                                 config: ExperimentConfig, variation_idx: int):
        """Save intermediate results during experiment"""
        intermediate_path = Path(config.results_dir) / f"{config.experiment_id}_intermediate_{variation_idx}.json"
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save partial results
        partial_results = {
            "experiment_id": results.experiment_id,
            "variations_completed": variation_idx + 1,
            "generation_results_count": len(results.generation_results),
            "judge_results_count": len(results.judge_results),
            "timestamp": time.time()
        }
        
        with open(intermediate_path, 'w') as f:
            json.dump(partial_results, f, indent=2)
    
    def _generate_summary_report(self, results: ExperimentResults, output_path: Path):
        """Generate human-readable summary report"""
        with open(output_path, 'w') as f:
            f.write(f"Experiment Summary Report\n")
            f.write(f"========================\n\n")
            f.write(f"Experiment ID: {results.experiment_id}\n")
            f.write(f"Experiment Type: {results.config.experiment_type}\n")
            f.write(f"Total Duration: {results.metadata.get('total_duration_seconds', 0):.1f} seconds\n\n")
            
            # Summary statistics
            stats = results.summary_stats
            if stats and "overall_success_rate" in stats:
                f.write(f"Overall Results:\n")
                f.write(f"- Total Evaluations: {stats['total_evaluations']}\n")
                f.write(f"- Correct Responses: {stats['correct_responses']}\n") 
                f.write(f"- Success Rate: {stats['overall_success_rate']:.2%}\n\n")
                
                # Best and worst variations
                if stats.get('best_variation'):
                    f.write(f"Best Variation: {stats['best_variation']}\n")
                if stats.get('worst_variation'):
                    f.write(f"Worst Variation: {stats['worst_variation']}\n\n")
                
                # Variation performance
                f.write("Performance by Variation:\n")
                for var_name, var_stats in stats['variation_performance'].items():
                    f.write(f"- {var_name}: {var_stats['success_rate']:.2%} ({var_stats['correct']}/{var_stats['total']})\n")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _estimate_api_calls(self, config: ExperimentConfig) -> int:
        """Estimate total API calls for experiment"""
        # Basic estimation
        needles_count = len(config.needles) if config.needles else 5
        haystacks_count = len(config.haystacks) if config.haystacks else 2
        variations = len(config.retrieval_k) * len(config.reranking) * len(config.context_types)
        
        generation_calls = needles_count * haystacks_count * variations
        judge_calls = generation_calls  # 1:1 ratio
        
        return generation_calls + judge_calls
    
    def _apply_config_variation(self, base_config: ExperimentConfig, 
                              variation: Dict[str, Any]) -> ExperimentConfig:
        """Apply variation to base config"""
        # Create a copy of the base config
        new_config = ExperimentConfig(
            experiment_id=base_config.experiment_id,
            experiment_type=base_config.experiment_type,
            **variation
        )
        return new_config
    
    def _analyze_comparative_results(self, results: Dict[str, ExperimentResults]) -> Dict[str, Any]:
        """Analyze comparative experiment results"""
        analysis = {
            "total_variations": len(results),
            "comparison_metrics": {}
        }
        
        for name, result in results.items():
            if name.startswith("_"):  # Skip analysis entries
                continue
                
            stats = result.summary_stats
            analysis["comparison_metrics"][name] = {
                "success_rate": stats.get("overall_success_rate", 0),
                "total_evaluations": stats.get("total_evaluations", 0)
            }
        
        # Find best and worst
        if analysis["comparison_metrics"]:
            best = max(analysis["comparison_metrics"].items(), key=lambda x: x[1]["success_rate"])
            worst = min(analysis["comparison_metrics"].items(), key=lambda x: x[1]["success_rate"])
            
            analysis["best_overall"] = {"name": best[0], **best[1]}
            analysis["worst_overall"] = {"name": worst[0], **worst[1]}
        
        return analysis
    
    def _calculate_experiment_retrieval_metrics(self, results: ExperimentResults, 
                                              ground_truth: Dict) -> Dict[str, float]:
        """Calculate retrieval metrics if ground truth is available"""
        # Placeholder for retrieval metrics calculation
        return {"mrr": 0.0, "precision@5": 0.0, "recall@5": 0.0}