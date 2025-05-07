import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from django.db import connection
from tqdm import tqdm

from factChecker.models import Benchmark, Article, GraphBenchmark
from factChecker.services.ragServiceOpenAi import RAGServiceOpenAI
from factChecker.services.ragVariations import RagVariations

class BenchmarkResult:
    """Data class to store benchmark results"""
    def __init__(self, question: str, expected_answer: bool, result: Optional[bool]):
        self.question = question
        self.expected_answer = expected_answer
        self.result = result
        self.status = self._determine_status()
    
    def _determine_status(self) -> str:
        if self.result is None:
            return 'Uncertain'
        return 'Correct' if self.result == self.expected_answer else 'Incorrect'
    
    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'expected_answer': self.expected_answer,
            'result': self.result,
            'status': self.status
        }

class BenchmarkVisualizer:
    """Handles the visualization of benchmark results"""
    def __init__(self, results: List[BenchmarkResult], folder_name: str, benchmark_name: str):
        self.results = results
        self.folder_name = folder_name
        self.benchmark_name = benchmark_name

    def generate_charts(self) -> None:
        """Generate all charts and statistics"""
        stats = self._calculate_statistics()
        self._create_pie_chart(stats, self.folder_name)
        self._create_bar_chart(stats, self.folder_name)
        self._save_statistics(stats, self.folder_name)

    def _calculate_statistics(self) -> Dict:
        return {
            'correct': sum(1 for r in self.results if r.status == 'Correct'),
            'incorrect': sum(1 for r in self.results if r.status == 'Incorrect'),
            'uncertain': sum(1 for r in self.results if r.status == 'Uncertain')
        }

    def _create_pie_chart(self, stats: Dict, folder: str) -> None:
        plt.figure(figsize=(10, 6))
        labels = ['Correct', 'Incorrect', 'Uncertain']
        sizes = [stats[k.lower()] for k in labels]
        colors = ['#2ecc71', '#e74c3c', '#3498db']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=140, shadow=True)
        plt.axis('equal')
        plt.title(f'Benchmark Results Distribution For {self.benchmark_name}')
        plt.savefig(os.path.join(folder, 'benchmark_results_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_bar_chart(self, stats: Dict, folder: str) -> None:
        plt.figure(figsize=(10, 6))
        labels = ['Correct', 'Incorrect', 'Uncertain']
        sizes = [stats[k.lower()] for k in labels]
        colors = ['#2ecc71', '#e74c3c', '#3498db']

        bars = plt.bar(labels, sizes, color=colors)
        plt.title(f'Benchmark Results by Category For {self.benchmark_name}')
        plt.xlabel('Result Type')
        plt.ylabel('Count')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        plt.savefig(os.path.join(folder, 'benchmark_results_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_statistics(self, stats: Dict, folder: str) -> None:
        total = sum(stats.values())
        total_excluding_uncertain = total - stats['uncertain']
        
        if total_excluding_uncertain > 0:
            accuracy = (stats['correct'] / total_excluding_uncertain) * 100
        else:
            accuracy = 0.0

        with open(os.path.join(folder, 'statistics.txt'), 'w') as f:
            f.write(f"Benchmark Statistics of {self.benchmark_name}\n")
            f.write(f"==================\n")
            f.write(f"Total Questions: {total}\n")
            f.write(f"Correct: {stats['correct']}\n")
            f.write(f"Incorrect: {stats['incorrect']}\n")
            f.write(f"Uncertain: {stats['uncertain']}\n")
            f.write(f"Accuracy (excluding uncertain): {accuracy:.2f}%\n")
            if total > 0:
                f.write(f"Uncertainty Rate: {(stats['uncertain']/total)*100:.2f}%\n")

class BenchmarkRunner:
    """Handles the execution and analysis of benchmarks"""
    def __init__(self, rag_service: RAGServiceOpenAI):
        self.rag_service = rag_service
        self.results: List[BenchmarkResult] = []

    def run_benchmarks(self, benchmarks: List[Benchmark]) -> None:
        for benchmark in tqdm(benchmarks, desc="Running benchmarks"):
            try:
                result = self.rag_service.query(benchmark.question)
                benchmark_result = BenchmarkResult(
                    question=benchmark.question,
                    expected_answer=benchmark.answer,
                    result=result
                )
                self.results.append(benchmark_result)
            except Exception as e:
                logging.error(f"Failed to run benchmark: {str(e)}")
                continue

    def calculate_accuracy(self) -> float:
        valid_results = [r for r in self.results if r.status != 'Uncertain']
        if not valid_results:
            return 0.0
        correct_count = sum(1 for r in valid_results if r.status == 'Correct')
        return (correct_count / len(valid_results)) * 100

class Command(BaseCommand):
    help = 'Run and analyze RAG service benchmarks with enhanced visualization and error handling'

    def add_arguments(self, parser):
        parser.add_argument('benchmark_name', type=str, help='Name of the benchmark run')
        parser.add_argument('--skip-charts', action='store_true', help='Skip generating charts')
        parser.add_argument(
            '--rag-variation',
            type=str,
            choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
            help='RAG variation to use. If not specified, uses default RAG service.'
        )

    def handle(self, *args, **options):
        try:
            # Setup output directory
            base_folder = os.path.join('benchmark', options["benchmark_name"])
            os.makedirs(base_folder, exist_ok=True)
            
            # Initialize services based on RAG variation
            if options.get('rag_variation'):
                rag_variations = RagVariations()
                variation_mapping = {
                    'v1': rag_variations.query_v1_baseline,
                    'v2': rag_variations.query_v2_semantic_chunks,
                    'v3': rag_variations.query_v3_semantic_chunks_with_imporved_prompt,
                    'v4': rag_variations.query_v4_with_reranking,
                    'v5': rag_variations.query_v5_mxbai,
                    'v6': rag_variations.query_v6_opeani_reranking_graph,
                }
                query_func = variation_mapping[options['rag_variation']]
                
                # Create a wrapper class to match RagServiceOpenai interface
                class RagVariationWrapper:
                    def __init__(self, query_func):
                        self.query_func = query_func
                    
                    def query(self, question: str) -> Optional[bool]:
                        try:
                            result = self.query_func(question)
                            return self._parse_response(result['response'])
                        except Exception as e:
                            logging.error(f"Query failed: {str(e)}")
                            return None
                    
                    def _parse_response(self, response: str) -> Optional[bool]:
                        try:
                            response_json = json.loads(response)
                            return response_json.get('result')
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse response: {response}")
                            return None
                
                rag_service = RagVariationWrapper(query_func)
            else:
                rag_service = RAGServiceOpenAI()
            
            runner = BenchmarkRunner(rag_service)
            
            # Run benchmarks
            self.stdout.write(self.style.SUCCESS("Running benchmarks..."))
            benchmarks = Benchmark.objects.all()
            
            if options['rag_variation'] == 'v6':
                benchmarks = GraphBenchmark.objects.all()
            runner.run_benchmarks(benchmarks)
            
            # Save results with variation info
            results_with_variation = {
                "name": options['benchmark_name'],
                "rag_variation": options.get('rag_variation', 'default'),
                "questions": [r.to_dict() for r in runner.results],
                "accuracy": runner.calculate_accuracy()
            }
            
            with open(os.path.join(base_folder, f'{options["benchmark_name"]}_results.json'), 'w', encoding='utf-8') as f:
                json.dump(results_with_variation, f, indent=4, ensure_ascii=False)
            
            # Generate visualizations
            if not options['skip_charts']:
                self.stdout.write(self.style.SUCCESS("Generating benchmark visualizations..."))
                visualizer = BenchmarkVisualizer(runner.results, base_folder, options['benchmark_name'])
                visualizer.generate_charts()
            
            # Display final results
            accuracy = runner.calculate_accuracy()
            self.stdout.write(self.style.SUCCESS(f"\nBenchmark completed successfully!"))
            self.stdout.write(f"RAG Variation: {options.get('rag_variation', 'default')}")
            self.stdout.write(f"Accuracy: {accuracy:.2f}%")
            self.stdout.write(f"Results saved to: {base_folder}")
            
        except Exception as e:
            logging.error(f"Benchmark failed: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Benchmark failed: {str(e)}"))
            raise