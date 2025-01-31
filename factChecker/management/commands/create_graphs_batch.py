from django.core.management.base import BaseCommand
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from factChecker.models import Article
from typing import List, Dict, Optional
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

class ArticleGraphCommand(BaseCommand):
    help = 'Create optimized knowledge graphs from articles'

    # Configuration constants
    PROGRESS_FILE = Path('article_graph_progress.json')
    BATCH_SIZE = 25
    MAX_WORKERS = 4
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 150

    def __init__(self):
        self.current_batch = 0
        try:
            self.graph = Neo4jGraph()
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4-mini",
                batch_size=self.BATCH_SIZE
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        except Exception as e:
            print(f"Failed to initialize components: {str(e)}")
            raise

    def preprocess_article(self, article: Article) -> str:
        """Preprocess article text with enhanced formatting"""
        return f"""Title: {article.title.strip()}
                    Lead: {article.lead.strip()}
                    Keywords: {', '.join(k.name for k in article.keywords.all())}
                    Content: {article.text.strip()}
                    Source: {article.source if hasattr(article, 'source') else 'Unknown'}
                    Date: {article.date}
                """

    def create_document_batch(self, articles: List[Article]) -> List[Document]:
        """Create optimized document batch with metadata enrichment"""
        documents = []
        for article in articles:
            try:
                preprocessed_text = self.preprocess_article(article)
                doc = Document(
                    page_content=preprocessed_text,
                    metadata={
                        "source": f"article_{article.id}",
                        "date": str(article.date),
                        "tags": article.tags,
                        "keywords": [k.name for k in article.keywords.all()],
                        "title": article.title,
                        "processing_batch": self.current_batch
                    }
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error processing article {article.id}: {str(e)}")

        return self.text_splitter.split_documents(documents)

    def process_batch(self, batch: List[Article]) -> tuple[List[int], List[int], Optional[str]]:
        """Process a batch of articles with comprehensive error handling"""
        batch_ids = [article.id for article in batch]
        print(f"\nStarting to process batch {self.current_batch} with articles: {batch_ids}")

        processed_ids = []
        error_ids = []
        error_message = None

        try:
            batch_documents = self.create_document_batch(batch)
            if not batch_documents:
                print(f"Batch {self.current_batch} failed: No documents created")
                return [], batch_ids, "No documents created"

            print(f"Created {len(batch_documents)} document chunks for batch {self.current_batch}")

            graph_documents = self.llm_transformer.convert_to_graph_documents(
                batch_documents,
                batch_mode=True
            )

            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            processed_ids = batch_ids
            print(f"Successfully completed batch {self.current_batch}. Processed articles: {processed_ids}")

        except Exception as e:
            error_message = str(e)
            error_ids = batch_ids
            print(f"Batch {self.current_batch} failed with error: {error_message}")
            print(f"Failed articles: {error_ids}")

        return processed_ids, error_ids, error_message

    def handle(self, *args, **options):
        """Main command execution with improved error handling and progress tracking"""
        try:
            processed_ids = self.load_processed_ids()

            # Get articles to process
            articles = Article.objects.exclude(
                id__in=processed_ids
            ).prefetch_related('keywords').order_by('id')

            total_articles = articles.count()
            if total_articles == 0:
                print("No new articles to process")
                return

            print(f"Processing {total_articles} articles in batches of {self.BATCH_SIZE}")
            print(f"Using {self.MAX_WORKERS} workers for parallel processing")

            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                futures = []
                failed_articles = []

                # Create batches
                for i in range(0, total_articles, self.BATCH_SIZE):
                    batch = list(articles[i:i + self.BATCH_SIZE])
                    self.current_batch = i // self.BATCH_SIZE
                    futures.append(executor.submit(self.process_batch, batch))

                # Process results with progress bar
                with tqdm(total=len(futures), desc="Processing batches") as pbar:
                    for future in as_completed(futures):
                        result_processed_ids, result_error_ids, _ = future.result()
                        processed_ids.extend(result_processed_ids)
                        if result_error_ids:
                            failed_articles.extend(result_error_ids)
                        self.save_processed_ids(processed_ids)
                        pbar.update(1)

            # Final status report
            print("\n=== Processing Summary ===")
            print(f"Total articles processed successfully: {len(processed_ids)}")
            if failed_articles:
                print(f"Total articles failed: {len(failed_articles)}")
                print(f"Failed article IDs: {failed_articles}")

        except Exception as e:
            print(f"Command execution failed: {str(e)}")
            raise

    def load_processed_ids(self) -> List[int]:
        """Load processed article IDs with robust error handling"""
        try:
            if self.PROGRESS_FILE.exists():
                with open(self.PROGRESS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading progress file: {str(e)}")
        return []

    def save_processed_ids(self, processed_ids: List[int]):
        """Save processed article IDs with atomic write operations"""
        temp_file = self.PROGRESS_FILE.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(processed_ids, f)
            temp_file.replace(self.PROGRESS_FILE)
        except Exception as e:
            print(f"Error saving progress: {str(e)}")
            if temp_file.exists():
                temp_file.unlink()
