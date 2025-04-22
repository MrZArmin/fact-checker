from django.core.management.base import BaseCommand
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from factChecker.models import Article
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Create optimized knowledge graphs from articles'
    PROGRESS_FILE = 'article_graph_progress.json'
    BATCH_SIZE = 25
    MAX_WORKERS = 4

    def __init__(self):
        super().__init__()
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # High chunk size for better context
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm
        )

    def preprocess_article(self, article: Article) -> str:
        """Preprocess article text to improve graph quality."""
        # Combine text with special separators for better context
        return f"""Title: {article.title.strip()}
        Lead: {article.lead.strip()}
        Keywords: {', '.join(k.name for k in article.keywords.all())}
        Content: {article.text.strip()}
        """

    def create_document_batch(self, articles: List[Article]) -> List[Document]:
        """Create optimized document batch from articles."""
        documents = []

        for article in articles:
            preprocessed_text = self.preprocess_article(article)
            doc = Document(
                page_content=preprocessed_text,
                metadata={
                    "source": f"article_{article.id}",
                    "date": str(article.date),
                    "tags": article.tags,
                    "keywords": [k.name for k in article.keywords.all()]
                }
            )
            documents.append(doc)

        # Split documents with overlap for better context preservation
        return self.text_splitter.split_documents(documents)

    def process_batch(self, batch: List[Article]):
        try:
            # Create optimized document batch
            batch_documents = self.create_document_batch(batch)

            # Convert to graph documents with retries
            graph_documents = self.llm_transformer.convert_to_graph_documents(
                batch_documents
            )

            # Add to graph with relationship metadata
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            return [article.id for article in batch]

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Batch processing error: {str(e)}'))
            return []

    def handle(self, *args, **options):
        processed_ids = self.load_processed_ids()
        total_articles = Article.objects.exclude(id__in=processed_ids).count()

        self.stdout.write(f'Processing {total_articles} articles in batches of {self.BATCH_SIZE}...')

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = []
            offset = 0

            # Create progress bar for total batches
            total_batches = (total_articles + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            with tqdm(total=total_batches, desc='Processing batches') as pbar:
                while offset < total_articles:
                    # Get next batch of articles
                    batch = self.get_article_batch(processed_ids, offset)
                    if not batch:
                        break

                    future = executor.submit(self.process_batch, batch)
                    futures.append(future)

                    # Process completed futures
                    for completed_future in [f for f in futures if f.done()]:
                        processed_ids.extend(completed_future.result())
                        self.save_processed_ids(processed_ids)
                        futures.remove(completed_future)
                        pbar.update(1)

                    offset += self.BATCH_SIZE

                # Process any remaining futures
                for future in futures:
                    processed_ids.extend(future.result())
                    self.save_processed_ids(processed_ids)
                    pbar.update(1)

        self.stdout.write(self.style.SUCCESS('Finished processing all articles'))

    def get_article_batch(self, processed_ids: List[int], offset: int) -> List[Article]:
        """Get a batch of unprocessed articles."""
        return list(Article.objects.exclude(
            id__in=processed_ids
        ).prefetch_related(
            'keywords'
        )[offset:offset + self.BATCH_SIZE])

    def load_processed_ids(self):
        """Load processed article IDs with error handling."""
        try:
            if os.path.exists(self.PROGRESS_FILE):
                with open(self.PROGRESS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'Error loading progress: {str(e)}'))
        return []

    def save_processed_ids(self, processed_ids):
        """Save processed article IDs with backup."""
        try:
            # Save with temporary file first
            temp_file = f'{self.PROGRESS_FILE}.tmp'
            with open(temp_file, 'w') as f:
                json.dump(processed_ids, f)

            # Rename to actual file
            os.replace(temp_file, self.PROGRESS_FILE)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error saving progress: {str(e)}'))
