from django.core.management.base import BaseCommand
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from factChecker.models import Article
import os
import json
import time
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Create knowledge graphs from all articles'

    PROGRESS_FILE = 'article_graph_progress.json'

    def handle(self, *args, **options):

        # Load processed article IDs
        processed_ids = self.load_processed_ids()

        # Initialize processing components
        graph = Neo4jGraph()
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        llm_transformer = LLMGraphTransformer(llm=llm)

        # Get all unprocessed articles
        articles = Article.objects.exclude(id__in=processed_ids)
        total_articles = articles.count()

        self.stdout.write(f'Processing {total_articles} unprocessed articles...')

        # Use tqdm for progress bar
        for article in tqdm(articles, desc='Processing Articles', total=total_articles):
            try:
                # Create document from article
                doc = Document(
                    page_content=article.title + " " + article.lead + " " + article.text,
                    metadata={"source": f"article_{article.id}"}
                )

                # Split document
                documents = text_splitter.split_documents([doc])

                # Convert to graph documents
                graph_documents = llm_transformer.convert_to_graph_documents(documents)

                # Add to graph
                graph.add_graph_documents(
                    graph_documents,
                    baseEntityLabel=True,
                    include_source=True
                )

                # Track processed article
                processed_ids.append(article.id)
                self.save_processed_ids(processed_ids)

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error processing article {article.id}: {str(e)}')
                )

        self.stdout.write(self.style.SUCCESS('Finished processing all unprocessed articles'))

    def load_processed_ids(self):
        """Load processed article IDs from file."""
        if os.path.exists(self.PROGRESS_FILE):
            with open(self.PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return []

    def save_processed_ids(self, processed_ids):
        """Save processed article IDs to file."""
        with open(self.PROGRESS_FILE, 'w') as f:
            json.dump(processed_ids, f)
