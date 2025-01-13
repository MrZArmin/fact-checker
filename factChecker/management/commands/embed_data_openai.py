from typing import List, Dict, Any
import numpy as np
from django.core.management.base import BaseCommand
from django.db import connection
from openai import OpenAI
from tqdm import tqdm
import time
from django.conf import settings
from factChecker.models import Article
from dotenv import load_dotenv
import os

class Command(BaseCommand):
    help = 'Generate OpenAI embeddings for articles without embeddings'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.batch_size = 100
        self.model = "text-embedding-3-small"
    
    def handle(self, *args, **options):        
        # Get articles without OpenAI embeddings
        articles = Article.objects.filter(
            embedding_openai__isnull=True,
            text__isnull=False
        ).iterator()
        
        # Process in batches
        batch = []
        total_processed = 0
        
        for article in tqdm(articles, desc="Processing articles"):
            batch.append(article)
            
            if len(batch) >= self.batch_size:
                self._process_batch(batch)
                total_processed += len(batch)
                batch = []
                time.sleep(0.5)  # Rate limiting
        
        # Process remaining articles
        if batch:
            self._process_batch(batch)
            total_processed += len(batch)
        
        self.stdout.write(
            self.style.SUCCESS(f"Successfully processed {total_processed} articles")
        )
    
    def chunked_list(self, lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks of specified size."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    def prepare_text(self, article: Article) -> str:
        """Prepare article text for embedding by combining relevant fields."""
        components = []
        if article.title:
            components.append(f"Title: {article.title}")
        if article.lead:
            lead = article.lead.replace("\n", " ")
            components.append(f"Lead: {lead}")
        if article.text:
            text = article.text.replace("\n", " ")
            components.append(f"Content: {text}")
        return " ".join(components)
    
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings from OpenAI API with rate limiting."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            return [np.array(data.embedding) for data in response.data]
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting embeddings: {str(e)}"))
            time.sleep(60)  # Wait before retrying
            return []
    
    def _process_batch(self, articles: List[Article]) -> None:
        """Process a batch of articles."""
        # Prepare texts for embedding
        texts = [self.prepare_text(article) for article in articles]
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        # Update articles with embeddings
        if embeddings:
            with connection.cursor() as cursor:
                for article, embedding in zip(articles, embeddings):
                    cursor.execute(
                        """
                        UPDATE articles 
                        SET embedding_openai = %s 
                        WHERE id = %s
                        """,
                        [embedding.tolist(), article.id]
                    )