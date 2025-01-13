from typing import List, Dict, Any
import numpy as np
from django.core.management.base import BaseCommand
from django.db import connection
from openai import OpenAI
from tqdm import tqdm
import time
from factChecker.models import Chunk
from dotenv import load_dotenv
import os

class Command(BaseCommand):
    help = 'Generate OpenAI embeddings for article chunks'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.batch_size = 100
        self.model = "text-embedding-3-small"
    
    def handle(self, *args, **options):        
        # Get chunks without OpenAI embeddings
        chunks = Chunk.objects.filter(
            embedding_openai__isnull=True,
            text__isnull=False
        ).iterator()
        
        # Process in batches
        batch = []
        total_processed = 0
        
        for chunk in tqdm(chunks, desc="Processing chunks"):
            batch.append(chunk)
            
            if len(batch) >= self.batch_size:
                self._process_batch(batch)
                total_processed += len(batch)
                batch = []
                time.sleep(0.5)  # Rate limiting
        
        # Process remaining chunks
        if batch:
            self._process_batch(batch)
            total_processed += len(batch)
        
        self.stdout.write(
            self.style.SUCCESS(f"Successfully processed {total_processed} chunks")
        )
    
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
    
    def _process_batch(self, chunks: List[Chunk]) -> None:
        """Process a batch of chunks."""
        # Prepare texts for embedding
        texts = [chunk.text for chunk in chunks]
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        # Update chunks with embeddings
        if embeddings:
            with connection.cursor() as cursor:
                for chunk, embedding in zip(chunks, embeddings):
                    cursor.execute(
                        """
                        UPDATE chunks 
                        SET embedding_openai = %s 
                        WHERE id = %s
                        """,
                        [embedding.tolist(), chunk.id]
                    )