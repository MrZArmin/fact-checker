from typing import List, Dict, Any
import numpy as np
from django.core.management.base import BaseCommand
from django.db import connection
from tqdm import tqdm
import gc
from openai import OpenAI
import time
import os

class OpenAIEmbedder:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "text-embedding-3-small"
        self.batch_size = 1000
        self.retry_attempts = 3
        self.retry_delay = 1

    def get_embeddings(self, texts: List[str], desc: str = "Generating embeddings") -> np.ndarray:
        embeddings = []
        
        # Create progress bar for embedding batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc, unit='batch'):
            batch_texts = texts[i:i + self.batch_size]
            
            for attempt in range(self.retry_attempts):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                        encoding_format="float"
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [data.embedding for data in response.data]
                    embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise e
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

        return np.array(embeddings)

class Command(BaseCommand):
    help = 'Generate embeddings for semantic chunks using OpenAI text-embedding-3-small'

    def __init__(self):
        super().__init__()
        self.embedder = None
        self.batch_size = 1000

    def handle(self, *args, **options):
        if not os.getenv('OPENAI_API_KEY'):
            self.stderr.write(self.style.ERROR("OPENAI_API_KEY environment variable not set"))
            return

        self.embedder = OpenAIEmbedder()

        try:
            self.process_chunks()
        finally:
            gc.collect()

    def get_total_chunks(self) -> int:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*)
                FROM semantic_chunks
                WHERE embedding_openai IS NULL
            """)
            return cursor.fetchone()[0]

    def fetch_chunks(self, offset: int) -> List[Dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, text
                FROM semantic_chunks
                WHERE embedding_openai IS NULL
                ORDER BY id
                LIMIT %s OFFSET %s
            """, [self.batch_size, offset])

            return [
                {'id': row[0], 'text': row[1]}
                for row in cursor.fetchall()
            ]

    def save_embeddings(self, chunk_ids: List[int], embeddings: np.ndarray):
        with connection.cursor() as cursor:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cursor.execute("""
                    UPDATE semantic_chunks
                    SET embedding_openai = %s
                    WHERE id = %s
                """, [embedding.tolist(), chunk_id])
        connection.commit()

    def process_chunks(self):
        total_chunks = self.get_total_chunks()
        self.stdout.write(f"Found {total_chunks} chunks to process")

        offset = 0
        total_processed = 0

        # Create main progress bar for overall progress
        with tqdm(total=total_chunks, desc="Overall progress", unit='chunk') as pbar:
            while True:
                chunks = self.fetch_chunks(offset)
                if not chunks:
                    break

                chunk_ids = [chunk['id'] for chunk in chunks]
                texts = [chunk['text'] for chunk in chunks]

                try:
                    # Pass custom description to nested progress bar
                    batch_desc = f"Batch {offset//self.batch_size + 1}"
                    embeddings = self.embedder.get_embeddings(texts, desc=batch_desc)
                    self.save_embeddings(chunk_ids, embeddings)

                    batch_size = len(chunks)
                    total_processed += batch_size
                    pbar.update(batch_size)

                except Exception as e:
                    self.stderr.write(f"Error processing batch at offset {offset}: {str(e)}")

                offset += self.batch_size
                gc.collect()

        self.stdout.write(self.style.SUCCESS(f"Finished processing {total_processed} chunks"))