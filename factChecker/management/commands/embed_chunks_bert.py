from typing import List
import numpy as np
from django.core.management.base import BaseCommand
from django.db import connection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
from factChecker.models import SemanticChunk

class Command(BaseCommand):
    help = 'Generate ModernBERT embeddings for article chunks'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 100
        self.model = None

    def handle(self, *args, **options):
        # Load the model
        self.stdout.write("Loading ModernBERT model...")
        self.model = SentenceTransformer("karsar/ModernBERT-base-hu_v3")
        
        self.stdout.write("Working on a GPU? {}".format(self.model.device))
        
        # Get chunks without ModernBERT embeddings
        chunks = SemanticChunk.objects.filter(
            embedding_modernbert__isnull=True,
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

        # Process remaining chunks
        if batch:
            self._process_batch(batch)
            total_processed += len(batch)

        self.stdout.write(
            self.style.SUCCESS(f"Successfully processed {total_processed} chunks")
        )

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings from ModernBERT model."""
        try:
            embeddings = self.model.encode(texts)
            return [np.array(embedding) for embedding in embeddings]
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting embeddings: {str(e)}"))
            return []

    def _process_batch(self, chunks: List[SemanticChunk]) -> None:
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
                        UPDATE semantic_chunks
                        SET embedding_modernbert = %s
                        WHERE id = %s
                        """,
                        [embedding.tolist(), chunk.id]
                    )