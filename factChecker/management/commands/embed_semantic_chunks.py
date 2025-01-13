import torch
from typing import List, Dict, Any
import numpy as np
from transformers import AutoModel, AutoTokenizer
from django.core.management.base import BaseCommand
from django.db import connection
from tqdm import tqdm
import gc

class MXBaiEmbedder:
    def __init__(self):
        self.model_name = "mixedbread-ai/mxbai-embed-large-v1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.batch_size = 32

    def _batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

    def get_embeddings(self, texts: List[str], desc: str = "Generating embeddings") -> np.ndarray:
        embeddings = []

        # Create progress bar for embedding batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc, unit='batch'):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self._batch_encode(batch_texts)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.extend(batch_embeddings)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.array(embeddings)

class Command(BaseCommand):
    help = 'Generate embeddings for semantic chunks using MXBai Embed Large V1'

    def __init__(self):
        super().__init__()
        self.embedder = None
        self.batch_size = 1000  # DB batch size

    def handle(self, *args, **options):
        self.print_cuda_info()
        self.embedder = MXBaiEmbedder()

        try:
            self.process_chunks()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def print_cuda_info(self):
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            self.stdout.write(self.style.SUCCESS(f"CUDA is available. Number of devices: {num_devices}"))
            for i in range(num_devices):
                device_name = torch.cuda.get_device_name(i)
                self.stdout.write(f"Device {i}: {device_name}")
        else:
            self.stdout.write(self.style.WARNING("CUDA is not available. Using CPU."))

    def get_total_chunks(self) -> int:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*)
                FROM semantic_chunks
                WHERE embedding IS NULL
            """)
            return cursor.fetchone()[0]

    def fetch_chunks(self, offset: int) -> List[Dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, text
                FROM semantic_chunks
                WHERE embedding IS NULL
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
                    SET embedding = %s
                    WHERE id = %s
                """, [embedding.tolist(), chunk_id])
        connection.commit()

    def process_chunks(self):
        # Get total number of chunks to process for the progress bar
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

                # Memory management
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.stdout.write(self.style.SUCCESS(f"Finished processing {total_processed} chunks"))
