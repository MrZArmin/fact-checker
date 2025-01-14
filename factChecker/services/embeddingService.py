from typing import List
import numpy as np
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModel
from django.db import connection
import os

class EmbeddingService:
    def __init__(self):
        # OpenAI setup
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.openai_model = "text-embedding-3-small"
        
        # MXBAI setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mxbai_tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
        self.mxbai_model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1").to(self.device)

    def get_openai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.openai_model,
                input=text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"OpenAI embedding error: {str(e)}")
            raise

    def get_mxbai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using MXBAI"""
        try:
            inputs = self.mxbai_tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.mxbai_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()[0]
        except Exception as e:
            print(f"MXBAI embedding error: {str(e)}")
            raise

    def find_similar_articles(self, query: str, model: str = "openai", top_k: int = 15) -> List[tuple]:
        """Find similar articles using specified model"""
        if model == "openai":
            embedding = self.get_openai_embedding(query)
            embedding_column = "embedding_openai"
        else:
            embedding = self.get_mxbai_embedding(query)
            embedding_column = "embedding"

        embedding_list = embedding.astype(np.float32).tolist()
        
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, 1 - ({embedding_column} <=> %s::vector) as similarity
                FROM articles
                WHERE {embedding_column} IS NOT NULL
                ORDER BY similarity DESC
                LIMIT %s
                """,
                [embedding_list, top_k]
            )
            return cursor.fetchall()

    def batch_embed_articles(self, model: str = "openai", batch_size: int = 32):
        """Batch process articles for the specified model"""
        embedding_column = "embedding_openai" if model == "openai" else "embedding"
        
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT id, title, lead, text 
                FROM articles 
                WHERE {embedding_column} IS NULL 
                AND text IS NOT NULL
            """)
            articles = cursor.fetchall()

        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            texts = [f"Title: {art[1]}\nLead: {art[2]}\nContent: {art[3]}" for art in batch]
            
            try:
                if model == "openai":
                    response = self.openai_client.embeddings.create(
                        model=self.openai_model,
                        input=texts,
                        encoding_format="float"
                    )
                    embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
                else:
                    embeddings = []
                    for text in texts:
                        embedding = self.get_mxbai_embedding(text)
                        embeddings.append(embedding)

                with connection.cursor() as cursor:
                    for j, embedding in enumerate(embeddings):
                        cursor.execute(
                            f"UPDATE articles SET {embedding_column} = %s WHERE id = %s",
                            [embedding.tolist(), batch[j][0]]
                        )
                        
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {str(e)}")
                continue