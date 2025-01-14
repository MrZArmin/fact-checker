from typing import List, Tuple, Dict, Optional
from factChecker.services.embeddingService import EmbeddingService
import numpy as np
from django.db import connection
from django.db.models import Case, When
from factChecker.models import Article

class ArticleRetrieverService:
    def __init__(self):
        # Initialize the EmbeddingService
        self.embedding_service = EmbeddingService()

    def find_similar_articles(self, query: str, model: str = "openai", top_k: int = 15) -> List[Tuple[Article, float]]:
        """
        Find similar articles using vector similarity search with specified embedding model
        
        Args:
            query (str): The search query text
            model (str): The embedding model to use ("openai" or "mxbai")
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[Article, float]]: List of tuples containing (article, similarity_score)
        """
        # Validate model choice
        if model not in ["openai", "mxbai"]:
            raise ValueError("Invalid model. Must be 'openai' or 'mxbai'")

        # Get embedding based on model choice
        if model == "openai":
            query_embedding = self.embedding_service.get_openai_embedding(query)
            embedding_column = "embedding_openai"
        else:
            query_embedding = self.embedding_service.get_mxbai_embedding(query)
            embedding_column = "embedding"

        embedding_list = query_embedding.astype(np.float32).tolist()

        # Query for similar articles using vector similarity
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                WITH ranked_chunks AS (
                    SELECT
                        id,
                        article_id,
                        1 - ({embedding_column} <=> %s::vector) as similarity,
                        ROW_NUMBER() OVER (PARTITION BY article_id ORDER BY {embedding_column} <=> %s::vector) as rank
                    FROM semantic_chunks
                    WHERE {embedding_column} IS NOT NULL
                )
                SELECT id, article_id, similarity
                FROM ranked_chunks
                WHERE rank = 1
                ORDER BY similarity DESC
                LIMIT %s
                """,
                [embedding_list, embedding_list, top_k]
            )
            chunk_results = cursor.fetchall()
            
        # print the chunks
        print(chunk_results)

        # Process results
        article_ids = [result[1] for result in chunk_results]  # Using article_id instead of id
        similarity_scores = {result[1]: result[2] for result in chunk_results}

        # Preserve the order of results using Case/When
        preserved_order = Case(*[
            When(id=pk, then=pos) for pos, pk in enumerate(article_ids)
        ])
        
        # Fetch articles in the correct order
        articles = Article.objects.filter(
            id__in=article_ids
        ).order_by(preserved_order)

        # Return articles with their similarity scores
        return [(article, similarity_scores[article.id]) for article in articles[:3]]

    def get_article_content(self, article_id: int) -> Optional[Dict]:
        """
        Retrieve article content by ID
        
        Args:
            article_id (int): The ID of the article to retrieve
            
        Returns:
            Optional[Dict]: Article content dictionary or None if not found
        """
        try:
            article = Article.objects.get(id=article_id)
            return article.to_small_dict()
        except Article.DoesNotExist:
            return None

    def batch_process_embeddings(self, model: str = "openai", batch_size: int = 32) -> None:
        """
        Process embeddings for articles in batches
        
        Args:
            model (str): The embedding model to use ("openai" or "mxbai")
            batch_size (int): Size of batches to process
        """
        self.embedding_service.batch_embed_articles(model=model, batch_size=batch_size)