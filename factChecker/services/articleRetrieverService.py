from typing import List, Tuple, Dict, Optional, Literal
from factChecker.services.embeddingService import EmbeddingService
import numpy as np
from django.db import connection
from django.db.models import Case, When
from factChecker.models import Article

TableType = Literal["articles", "chunks", "semantic_chunks", "enhanced_semantic_chunks"]

class ArticleRetrieverService:
    # Define valid tables and their configurations
    VALID_TABLES = {
        "articles": {
            "table_name": "articles",
            "openai_column": "embedding_openai",
            "mxbai_column": "embedding",
            "position_column": None,
            "direct_article_query": True
        },
        "chunks": {
            "table_name": "chunks",
            "openai_column": "embedding_openai",
            "mxbai_column": "embedding",
            "position_column": None,
            "direct_article_query": False
        },
        "semantic_chunks": {
            "table_name": "semantic_chunks",
            "openai_column": "embedding_openai",
            "mxbai_column": "embedding",
            "position_column": None,
            "direct_article_query": False
        },
        "enhanced_semantic_chunks": {
            "table_name": "enhanced_semantic_chunks",
            "openai_column": "embedding_openai",
            "mxbai_column": "embedding",
            "position_column": "position",
            "direct_article_query": False
        }
    }

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def _get_table_config(self, table: str) -> Dict:
        """
        Get configuration for specified table
        
        Args:
            table (str): Name of the table to get configuration for
            
        Returns:
            Dict: Table configuration
            
        Raises:
            ValueError: If table is not valid
        """
        if table not in self.VALID_TABLES:
            raise ValueError(f"Invalid table. Must be one of: {', '.join(self.VALID_TABLES.keys())}")
        return self.VALID_TABLES[table]

    def find_similar_articles(
        self,
        query: str,
        table: TableType = "semantic_chunks",
        model: str = "openai",
        top_k: int = 15
    ) -> List[Tuple[Article, float]]:
        """
        Find similar articles using vector similarity search with specified embedding model and table
        
        Args:
            query (str): The search query text
            table (str): The table to search in
            model (str): The embedding model to use ("openai" or "mxbai")
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[Article, float]]: List of tuples containing (article, similarity_score)
        """
        # Validate model choice
        if model not in ["openai", "mxbai"]:
            raise ValueError("Invalid model. Must be 'openai' or 'mxbai'")

        # Get table configuration
        table_config = self._get_table_config(table)
        
        # Get appropriate embedding column based on model
        embedding_column = (
            table_config["openai_column"] if model == "openai"
            else table_config["mxbai_column"]
        )

        # Get embedding based on model choice
        if model == "openai":
            query_embedding = self.embedding_service.get_openai_embedding(query)
        else:
            query_embedding = self.embedding_service.get_mxbai_embedding(query)
            
        embedding_list = query_embedding.astype(np.float32).tolist()

        # Different query strategy based on whether we're querying articles directly or chunks
        if table_config["direct_article_query"]:
            # Direct article query
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT 
                        id,
                        1 - ({embedding_column} <=> %s::vector) as similarity
                    FROM {table_config['table_name']}
                    WHERE {embedding_column} IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    [embedding_list, top_k]
                )
                results = cursor.fetchall()

            # Process results
            article_ids = [result[0] for result in results]
            similarity_scores = {result[0]: result[1] for result in results}

        else:
            # Chunk-based query
            position_order = ""
            if table_config["position_column"]:
                position_order = f", {table_config['position_column']}"
                
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH ranked_chunks AS (
                        SELECT
                            id,
                            article_id,
                            1 - ({embedding_column} <=> %s::vector) as similarity,
                            ROW_NUMBER() OVER (
                                PARTITION BY article_id 
                                ORDER BY {embedding_column} <=> %s::vector {position_order}
                            ) as rank
                        FROM {table_config['table_name']}
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

            # Process results
            article_ids = [result[1] for result in chunk_results]
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
        return [(article, similarity_scores[article.id]) for article in articles[:4]]

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

    def batch_process_embeddings(
        self,
        table: TableType = "semantic_chunks",
        model: str = "openai",
        batch_size: int = 32
    ) -> None:
        """
        Process embeddings for articles in batches
        
        Args:
            table (str): The table to process embeddings for
            model (str): The embedding model to use ("openai" or "mxbai")
            batch_size (int): Size of batches to process
        """
        # Validate table and get configuration
        table_config = self._get_table_config(table)
        
        self.embedding_service.batch_embed_articles(
            table=table_config["table_name"],
            model=model,
            batch_size=batch_size
        )