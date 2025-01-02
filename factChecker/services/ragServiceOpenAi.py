from typing import List, Tuple
import numpy as np
from django.db import connection
from psycopg2.extensions import register_adapter, AsIs
import numpy
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import logging


def adapt_numpy_array(numpy_array):
    return AsIs(repr(numpy_array.tolist()))


register_adapter(numpy.ndarray, adapt_numpy_array)

# Configure logging
# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('rag_service.log')  # File handler
    ]
)
logger = logging.getLogger(__name__)


class RAGServiceOpenAI:
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    def find_similar_articles(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find similar articles using vector similarity search with OpenAI embeddings
        Returns: List of tuples (article_id, similarity_score)
        """
        query_embedding = self.get_embedding(query)
        embedding_list = query_embedding.astype(np.float32).tolist()

        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, 1 - (embedding_openai <=> %s::vector) as similarity
                FROM articles
                WHERE embedding_openai IS NOT NULL
                ORDER BY embedding_openai <=> %s::vector
                LIMIT %s
            """, [embedding_list, embedding_list, top_k])

            results = cursor.fetchall()

        return [(row[0], float(row[1])) for row in results]

    def get_article_content(self, article_id: int) -> dict:
        """Retrieve article content by ID"""
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT text, title, lead, link
                FROM articles
                WHERE id = %s
            """, [article_id])

            result = cursor.fetchone()

            if result:
                text, title, lead, link = result
                return {
                    'title': title,
                    'lead': lead,
                    'text': text,
                    'link': link
                }
            return {}

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using GPT-4"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Based on the provided context, answer the question. If the answer cannot be found in the context, say so. Always provide a clear and concise answer. Avoid unnecessary information. Answer in a complete sentence. Answer in Hungarian."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\nQuestion: {query}"
                    }
                ],
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise

    def generate_title(self, text: str) -> str:
        """Generate title using GPT"""
        logger.info(f"Generating title for text: {text}")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a concise title for the given text. The title should be clear and informative. Avoid unnecessary information. Answer in Hungarian. Use a maximum of 50 characters. Remember to include the main topic of the text. Answer only with the title."
                    },
                    {
                        "role": "user",
                        "content": f"Generate a title for the following text: {text}"
                    }
                ],
                temperature=0.7,
                max_tokens=50
            )
            title = response.choices[0].message.content.strip()
            logger.info(f"Generated title: {title}")
            return title
        except Exception as e:
            logger.error(f"Error generating title: {str(e)}", exc_info=True)
            raise

    def query(self, user_query: str) -> dict:
        """Main RAG pipeline using OpenAI embeddings"""
        try:
            # 1. Find similar articles using OpenAI embeddings
            similar_articles = self.find_similar_articles(user_query)

            if not similar_articles:
                return {
                    'response': "Nem találtunk releváns cikkeket az adatbázisban.",
                    'sources': []
                }

            # 2. Retrieve article contents
            articles = []
            context = ""

            for article_id, similarity_score in similar_articles:
                article_content = self.get_article_content(article_id)
                if article_content:
                    context += f"\nCím: {article_content['title']}\nBevezető: {
                        article_content['lead']}\nTartalom: {article_content['text']}\n"
                    articles.append({
                        'id': article_id,
                        'title': article_content['title'],
                        'lead': article_content['lead'],
                        'link': article_content['link'],
                        'similarity_score': round(similarity_score, 4)
                    })

            if not context.strip():
                return {
                    'response': "Találtunk cikkeket, de nem sikerült lekérni a tartalmukat.",
                    'sources': []
                }

            # 3. Generate response using GPT-4
            response = self.generate_response(user_query, context)

            return {
                'response': response,
                'sources': articles
            }

        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            raise

    def batch_embed_articles(self, batch_size: int = 50) -> None:
        """Batch process articles without OpenAI embeddings"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, lead, text 
                    FROM articles 
                    WHERE embedding_openai IS NULL 
                    AND text IS NOT NULL
                """)
                articles = cursor.fetchall()

            for i in tqdm(range(0, len(articles), batch_size), desc="Processing batches"):
                batch = articles[i:i + batch_size]
                texts = [f"Title: {art[1]}\nLead: {
                    art[2]}\nContent: {art[3]}" for art in batch]

                # Get embeddings for batch
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=texts,
                        encoding_format="float"
                    )
                    embeddings = [np.array(data.embedding, dtype=np.float32)
                                  for data in response.data]

                    # Update database
                    with connection.cursor() as cursor:
                        for j, embedding in enumerate(embeddings):
                            cursor.execute(
                                "UPDATE articles SET embedding_openai = %s WHERE id = %s",
                                [embedding.tolist(), batch[j][0]]
                            )

                except Exception as e:
                    print(f"Error processing batch starting at index {
                          i}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error in batch embedding: {str(e)}")
            raise
