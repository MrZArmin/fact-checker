from typing import List, Tuple
import numpy as np
from django.db import connection
from psycopg2.extensions import register_adapter, AsIs
import numpy
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from factChecker.models import ChatMessageArticle, Article


def adapt_numpy_array(numpy_array):
    return AsIs(repr(numpy_array.tolist()))


register_adapter(numpy.ndarray, adapt_numpy_array)


class RAGServiceOpenAI:
    def __init__(self):
        load_dotenv(override=True)
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        try:
            prompt_path = self.prompts_dir / filename
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Could not load prompt {
                                    filename}: {str(e)}")

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
        try:
            article = Article.objects.get(id=article_id)
            
            return article.to_small_dict()
        except Article.DoesNotExist:
            return None

    def generate_response(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
        model: str = "gpt-4o",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using GPT model based on query and context.

        Args:
            query: User's question
            context: Retrieved context for answering the question
            temperature: Controls randomness in response (0.0-1.0)
            model: GPT model to use
            max_tokens: Maximum tokens in response (optional)

        Returns:
            str: Generated response in Hungarian

        Raises:
            Exception: If response generation fails
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._load_prompt("response_prompt.txt")
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {query}"
                }
            ]

            completion_params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }

            if max_tokens:
                completion_params["max_tokens"] = max_tokens

            chat_completion = self.client.chat.completions.create(
                **completion_params)

            return chat_completion.choices[0].message.content

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            raise RuntimeError(error_msg)

    def generate_title(
        self,
        text: str,
        temperature: float = 0.7,
        model: str = "gpt-3.5-turbo",
    ) -> str:
        """
        Generate a title for a conversation.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._load_prompt("title_prompt.txt")
                },
                {
                    "role": "user",
                    "content": f"Generate a title for the following text: {text}"
                }
            ]

            completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=50  # Title length limit
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"Error generating title: {str(e)}"
            raise RuntimeError(error_msg)
        
    def extract_valuable_info(self, text: str) -> str:
        """
        Extract valuable information from the text.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._load_prompt("extract_info_prompt.txt")
                },
                {
                    "role": "user",
                    "content": f"Extract valuable information from the following text: {text}"
                }
            ]

            completion = self.client.chat.completions.create(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=150  # Limit for extracted info
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"Error extracting information: {str(e)}"
            raise RuntimeError(error_msg)

    def query(self, user_query: str) -> dict:
        """Main RAG pipeline using OpenAI embeddings"""
        try:
            valuable_info_in_query = self.extract_valuable_info(user_query)
            context = ""
            articles = []
            if valuable_info_in_query.strip() != "Null":          
                # 1. Find similar articles using OpenAI embeddings
                similar_articles = self.find_similar_articles(user_query)

                if not similar_articles:
                    return {
                        'response': "Nem találtunk releváns cikkeket az adatbázisban.",
                        'sources': []
                    }

                # 2. Retrieve article contents
                for article_id, similarity_score in similar_articles:
                    article_content = self.get_article_content(article_id)
                    if article_content:
                        context += f"\nCím: {article_content['title']}\nBevezető: {article_content['lead']}\nTartalom: {article_content['text']}\n"
                        
                        articles.append({
                            'id': article_id,
                            'similarity_score': round(similarity_score, 4)
                        })

                if not context.strip():
                    return {
                        'response': "Találtunk cikkeket, de nem sikerült lekérni a tartalmukat.",
                        'sources': []
                    }
                
                response = self.generate_response(valuable_info_in_query, context)
                

            # 3. Generate response using GPT-4
            response = self.generate_response(user_query, context)
            print(articles)

            return {
                'response': response,
                'sources': articles,
                'valuable_info': valuable_info_in_query
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
