from typing import List, Tuple
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from django.db import connection
import torch
from psycopg2.extensions import register_adapter, AsIs
import numpy

# Register adapter for numpy array to PostgreSQL array

def adapt_numpy_array(numpy_array):
    return AsIs(repr(numpy_array.tolist()))


register_adapter(numpy.ndarray, adapt_numpy_array)


class RAGService:
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('WhereIsAI/UAE-Large-V1')

        # Initialize the generator
        self.generator = pipeline(
            'text-generation',
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',     # Buta mint a seggem
            #model='mistralai/Mixtral-8x7B-Instruct-v0.1',   # Ez meg megöli Szaniszlót
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text"""
        embedding = self.embedding_model.encode(text)
        return embedding.astype(np.float32)  # Ensure float32 type

    def find_similar_articles(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        """
        Find similar articles using vector similarity search
        Returns: List of tuples (article_id, similarity_score)
        """
        query_embedding = self.get_embedding(query)

        # Convert embedding to list and ensure float32
        embedding_list = query_embedding.astype(np.float32).tolist()

        # Using raw SQL with pgvector's cosine similarity
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, 1 - (embedding <=> %s::vector) as similarity
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
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
        """Generate response using the LLM"""
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.
        Always provide a clear and concise answer. Avoid unnecessary information. Answer in a complete sentence. Answer in Hungarian.        

        Context: {context}

        Question: {query}

        Answer: """
        
        print(prompt)
        with open('prompt.txt', 'w') as f:
            f.write(prompt)

        response = self.generator(
            prompt,
            max_new_tokens=1024,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True
        )

        return response[0]['generated_text'].split("Answer: ")[-1].strip()

    def query(self, user_query: str) -> dict:
        """Main RAG pipeline"""
        try:
            # 1. Hasonló cikkek keresése
            similar_articles = self.find_similar_articles(user_query)
            
            if not similar_articles:
                return {
                    'response': "No relevant articles found in the database.",
                    'sources': []
                }
            
            # 2. Cikkek tartalmának lekérése
            articles = []
            context = ""
            
            for article_id, similarity_score in similar_articles:
                article_content = self.get_article_content(article_id)
                if article_content:
                    context += f"\nTitle: {article_content['title']}\nLead: {article_content['lead']}\nContent: {article_content['text']}\n"
                    articles.append({
                        'id': article_id,
                        'title': article_content['title'],
                        'lead': article_content['lead'],
                        'link': article_content['link'],
                        'similarity_score': round(similarity_score, 4)
                    })
            
            if not context.strip():
                return {
                    'response': "Found articles but couldn't retrieve their content.",
                    'sources': []
                }
            
            # 3. Válasz generálása az LLM segítségével
            response = self.generate_response(user_query, context)
            
            return {
                'response': response,
                'sources': articles
            }
            
        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            raise
