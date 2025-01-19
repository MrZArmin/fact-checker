from typing import List, Tuple, Optional
import numpy as np
from django.db import connection
from psycopg2.extensions import register_adapter, AsIs
import numpy
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
from pathlib import Path
from factChecker.models import ChatMessageArticle, Article
from django.db.models import Case, When
from factChecker.services.articleRetrieverService import ArticleRetrieverService
import huspacy

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
        self.article_retriever = ArticleRetrieverService()

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        try:
            prompt_path = self.prompts_dir / filename
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Could not load prompt {filename}: {str(e)}")

    def generate_response(
        self,
        query: str,
        context: str,
        systemPromptTxt: str = "response_prompt.txt",
        temperature: float = 0.4,
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response using GPT model based on query and context."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._load_prompt(systemPromptTxt)
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

            chat_completion = self.client.chat.completions.create(**completion_params)
            return chat_completion.choices[0].message.content

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            raise RuntimeError(error_msg)

    def query(self, user_query: str, threshold=0.6) -> dict:
        """Main RAG pipeline using OpenAI embeddings"""
        try:
            improved_prompt = self.improve_user_prompt(user_query)

            #prompt_ents = self._extract_entites_from_prompt(user_query)
            #print(prompt_ents)

            #print("Improved prompt:" + improved_prompt)
            context = ""
            articles = []

            # 1. Find similar articles using the article retriever instance
            similar_articles = self.article_retriever.find_similar_articles(
                query=improved_prompt,
                model="openai"
            )
            top_similarity = round(float(similar_articles[0][1]), 4)
            print(top_similarity)
            if top_similarity < threshold:
                response = self.generate_response(user_query, context, "basic_answer_prompt.txt")
                return {
                    'response': response,
                    'sources': []
                }
            # 2. Retrieve article contents
            for article, similarity_score in similar_articles:
                if article:
                    context += f"\nCím: {article.title}\nBevezető: {article.lead}\nTartalom: {article.text}\n"

                    articles.append({
                        'id': article.id,
                        'similarity_score': round(float(similarity_score), 4)
                    })

            if not context.strip():
                return {
                    'response': "Találtunk cikkeket, de nem sikerült lekérni a tartalmukat.",
                    'sources': []
                }

            response = self.generate_response(user_query, context)

            return {
                'response': response,
                'sources': articles
            }

        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            raise

    def improve_user_prompt(self, text: str) -> str:
        """Improve user query prompt by adding context and structure."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._load_prompt("extract_info_prompt.txt")
                },
                {
                    "role": "user",
                    "content": f"""Original user query: {text}
                    Please transform this query into a detailed search prompt following the above guidelines."""
                }
            ]

            completion = self.client.chat.completions.create(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.4,
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"Error extracting information: {str(e)}"
            raise RuntimeError(error_msg)

    def _extract_entites_from_prompt(self, query):
        nlp = huspacy.load()
        doc = nlp(query)
        entities = {}

        for ent in doc.ents:
            if ent.label not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities

    def generate_title(
        self,
        text: str,
        temperature: float = 0.7,
        model: str = "gpt-3.5-turbo",
    ) -> str:
        """Generate a title for a conversation."""
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
                max_tokens=50
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"Error generating title: {str(e)}"
            raise RuntimeError(error_msg)
