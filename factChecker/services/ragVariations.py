from typing import List, Tuple, Optional, Dict
import numpy as np
from django.db import connection
from openai import OpenAI
import cohere
from dotenv import load_dotenv
import os
from pathlib import Path
from factChecker.models import Article
from factChecker.services.articleRetrieverService import ArticleRetrieverService
from factChecker.services.ragServiceOpenAi import RAGServiceOpenAI

class RagVariations:
    
    THRESHOLD = 0.4
    
    def __init__(self):
        load_dotenv(override=True)
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.article_retriever = ArticleRetrieverService()
        self.cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
        self.rag_service = RAGServiceOpenAI()
        
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
        systemPromptTxt: str = "response_prompt_benchmark.txt",
        temperature: float = 0.4,
        model: str = "gpt-4o-mini",
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response using GPT model."""
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

    def query_v1_baseline(self, user_query: str, threshold=THRESHOLD) -> dict:
        """Baseline approach: Direct article embeddings without reranking"""
        similar_articles = self.article_retriever.find_similar_articles(
            query=user_query,
            table="articles",  # Use article-level embeddings
            model="openai",
            top_k=4
        )

        context = ""
        articles = []
        
        for article, similarity_score in similar_articles:
            if similarity_score < threshold:
                continue
            context += f"\nCím: {article.title}\nBevezető: {article.lead}\nTartalom: {article.text}\n"
            articles.append({
                'id': article.id,
                'similarity_score': round(float(similarity_score), 4)
            })

        if not context.strip():
            return {'response': "No relevant articles found.", 'sources': []}

        response = self.generate_response(user_query, context)
        return {'response': response, 'sources': articles}

    def query_v2_semantic_chunks(self, user_query: str, threshold=THRESHOLD) -> dict:
        """Semantic chunks approach without reranking"""
        similar_articles = self.article_retriever.find_similar_articles(
            query=user_query,
            table="semantic_chunks",
            model="openai",
            top_k=4
        )

        context = ""
        articles = []
        
        for article, similarity_score in similar_articles:
            if similarity_score < threshold:
                continue
            context += f"\nCím: {article.title}\nBevezető: {article.lead}\nTartalom: {article.text}\n"
            articles.append({
                'id': article.id,
                'similarity_score': round(float(similarity_score), 4)
            })

        if not context.strip():
            return {'response': "No relevant articles found.", 'sources': []}

        response = self.generate_response(user_query, context)
        return {'response': response, 'sources': articles}

    def query_v3_semantic_chunks_with_imporved_prompt(self, user_query: str, threshold=THRESHOLD) -> dict:
        """Enhanced semantic chunks with position information"""
        improved_prompt = self.rag_service.improve_user_prompt(user_query)
        similar_articles = self.article_retriever.find_similar_articles(
            query=improved_prompt,
            table="semantic_chunks",
            model="openai",
            top_k=4
        )

        context = ""
        articles = []
        
        for article, similarity_score in similar_articles:
            if similarity_score < threshold:
                continue
            context += f"\nCím: {article.title}\nBevezető: {article.lead}\nTartalom: {article.text}\n"
            articles.append({
                'id': article.id,
                'similarity_score': round(float(similarity_score), 4)
            })

        if not context.strip():
            return {'response': "No relevant articles found.", 'sources': []}

        response = self.generate_response(user_query, context)
        return {'response': response, 'sources': articles}

    def query_v4_with_reranking(self, user_query: str, threshold=THRESHOLD) -> dict:
        try:
            improved_prompt = self.rag_service.improve_user_prompt(user_query)
            context = ""
            articles = []

            # 1. Find similar articles using the article retriever instance
            similar_articles = self.article_retriever.find_similar_articles(
                query=improved_prompt,
                model="openai"
            )

            articles_str = [str(article) for article in similar_articles]

            reranked_data = self.rag_service.rerank_documents(user_query, articles_str)
            reranked_docs = []

            for index, result in enumerate(reranked_data.results):
                reranked_docs.append(similar_articles[result.index])

            top_similarity = round(float(similar_articles[0][1]), 4)
            if top_similarity < threshold:
                return {
                    'result': null,
                }


            # 2. Retrieve article contents
            for article, similarity_score in reranked_docs:
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
    def query_v5_mxbai(self, user_query: str, threshold=THRESHOLD) -> dict:
        """Query using mxbai embeddings"""
        # Get articles using article-level embeddings
        enhanced_user_query = self.rag_service.improve_user_prompt(user_query)
        article_results = self.article_retriever.find_similar_articles(
            query=enhanced_user_query,
            table="enhanced_semantic_chunks",
            model="mxbai",
            top_k=4
        )

        # Prepare for reranking
        articles_str = [
            f"Title: {article.title}\nLead: {article.lead}\nContent: {article.text}" 
            for article, _ in article_results
        ]

        # Rerank the combined results
        reranked = self.cohere_client.rerank(
            model="rerank-multilingual-v3.0",
            query=user_query,
            documents=articles_str,
            top_n=4
        )

        context = ""
        articles = []
        
        for result in reranked.results:
            article, _ = combined_articles[result.index]
            if result.relevance_score < threshold:
                continue
            context += f"\nCím: {article.title}\nBevezető: {article.lead}\nTartalom: {article.text}\n"
            articles.append({
                'id': article.id,
                'similarity_score': round(float(result.relevance_score), 4)
            })

        response = self.generate_response(user_query, context)
        return {'response': response, 'sources': articles}