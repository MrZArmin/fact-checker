from django.core.management.base import BaseCommand
from dotenv import load_dotenv
import re
from factChecker.models import Article
from factChecker.models import SemanticChunk
from openai import OpenAI
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# https://python.langchain.com/docs/how_to/semantic-chunker/

class Command(BaseCommand):
    help = "Semanically chunks articles for processing with semantic awareness"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_dotenv()

    def handle(self, *args, **options):
        articles = Article.objects.all()[:5]
        
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_amount=80)
        
        for article in articles:
            text = article.lead + " " + article.text
            article_docs = text_splitter.create_documents([self.clean_text(text)])
            
            keywords = article.keywords.filter(articlekeyword__weight__gte=100)
            
            for doc in article_docs:
                text = article.title + ' ' + doc.page_content
                
                for keyword in keywords:
                    text += f" {keyword}"
                    
                SemanticChunk.objects.create(article=article, text=text)
            
                    
    def clean_text(self, text: str) -> str:
        """
        Cleans the text by removing unwanted characters and normalizing spacing.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove +++ sequences
        text = text.replace("+++", "")
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
