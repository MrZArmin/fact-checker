from django.core.management.base import BaseCommand
from dotenv import load_dotenv
import re
from factChecker.models import Article, SemanticChunk
from openai import OpenAI
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

class Command(BaseCommand):
    help = "Semantically chunks articles for processing with semantic awareness"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_dotenv()

    def handle(self, *args, **options):
        # Get total count for progress bar
        articles = Article.objects.all()

        # Create progress bar for articles
        with tqdm(articles, desc="Processing articles", unit="article") as pbar:
            text_splitter = SemanticChunker(
                OpenAIEmbeddings(), 
                breakpoint_threshold_amount=80,
                buffer_size=0
            )

            for article in pbar:
                # Skip articles that are already processed
                if SemanticChunk.objects.filter(article=article).exists():
                    continue

                # Set description to show current article title
                truncated_title = article.title[:30] + "..." if len(article.title) > 30 else article.title
                pbar.set_description(f"Processing '{truncated_title}'")

                # Process the article
                try:
                    # Ensure lead and text exist before concatenating
                    text = f"{getattr(article, 'lead', '')} {article.text}".strip()
                    article_docs = text_splitter.create_documents([self.clean_text(text)])
                    
                    # Get keywords with weight >= 100
                    keywords = article.keywords.filter(articlekeyword__weight__gte=100)

                    # Skip if too many chunks or article is flagged as bullshit
                    if len(article_docs) > 20 or self.is_article_bullshit(article):
                        self.stdout.write(
                            self.style.WARNING(f"Skipping article '{truncated_title}' - too many chunks or filtered content")
                        )
                        continue

                    # Create semantic chunks
                    for doc in article_docs:
                        # Combine title, content and keywords
                        chunk_text = f"{article.title} {doc.page_content}"
                        if keywords:
                            chunk_text += " " + " ".join(str(keyword) for keyword in keywords)
                        
                        SemanticChunk.objects.create(
                            article=article,
                            text=chunk_text.strip()
                        )

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"Error processing article '{truncated_title}': {str(e)}")
                    )

    def clean_text(self, text: str) -> str:
        """
        Cleans the text by removing unwanted characters and normalizing spacing.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove any length --- sequences
        text = re.sub(r'-{3,}', '', text)
        
        # Remove +++ sequences
        text = text.replace("+++", "")
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def is_article_bullshit(self, article: Article) -> bool:
        """
        Determines if an article should be filtered out based on certain keywords.
        
        Args:
            article (Article): Article object to check
            
        Returns:
            bool: True if article should be filtered out, False otherwise
        """
        if not article.title:
            return True
            
        filter_keywords = ["Tippmix", "TOTÓ", "eredmények"]
        return any(keyword in article.title for keyword in filter_keywords)