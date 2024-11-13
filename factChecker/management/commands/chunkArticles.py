from django.core.management.base import BaseCommand
from dotenv import load_dotenv
import re
from typing import List
from factChecker.models import Article
from factChecker.models import Chunk


class Command(BaseCommand):
    help = "Chunks articles for processing with semantic awareness"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_dotenv()

    def handle(self, *args, **options):
        articles = Article.objects.all()

        for article in articles:
            self.stdout.write(f"\nProcessing article ID: {article.id}")
            
            # Create text from title, lead and text
            article_text = f"{article.lead} {article.text}"

            # Chunk the article with semantic awareness
            chunks = self.chunk_text_semantic(article_text)

            # Analyze and report chunking results
            stats = self.analyze_chunks(chunks)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Chunked into {stats['total_chunks']} parts\n"
                    f"Average length: {stats['avg_length']:.2f} characters\n"
                ))

            # Append keywords with 200 or 100 weight to each chunk
            keywords = article.keywords.filter(articlekeyword__weight__gte=100)
            title = article.title
            
            new_chunks = []
            
            for chunk in chunks:
                chunk = f"{title} {chunk} "
                for keyword in keywords:
                    chunk += f"{keyword} "
                new_chunks.append(chunk)
                    
                          
               
            # Save chunks to the database
            for chunk in new_chunks:
                Chunk.objects.create(article=article, text=chunk)
                
        
                
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

    def chunk_text_semantic(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Chunks text while preserving semantic meaning and maintaining 15% overlap.

        Args:
            text (str): The input text to be chunked
            chunk_size (int): Target size for each chunk

        Returns:
            List[str]: List of semantically meaningful chunks
        """
        # Clean the text
        text = self.clean_text(text)

        # Calculate overlap size (15% of chunk_size)
        overlap_size = int(chunk_size * 0.15)

        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        # Common sentence-ending punctuation
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        # Common paragraph and section markers
        section_breaks = ['\n\n', '\n###', '\n##',
                          '\n#']

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to find a section break first
            found_break = False
            for break_pattern in section_breaks:
                pos = text.find(break_pattern, end -
                                overlap_size, end + overlap_size)
                if pos != -1:
                    chunks.append(text[start:pos].strip())
                    start = pos + 1
                    found_break = True
                    break

            if found_break:
                continue

            # Try to find a sentence ending
            best_end = None
            for ending in sentence_endings:
                pos = text.find(ending, end - overlap_size, end + overlap_size)
                if pos != -1:
                    best_end = pos + len(ending) - 1
                    break

            if best_end:
                chunks.append(text[start:best_end].strip())
                start = best_end - overlap_size  # Maintain 15% overlap
                continue

            # If no good breaking point found, fall back to last word boundary
            last_space = text.rfind(' ', end - overlap_size, end)
            if last_space != -1:
                chunks.append(text[start:last_space].strip())
                start = last_space - overlap_size
            else:
                # Worst case: break at chunk_size
                chunks.append(text[start:end].strip())
                start = end - overlap_size

        # Remove empty or very small chunks
        return [chunk for chunk in chunks if chunk and len(chunk) > 50]

    def analyze_chunks(self, chunks: List[str]) -> dict:
        """
        Analyzes the quality of chunks.
        """
        lengths = [len(chunk) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(chunks) if chunks else 0,
            "min_length": min(lengths) if chunks else 0,
            "max_length": max(lengths) if chunks else 0
        }
