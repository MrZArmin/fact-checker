from django.core.management.base import BaseCommand
from transformers import AutoTokenizer, AutoModel
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
import spacy
from typing import List, Dict, Optional, Union, Any
import re
from factChecker.models import Article, SemanticChunkEnhanced
import torch
from django.db import transaction
import os
from openai import OpenAI
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    text: str
    semantic_score: float
    embedding: List[float]
    position: int  # Track chunk position in original document

class EmbeddingProvider:
    """Base class for embedding providers"""
    def __init__(self):
        self.dimensions: int = None
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
        
    def validate_dimensions(self, embeddings: List[List[float]]) -> bool:
        """Validate that embeddings match expected dimensions"""
        if not embeddings:
            return False
        return all(len(emb) == self.dimensions for emb in embeddings)

class OpenAIProvider(EmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "text-embedding-3-small"
        self.dimensions = 1024
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            embeddings = [data.embedding for data in response.data]
            
            if not self.validate_dimensions(embeddings):
                raise ValueError(f"Invalid embedding dimensions. Expected {self.dimensions}")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {str(e)}")
            raise

class MXBaiProvider(EmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dimensions = 1024
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            embeddings_list = embeddings.tolist()
            
            if not self.validate_dimensions(embeddings_list):
                raise ValueError(f"Invalid embedding dimensions. Expected {self.dimensions}")
                
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error getting MXBai embeddings: {str(e)}")
            raise

class CustomSemanticChunker:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.breakpoint_threshold = 0.7
        
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0

    def create_documents(self, texts: List[str], chunk_size: int = 512) -> List[str]:
        chunks = []
        position = 0
        
        for text in texts:
            # Split into initial segments
            segments = self._split_into_segments(text, chunk_size)
            
            try:
                # Get embeddings for all segments
                embeddings = self.embedding_provider.get_embeddings(segments)
                
                # Merge segments based on semantic similarity
                current_chunk = segments[0]
                current_embedding = embeddings[0]
                
                for i in range(1, len(segments)):
                    if self.similarity(current_embedding, embeddings[i]) >= self.breakpoint_threshold:
                        current_chunk += " " + segments[i]
                    else:
                        chunks.append((current_chunk, position))
                        position += 1
                        current_chunk = segments[i]
                        current_embedding = embeddings[i]
                
                chunks.append((current_chunk, position))
                position += 1
                
            except Exception as e:
                logger.error(f"Error creating documents: {str(e)}")
                continue
        
        return chunks

    def _split_into_segments(self, text: str, chunk_size: int) -> List[str]:
        """Split text into initial segments at sentence boundaries"""
        segments = []
        sentences = text.split(". ")
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= chunk_size:
                current_segment += sentence + ". "
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + ". "
        
        if current_segment:
            segments.append(current_segment.strip())
            
        return segments

class HungarianSemanticChunker:
    def __init__(self, embedding_type: str = "openai"):
        self.embedding_type = embedding_type
        
        # Initialize embedding provider based on type
        if embedding_type == "openai":
            self.embedding_provider = OpenAIProvider()
        else:
            self.embedding_provider = MXBaiProvider()
            
        # Initialize semantic chunker
        self.semantic_chunker = CustomSemanticChunker(self.embedding_provider)
        
        # Initialize Hungarian NLP
        try:
            self.nlp = spacy.load("hu_core_news_lg")
        except:
            logger.warning("Hungarian model not found, falling back to multilingual model")
            self.nlp = spacy.load("xx_ent_wiki_sm")
            
        # Common noise patterns in Hungarian content
        self.noise_patterns = [
            r'katt.*ide',
            r'fotó:.*',
            r'(?i)hirdetés',
            r'©.*rights.*reserved',
            r'minden\s*jog\s*fenntartva',
            r'(\d+)\s*perc\s*olvasási\s*idő',
        ]
        
        # Content to filter out
        self.low_value_markers = {
            'Tippmix', 'TOTÓ', 'eredmények', 'nyerőszámok',
            'horoszkóp', 'időjárás', 'lottószámok'
        }

    def clean_text(self, text: str) -> str:
        """Clean Hungarian text."""
        if not isinstance(text, str):
            return ""
            
        # Remove noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean formatting
        text = re.sub(r'-{3,}', '', text)
        text = re.sub(r'\+{3,}', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def is_valuable_content(self, title: str, text: str) -> bool:
        """Determine if content is worth processing."""
        if not title or not text:
            return False
            
        if any(marker.lower() in title.lower() for marker in self.low_value_markers):
            return False
            
        if len(''.join(text.split())) < 200:
            return False
            
        doc = self.nlp(text[:1000])
        if len(doc.ents) < 2:
            return False
            
        return True

    def create_semantic_chunks(self, article: Article) -> List[ChunkMetadata]:
        """Create semantic chunks from article content with embeddings."""
        title = article.title
        content = self.clean_text(f"{article.lead or ''}\n{article.text or ''}")
        
        if not self.is_valuable_content(title, content):
            return []
        
        try:
            # Generate semantic chunks
            chunks_with_positions = self.semantic_chunker.create_documents([content])
            
            # Get keywords
            keywords = list(article.keywords.filter(
                articlekeyword__weight__gte=100
            ).values_list('name', flat=True))
            
            # Enhance and score chunks
            enhanced_chunks = []
            
            # Create enhanced texts first
            enhanced_texts = []
            positions = []
            
            for chunk, position in chunks_with_positions:
                if len(chunk.strip()) < 100:
                    continue
                    
                enhanced_text = (
                    f"Cím: {title}\n\n"
                    f"{chunk}"
                )
                
                if keywords:
                    enhanced_text += f"\nKulcsszavak: {' '.join(keywords)}"
                
                enhanced_texts.append(enhanced_text)
                positions.append(position)
            
            # Get embeddings for all enhanced texts in one batch
            embeddings = self.embedding_provider.get_embeddings(enhanced_texts)
            
            # Create chunk metadata with embeddings
            for enhanced_text, embedding, position in zip(enhanced_texts, embeddings, positions):
                semantic_score = sum(x * x for x in embedding) ** 0.5
                
                enhanced_chunks.append(ChunkMetadata(
                    text=enhanced_text,
                    semantic_score=semantic_score,
                    embedding=embedding,
                    position=position
                ))
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks for article {article.id}: {str(e)}")
            return []

class Command(BaseCommand):
    help = "Creates semantic chunks using either OpenAI or MXBai embeddings"

    def add_arguments(self, parser):
        parser.add_argument(
            '--embedding-type',
            type=str,
            default='openai',
            choices=['openai', 'mxbai'],
            help='Embedding model to use'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for processing'
        )
        parser.add_argument(
            '--cleanup',
            action='store_true',
            help='Clean up orphaned chunks before processing'
        )

    def cleanup_orphaned_chunks(self):
        """Clean up chunks whose articles no longer exist"""
        try:
            with transaction.atomic():
                orphaned = SemanticChunkEnhanced.objects.filter(
                    article_id__in=Article.objects.values_list('id', flat=True)
                )
                count = orphaned.count()
                orphaned.delete()
                return count
        except Exception as e:
            logger.error(f"Error cleaning up orphaned chunks: {str(e)}")
            return 0

    def handle(self, *args, **options):
        start_time = time.time()
        
        self.stdout.write(f"Starting semantic chunk processing using {options['embedding_type']} embeddings")
        # Cleanup if requested
        if options['cleanup']:
            cleaned = self.cleanup_orphaned_chunks()
            self.stdout.write(f"Cleaned up {cleaned} orphaned chunks")
        
        # Validate environment
        if options['embedding_type'] == 'openai' and not os.getenv('OPENAI_API_KEY'):
            self.stderr.write(self.style.ERROR("OPENAI_API_KEY not set"))
            return

        # Initialize chunker with specified embedding type
        chunker = HungarianSemanticChunker(embedding_type=options['embedding_type'])
        
        # Get unprocessed articles
        articles = Article.objects.exclude(
            id__in=SemanticChunkEnhanced.objects.values_list('article_id', flat=True)
        ).filter(text__isnull=False)
        
        if not articles:
            self.stdout.write(self.style.SUCCESS("No new articles to process"))
            return
            
        total_articles = articles.count()
        self.stdout.write(f"Processing {total_articles} articles using {options['embedding_type']} embeddings")
        
        # Process articles
        chunks_created = 0
        errors = 0
        
        with tqdm(total=total_articles, desc="Creating semantic chunks") as pbar:
            for article in articles:
                try:
                    chunks = chunker.create_semantic_chunks(article)
                    
                    if chunks:
                        with transaction.atomic():
                            created_chunks = SemanticChunkEnhanced.objects.bulk_create([
                                SemanticChunkEnhanced(
                                    article=article,
                                    text=chunk.text,
                                    embedding=chunk.embedding if chunker.embedding_type == 'mxbai' else None,
                                    embedding_openai=chunk.embedding if chunker.embedding_type == 'openai' else None,
                                    position=chunk.position
                                )
                                for chunk in chunks
                                if chunk.semantic_score > 0.5
                            ])
                            chunks_created += len(created_chunks)
                    
                except Exception as e:
                    errors += 1
                    self.stderr.write(
                        self.style.ERROR(f"Error processing article {article.id}: {str(e)}")
                    )
                    
                finally:
                    pbar.update(1)
        
        # Report statistics
        duration = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"\nProcessing completed in {duration:.2f} seconds\n"
            f"Articles processed: {total_articles}\n"
            f"Chunks created: {chunks_created}\n"
            f"Errors encountered: {errors}"
        ))