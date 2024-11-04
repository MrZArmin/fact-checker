import re
from urllib.parse import urlparse
import numpy as np
from llama_index.core import Document
from llama_index.readers.database import DatabaseReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from django.core.management.base import BaseCommand
from factChecker.models import Article
from dotenv import load_dotenv
import os

EMBED_MODEL = HuggingFaceEmbedding(
  model_name="WhereIsAI/UAE-Large-V1",
  embed_batch_size=10
)

class Command(BaseCommand):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def handle(self, *args, **options):
    load_dotenv()
    query = """
    SELECT e.id, e.text, e.lead
    FROM articles e
    WHERE e.text IS NOT NULL
    AND e.embedding IS NULL
    """
    parsed_url = urlparse(os.getenv("DB_URL"))
    documents = self.fetch_documents_from_storage(query=query, parsed_url=parsed_url)
    self.save_embeddings(documents)

  def fetch_documents_from_storage(self, query: str, parsed_url) -> list[Document]:
    db_config = {
      "scheme": "postgresql",
      "host": parsed_url.hostname,
      "port": parsed_url.port,
      "user": parsed_url.username,
      "password": parsed_url.password,
      "dbname": parsed_url.path[1:],
    }
    
    reader = DatabaseReader(
      scheme=db_config["scheme"],
      host=db_config["host"],
      port=db_config["port"],
      user=db_config["user"],
      password=db_config["password"],
      dbname=db_config["dbname"],
    )
    documents = reader.load_data(query=query)
    print(f"Loaded {len(documents)} documents")
    return documents

  def save_embeddings(self, documents: list[Document]) -> None:
    for i, document in enumerate(documents):
      try:
        match = re.match(r"id: (\d{1,6})", document.text)
        if not match:
          print(f"Skipping document {i}: No valid ID found")
          continue
          
        article_id = int(match.group(1))
          
        # Get the article
        article = Article.objects.get(id=article_id)
        
        # Get the embedding
        embedding = EMBED_MODEL.get_text_embedding(document.text)
        
        # Convert embedding to correct format for pgvector
        # Ensure it's a 1D numpy array
        if isinstance(embedding, list):
          embedding = np.array(embedding)
        if len(embedding.shape) > 1:
          embedding = embedding.flatten()
        
        # Save the embedding
        article.embedding = embedding
        article.save()
        print(f"Processed {i + 1} documents")
        
      except Exception as e:
        print(f"Error processing document {i}: {str(e)}")
        continue