from urllib.parse import urlparse
from llama_index.embeddings import HuggingFaceEmbedding
from django.core.management.base import BaseCommand
from dotenv import load_dotenv
import os
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index import (
    set_global_tokenizer,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    Document,
)
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoTokenizer

class Command(BaseCommand):
  help = "Sets up PostgreSQL vector store for article embeddings"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    load_dotenv()


  def handle(self, *args, **options):
    vector_store = self.setup_pg_vector_store()
    service_context = self.get_service_context()
    
    
    print("Setting up PostgreSQL vector store")
    
  def setup_pg_vector_store(self):
    try:
      # Get database URL from environment
      db_url = os.getenv("DB_URL")
      if not db_url:
        raise ValueError("DB_URL environment variable is not set")

      # Parse database URL
      parsed_url = urlparse(db_url)
      if not all([parsed_url.hostname, parsed_url.username, parsed_url.password]):
        raise ValueError("Invalid DATABASE_URL format")

      # Get embedding dimension
      embed_dim = os.getenv("EMBED_DIM")
      if not embed_dim:
        raise ValueError("EMBED_DIM environment variable is not set")
      try:
        embed_dim = int(embed_dim)
      except ValueError:
        raise ValueError("EMBED_DIM must be an integer")

      # Initialize vector store
      vector_store = PGVectorStore.from_params(
        database=parsed_url.path[1:],
        host=parsed_url.hostname,
        password=parsed_url.password,
        port=parsed_url.port or 5432,
        user=parsed_url.username,
        table_name="article_embeddings",
        embed_dim=embed_dim,
      )
      
      self.stdout.write(
        self.style.SUCCESS("Successfully set up PostgreSQL vector store")
      )
      
      return vector_store

    except ValueError as e:
      self.stdout.write(self.style.ERROR(f"Configuration error: {str(e)}"))
    except Exception as e:
      self.stdout.write(
        self.style.ERROR(f"Failed to set up PostgreSQL vector store: {str(e)}")
      )
      
  def get_service_context(self) -> ServiceContext:
    llm = LlamaCPP(
      model_path=os.getenv("LLM_MODEL_PATH"),
      context_window=int(os.getenv("LLM_CONTEXT_WINDOW")),
      max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
      model_kwargs={"n_gpu_layers": 1},
      verbose=True,
    )
    
    embed_model = HuggingFaceEmbedding(
      model_name="WhereIsAI/UAE-Large-V1",
      embed_batch_size=10
    )
    
    set_global_tokenizer(
      AutoTokenizer.from_pretrained(f"mistralai/Mixtral-8x7B-Instruct-v0.1").encode
    )  # must match your LLM model
    
    service_context = ServiceContext.from_defaults(
      llm=llm,
      embed_model=embed_model,
      system_prompt="You are a bot that answers questions shortly in Hungarian.",
    )
    
    return service_context