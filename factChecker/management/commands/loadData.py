from urllib.parse import urlparse
from llama_index import (
    download_loader,
    Document,
)
from llama_index.embeddings import HuggingFaceEmbedding
from django.core.management.base import BaseCommand
from factChecker.models import Article

EMBED_MODEL = HuggingFaceEmbedding(
    model_name="WhereIsAI/UAE-Large-V1", embed_batch_size=10  # open-source embedding model
)

class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, *args, **options):
        query = f"""
            SELECT e.text
            FROM articles e
            LEFT JOIN article_keywords ak ON e.id = ak.article_id
        """
        # Use urlparse instead of argparse
        parsed_url = urlparse("postgresql://szakdoga:Kutyakutya1@localhost:5432/archivum")
        documents = self.fetch_documents_from_storage(query=query, parsed_url=parsed_url)
        print(f"Found {len(documents)} documents")
        self.save_embeddings(documents)

    def fetch_documents_from_storage(self, query: str, parsed_url) -> list[Document]:
        # Prep documents - fetch from DB
        DatabaseReader = download_loader("DatabaseReader")
        reader = DatabaseReader(
            scheme="postgresql",  # Database Scheme
            host=parsed_url.hostname,  # Database Host
            port=parsed_url.port,  # Database Port
            user=parsed_url.username,  # Database User
            password=parsed_url.password,  # Database Password
            dbname=parsed_url.path[1:],  # Database Name
        )
        return reader.load_data(query=query)
    
    def save_embeddings(self, documents: list[Document]) -> None:
        for document in documents:

            # Get the article
            article = Article.objects.get(id=document.id)

            # Get the embedding
            embedding = EMBED_MODEL.embed(document.text)

            # Save the embedding
            article.embedding = embedding
            article.save()