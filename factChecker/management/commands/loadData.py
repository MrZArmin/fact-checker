import argparse
from llama_index import (
    download_loader,
    Document,
)
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, *args, **options):
        query = f"""
        SELECT e.text
        FROM  articles e
        LEFT JOIN article_keywords ak ON e.id = ak.article_id
        """
        parsed_url = argparse("postgresql://szakdoga:r)<6!x5uJfaA?w@@localhost:5432/archivum")
        documents = self.fetch_documents_from_storage(query=query, parsed_url=parsed_url)

        print(f"Found {len(documents)} documents")
        
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