from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from factChecker.services.articleGraphRetrieverService import ArticleGraphRetrieverService

retriver = ArticleGraphRetrieverService()

class Command(BaseCommand):
    def handle(self, *args, **options):
        retriver.findSimilarNodes(user_query="Ki az az Orbán Viktor?")
