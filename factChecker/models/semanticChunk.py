from django.db import models
from pgvector.django import VectorField
from .article import Article

class SemanticChunk(models.Model):
    id = models.AutoField(primary_key=True)
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    text = models.TextField()
    embedding = VectorField(dimensions=1024, null=True)  # 1024-dimensional vector
    embedding_openai = VectorField(dimensions=1536, null=True)

    def __str__(self):
        return self.text

    class Meta:
        db_table = 'semantic_chunks'
        managed = True