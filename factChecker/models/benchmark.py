from django.db import models
from .article import Article

class Benchmark(models.Model):
    id = models.AutoField(primary_key=True)
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.BooleanField()

    class Meta:
        db_table = 'benchmark'
        managed = True
