from django.db import models
from .keyword import Keyword
from pgvector.django import VectorField

class Article(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateField()
    tags = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    lead = models.TextField()
    text = models.TextField()
    keywords = models.ManyToManyField(Keyword, through='ArticleKeyword', related_name='articles')
    link = models.URLField(max_length=400)
    created_at = models.DateTimeField(auto_now_add=True)
    embedding = VectorField(dimensions=1024, null=True)  # 1024-dimensional vector
    embedding_openai = VectorField(dimensions=1536, null=True)

    def __str__(self):
        return f"\nCím: {self.title}\nBevezető: {self.lead}\nTartalom: {self.text}\n"

    def to_small_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'date': self.date,
            'lead': self.lead,
            'text': self.text,
            'link': self.link
        }

    class Meta:
        db_table = 'articles'
        managed = True

class ArticleKeyword(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE)
    weight = models.IntegerField(default=0)

    class Meta:
        db_table = 'article_keywords'
        unique_together = ('article', 'keyword')
