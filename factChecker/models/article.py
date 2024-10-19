from django.db import models
from .keyword import Keyword

class Article(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateField()
    tags = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    lead = models.TextField()
    text = models.TextField()
    keywords = models.ManyToManyField(Keyword, through='ArticleKeyword', related_name='articles')
    link = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class ArticleKeyword(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE)
    weight = models.IntegerField(default=0)

    class Meta:
        unique_together = ('article', 'keyword')