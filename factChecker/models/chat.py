from django.db import models
from django.contrib.auth.models import User
from .article import Article
import uuid
from django.forms.models import model_to_dict


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, default="Untitled Session")
    status = models.CharField(max_length=50, default="active")
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.id} for {self.user.username}"

    def to_dict(self):
        data = model_to_dict(self, fields=['user', 'title', 'status'])
        data['id'] = self.id
        data['updated_at'] = self.updated_at
        data['created_at'] = self.created_at
        return data


class ChatMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ChatSession, related_name='messages', on_delete=models.CASCADE)
    sender = models.CharField(max_length=255)
    message = models.TextField()
    valuable_info = models.TextField(null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def to_dict(self):
        data = {
            'id': self.id,
            'sender': self.sender,
            'message': self.message,
            'timestamp': self.timestamp,
            'valuable_info': self.valuable_info,
            'articles': []
        }

        # Only include articles if this is an AI message
        if self.sender == 'ai':
            article_relations = self.article_relations.select_related(
                'article').all()
            data['articles'] = [{
                'id': relation.article.id,
                'title': relation.article.title,
                'lead': relation.article.lead,
                'text': relation.article.text,
                'link': relation.article.link,
                'similarity_score': relation.similarity_score
            } for relation in article_relations]

        return data

    def __str__(self):
        return f"Message from {self.sender} at {self.timestamp}"


class ChatMessageArticle(models.Model):
    chat_message = models.ForeignKey(
        ChatMessage, related_name='article_relations', on_delete=models.CASCADE)
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    similarity_score = models.FloatField()

    class Meta:
        unique_together = ('chat_message', 'article')
