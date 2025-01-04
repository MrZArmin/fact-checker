from django.db import models
from django.contrib.auth.models import User
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
    session = models.ForeignKey(ChatSession, related_name='messages', on_delete=models.CASCADE)
    sender = models.CharField(max_length=255)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message from {self.sender} at {self.timestamp}"