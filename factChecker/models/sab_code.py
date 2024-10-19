from django.db import models

class SABCode(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return f"{self.name}"