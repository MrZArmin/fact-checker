from django.db import models

class Keyword(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return f"{self.name}"

    class Meta:
        db_table = 'keywords'
        managed = True