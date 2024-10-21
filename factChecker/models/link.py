from django.db import models

class Link(models.Model):
    url = models.URLField(max_length=400, unique=True)
    scraped = models.BooleanField(default=False)

    def __str__(self):
        return self.url

    class Meta:
        db_table = 'links'
        managed = True