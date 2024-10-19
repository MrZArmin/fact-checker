from django.db import models

class IPTCCode(models.Model):
    name = models.CharField(max_length=255, unique=True)
    weight = models.IntegerField(help_text="Percentage value associated with this IPTC code")

    def __str__(self):
        return f"{self.name} ({self.weight}%)"

    class Meta:
        ordering = ['-weight', 'name']