from django.db import models
from .iptc_code import IPTCCode
from .geographic_code import GeographicCode
from .sab_code import SABCode
from .person import Person
from .institution import Institution

class Article(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateField()
    tags = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    lead = models.TextField()
    text = models.TextField()
    iptc_codes = models.ManyToManyField(IPTCCode, through='ArticleIPTCCode', related_name='articles')
    geographic_codes = models.ManyToManyField(GeographicCode, through="ArticleGeographicCode", related_name='articles')
    sab_codes = models.ManyToManyField(SABCode, through="ArticleSABCode", related_name='articles')
    persons = models.ManyToManyField(Person, through='ArticlePerson', related_name='articles')
    institutions = models.ManyToManyField(Institution, through='ArticleInstitution', related_name='articles')
    link = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class ArticleIPTCCode(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    iptc_code = models.ForeignKey(IPTCCode, on_delete=models.CASCADE)
    weight = models.IntegerField()

    class Meta:
        unique_together = ('article', 'iptc_code')

class ArticlePerson(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    weight = models.IntegerField()

    class Meta:
        unique_together = ('article', 'person')

class ArticleInstitution(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE)
    weight = models.IntegerField()

    class Meta:
        unique_together = ('article', 'institution')

class ArticleGeographicCode(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    geographic_code = models.ForeignKey(GeographicCode, on_delete=models.CASCADE)
    weight = models.IntegerField()

    class Meta:
        unique_together = ('article', 'geographic_code')

class ArticleSABCode(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    sab_code = models.ForeignKey(SABCode, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('article', 'sab_code')