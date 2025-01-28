from django.core.management.base import BaseCommand
from factChecker.models import Article

class Command(BaseCommand):
    help = 'Clean articles'

    def handle(self, *args, **kwargs):      
        
        articles = Article.objects.all()
        starting_count = len(articles)
        self.stdout.write(self.style.SUCCESS(f'Found {len(articles)} articles'))
        
        self.delete_articles_by_title()
        self.stdout.write(self.style.SUCCESS('Successfully cleaned articles by title'))
        
        self.delete_articles_by_tags()
        self.stdout.write(self.style.SUCCESS('Successfully cleaned articles by tags'))
        
        self.delete_empty()
        self.stdout.write(self.style.SUCCESS('Successfully cleaned person titles'))
        
        current_count = Article.objects.all().count()
        self.stdout.write(self.style.SUCCESS(f'Successfully cleaned {starting_count - current_count} articles'))
        
    def delete_articles_by_title(self):
        banned_titles = ['A Magyar Nemzeti Bank hivatalos devizaárfolyamai']
        
        # If the article's tittle contains(dont have to be the exact same) any of the banned titles, delete it
        for title in banned_titles:
            deleteable_articles = Article.objects.filter(title__icontains=title)
            deleteable_articles.delete()
            
    def delete_articles_by_tags(self):
        banned_tags = ['tippmix', 'Szerencsejáték', 'tőzsde', 'Totó', 'BÉT', 'skandináv', 'ötös lottó', 'hatos lottó' ]
        
        # If the article's tags contain any of the banned tags, delete it
        for tag in banned_tags:
            deleteable_articles = Article.objects.filter(tags__icontains=tag)
            deleteable_articles.delete()
            
    def delete_empty(self):
        empty_articles = Article.objects.filter(text__exact='', lead__exact='RANG, FUNKCIÓ: ---')
        empty_articles = empty_articles | Article.objects.filter(text__exact='RANG, FUNKCIÓ: ---', lead__exact='RANG, FUNKCIÓ: ---')
        empty_articles = empty_articles | Article.objects.filter(text__exact='RANG, FUNKCIÓ: ---', lead__exact='---')
        empty_articles = empty_articles | Article.objects.filter(text__exact='', lead__exact='---')
        
        empty_articles.delete()