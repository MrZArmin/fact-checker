from django.shortcuts import render
from django.http import JsonResponse
from .models import Link

def index(request):
    return render(request, 'index.html')


def get_unscraped_links(request):
    # Fetch the first 10 unscraped links
    unscraped_links = Link.objects.filter(scraped=False).order_by('id')[:10]
    
    # Prepare the data for JSON response
    links_data = [
        {
            'id': link.id,
            'url': link.url,
            'scraped': link.scraped
        }
        for link in unscraped_links
    ]
    
    # Return the data as JSON
    return JsonResponse({'links': links_data})