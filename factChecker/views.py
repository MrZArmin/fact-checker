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
  
# MAJD SZERVEZD KI A KÓDOT, HOGY A KÉT FÜGGVÉNY KÜLÖN FÁJLBAN LEGYEN

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# import json
# from .services.ragService import RAGService
# from .services.ragServiceOpenAi import RAGServiceOpenAI

# # Initialize RAG service
# rag_service = RAGServiceOpenAI()

# @csrf_exempt
# @require_http_methods(["POST"])
# def query_rag(request):
#   try:
#     data = json.loads(request.body)
#     query = data.get('query')
    
#     if not query:
#       return JsonResponse({
#         'error': 'Query is required'
#       }, status=400)
    
#     response = rag_service.query(query)
    
#     return JsonResponse({
#       'response': response
#     })
  
#   except Exception as e:
#     return JsonResponse({
#       'error': str(e)
#     }, status=500)