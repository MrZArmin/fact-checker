from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from factChecker.services.ragServiceOpenAi import RAGServiceOpenAI
from factChecker.models import ChatSession, ChatMessage, ChatMessageArticle, Article
import json

rag_service = RAGServiceOpenAI()

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def start_session(request):
    session = ChatSession.objects.create(user=request.user)
    data = json.loads(request.body)
    prompt = data.get('prompt')

    try:
        title = rag_service.generate_title(prompt)
    except Exception as e:
        title = "Cím nélküli beszélgetés"
    session.title = title
    session.save()

    return Response({'code': 200, 'payload': {'session': session.to_dict()}})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def get_sessions(request):
    sessions = ChatSession.objects.filter(
        user=request.user).order_by('updated_at', 'title')
    sessions_data = [session.to_dict() for session in sessions]
    return Response({
        'code': 200,
        'payload': {'sessions': sessions_data}
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def send_message(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    data = json.loads(request.body)
    message = data.get('message')

    if not message:
        return Response({'code': 400, 'payload': {'error': 'Message is required'}})

    # Save user message to session
    user_message = ChatMessage.objects.create(
        session=session,
        sender='user',
        message=message,
    )

    try:
        # Generate response using RAGServiceOpenAI
        response = rag_service.query(message)
        
        # Save bot response
        ai_message = ChatMessage.objects.create(
            session=session,
            sender='ai',
            message=response['response'],
        )

        # Create article relations
        ChatMessageArticle.objects.bulk_create([
            ChatMessageArticle(
                chat_message=ai_message,
                article=Article.objects.get(id=source['id']),
                similarity_score=source['similarity_score']
            ) for source in response['sources']
        ])

        # Update session
        session.updated_at = ai_message.timestamp
        session.save()

        return Response({
            'code': 200,
            'payload': {
                'user_message': user_message.to_dict(),
                'ai_message': ai_message.to_dict(),
                'session': session.to_dict()
            }
        })

    except Exception as e:
        # Update session even if RAG fails
        session.updated_at = user_message.timestamp
        session.save()
        
        return Response({
            'code': 500,
            'payload': {
                'error': str(e),
                'user_message': user_message.to_dict()
            }
        })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_messages(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.all().order_by('timestamp')
    messages_data = [msg.to_dict() for msg in messages]
    return Response({'code': 200, 'payload': {'messages': messages_data}})

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.delete()
    return Response({'code': 200, 'payload': {'message': 'Session deleted'}})

@api_view(['POST'])
def extract(request):
    data = json.loads(request.body)
    text = data.get('text')
    response = rag_service.improve_user_prompt(text)
    return Response({'code': 200, 'payload': {'response': response}})
