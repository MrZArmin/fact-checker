from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from factChecker.services.ragServiceOpenAi import RAGServiceOpenAI
from factChecker.models import ChatSession, ChatMessage
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
    
    return Response({'code': 200, 'payload': {'session': {'id': session.id, 'title': session.title}}})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def get_sessions(request):
    sessions = ChatSession.objects.filter(user=request.user)
    sessions_data = [
        {
            'id': session.id,
            'title': session.title,
            'status': session.status,
            'created_at': session.created_at
        }
        for session in sessions
    ]
    return Response(
        {'code': 200, 'payload': {'sessions': sessions_data}}
    )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def send_message(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    data = json.loads(request.body)
    message = data.get('message')

    if not message:
        return Response({'code': 400, 'payload': {'error': 'Message is required'}})

    # Save user message
    ChatMessage.objects.create(
        session=session,
        sender='user',
        message=message
    )

    # Generate response using RAGServiceOpenAI
    try:
        response = rag_service.query(message)
    except Exception as e:
        return Response({'code': 500, 'payload': {'error': str(e)}})

    # Save bot response
    ChatMessage.objects.create(
        session=session,
        sender='ai',
        message=response['response']
    )

    return Response(
        {
            'code': 200,
            'payload': {
                'response': response['response'],
                'sources': response['sources']
            }
        }
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_messages(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.all().order_by('timestamp')
    messages_data = [
        {'sender': msg.sender, 'message': msg.message,
            'timestamp': msg.timestamp} for msg in messages]
    return Response({'code': 200, 'payload': {'messages': messages_data}})


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.delete()
    return Response({'code': 200, 'payload': {'message': 'Session deleted'}})
