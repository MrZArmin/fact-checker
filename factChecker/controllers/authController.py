from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated


@api_view(['POST'])
def login(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(username=username, password=password)

    if user is not None:
        refresh = RefreshToken.for_user(user)

        return Response({
            'code': 200,
            'payload': {
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                },
                'access_token': str(refresh.access_token),
                'token_type': 'Bearer',
                'expires_in': 3600,  # 1 hour
            }
        }, status=status.HTTP_200_OK)
    else:
        return Response({
            'code': 401,
            'error': 'Invalid credentials'
        }, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
def logout(request):
    return Response({'message': 'Logged out successfully'}, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def me(request):
    user = request.user
    return Response({
        'code': 200,
        'payload': {
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            }
        }
    }, status=status.HTTP_200_OK)
