"""
URL configuration for factChecker project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from factChecker import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

from factChecker.controllers.authController import login
from factChecker.controllers.authController import me
from factChecker.controllers.authController import logout
from factChecker.controllers.chatController import start_session
from factChecker.controllers.chatController import send_message
from factChecker.controllers.chatController import get_messages
from factChecker.controllers.chatController import get_sessions
from factChecker.controllers.chatController import delete_session
from factChecker.controllers.chatController import extract

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('unscraped-links/', views.get_unscraped_links, name='unscraped_links'),
    #path('api/rag/query/', views.query_rag, name='query_rag'),
    
    path('login/', login, name='login'),
    path('logout/', logout, name='logout'),
    path('me/', me, name='me'),
    
    path('chat/sessions/', get_sessions, name='get_sessions'),
    path('chat/start/', start_session, name='start_session'),
    path('chat/<uuid:session_id>/send/', send_message, name='send_message'),
    path('chat/<uuid:session_id>/messages/', get_messages, name='get_messages'),
    path('chat/<uuid:session_id>/', delete_session, name='delete_session'),
    
    path('extract/', extract, name='extract'),
]
