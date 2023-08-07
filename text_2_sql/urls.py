from django.contrib importadmin 
from django.urls import path
from database_app.views import TextToSqlService, TopicExtractionService, TopicMatchingService

urlpatterns = [
    path('TextToSqlService', TextToSqlService.as_view(), name='text_to_sql'),
    path('TopicExtractionService, TopicExtractionService.as_view(), name='topic_extraction'),
]
