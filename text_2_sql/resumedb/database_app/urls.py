from django.contrib import admin
from django.urls import include, path
from database_app.views import TextToSqlService, TopicExtractionService, TopicMatchingService
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
        path('TextToSqlService', TextToSqlService.as_view(), name='text_to_sql'),
        path('TopicExtractionService', TopicExtractionService.as_view(), name='topic_extraction'),
]

