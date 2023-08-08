import os
from dataclasses import dataclass
from django.apps import AppConfig
from typing import Optional
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from .utils import load_model, load_tokenizer

from bertopic import BERTopic

class TextToSqlModel(AppConfig):
    model_id: str = '/home/ubuntu/projects/resumedb/database_app/models_hf/7B-chat'
    name = 'database_app'
    model = load_model(model_id, quantization=True)
    tokenizer = load_tokenizer(model_id)
    
class TopicModel(AppConfig):
    model_id: str = '/home/ubuntu/projects/resumedb/database_app/models/topic_model'
    embed_id: str = 'distilbert-base-cased'
    model = BERTopic.load(model_id, embed_id)
