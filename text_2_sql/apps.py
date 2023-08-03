from django.apps import AppConfig
from typing import Optional
from .utils import load_model, load_tokenizer

class TextToSqlModel(AppConfig):
    model_id: str = 'model_path' # path to the model weights
    name: str = 'database_app'
    model = load_model(model_id, quantization=True)
    tokenizer = load_tokenizer(model_id)
