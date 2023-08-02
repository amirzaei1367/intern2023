from django.apps import AppConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

class DatabaseAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'database_app'

class FileExtraction(AppConfig):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model_8bit = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", load_in_8bit=True)