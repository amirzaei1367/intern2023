from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import os
import torch
import transformers 
import re
import random

from sqlalchemy.exc import SQLAlchemyError
from transformers.trainer_utils import set_seed
from .apps import TextToSqlModel
from .utils.chat_utils import format_tokens
from .utils.sql_utils import evaluate_sql
from .text_generation import TextGenerator
from .configs import text_generation_config

SCHEMA = """Table resume_data = [*, id, first_name, last_name, work_history, skills, certifiaction, clearance, education]"""

dialog = [
    [
        {'role': 'system',
         'content': ('Convert the given question into a sql query using the schema given. Do not provide an explanation.')
        }
    ]
]

class TextToSqlService(APIView):
    def get(self, request):
        torch.cuda.manual_seed(32)
        torch.manual_seed(32)
        nl_question = request.GET.get('question', '')
        question = SCHEMA = '\n Q: ' + nl_question
        model = TextToSqlModel.model
        tokenizer = TextToSqlModel.tokenizer
        user_msg = {'role': 'user', 'content': question}

        text_generator = TextGenerator(
            model=model,
            tokenizer=tokenizer,
            config=text_generation_config,
        )

        output = text_generator.generate(tokens)
        pred_sql = output['content']
        dialog[0].append(output)
        dialog[0].pop()
        dialog[0].pop()

        entities = evaluate_sql(pred_sql)

        sql = {
            'sql': pred_sql,
            'entities': entities,
            'resume_text': nl_question,
        }
        return sql
