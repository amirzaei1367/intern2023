from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import HttpResponse
import os
import torch
import transformers
import re
import random
from tokenizers import AddedToken
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, MetaData, Table, text
import sqlite3

from .utils.chat_utils import format_tokens
from .text_generation import TextGenerator

from .configs import (
    text_generation_config,
    sql_generation_config,
)

from transformers.trainer_utils import set_seed
from .apps import TextToSqlModel, TopicModel

SCHEMA = """Table resume_data = [*, id, first_name, last_name, work_history, skills, certification, clearance, education]
look for the following values in the education column  = [high_school, Bachelor Of Science, M.S., PhD]
look for the following values in the skills colummn = [python, javascript, java, C/C++, SQL, Tableau, Excel]
look for the following values in the clearance colummn= [NACI, MBI, NACLC, ANACI, BI, SSBI, 'nan']
look for the folloeing values in the certification column certification = [AWS, CISM, google_cloud, CISSP, PMP, NCP-MCI, Microsoft_Azure, no_certification, CISA, 
                            VCP-DCV, CCNP, PHR, SPHR, SHRM]
values for work history is an integer between (0,30)
primary_key is the id column.
"""

dialog = [
    [
        {'role': 'system',
         'content': ('Convert the given question into a sql query using the given schema.'
                     'Do not provide an explanation. Use the following instructions.'
                     '1. Begin every query with select *'
                     '2. Do not use any join by clauses as there are no foreign keys in the database.')
        },
    ]
]

# Create your views here.
class TopicExtractionService(APIView):
    def get(self, request):
        resumeText = request.GET.get('resume_text', '')
        model = TopicModel.model
        output = model.transform(resumeText)
        topic = output[0][0]
        topics = model.get_topic(topic)
        topics = {
            'topics': topics
        }
        return Response(topics)

class TextToSqlService(APIView):
    def get(self, request):
        torch.cuda.manual_seed(32)
        torch.manual_seed(32)
        nl_question = request.GET.get('question', '')
        question = SCHEMA + '\n Q: ' + nl_question

        print(question)

        model = TextToSqlModel.model
        tokenizer = TextToSqlModel.tokenizer

        user_msg = {'role': 'user', 'content': question}
        dialog[0].append(user_msg)
        tokens = format_tokens(dialog, tokenizer)

        text_generator = TextGenerator(
            model=model,
            tokenizer=tokenizer,
            config=sql_generation_config
        )

        output = text_generator.generate(tokens)
        pred_sql = output['content']

        dialog[0].append(output)

        dialog[0].pop()
        dialog[0].pop()

        entities = evaluate_sql(pred_sql)

        print(entities)

        sql = {
        'sql': pred_sql,
        'entities': entities, 
        "resume_text": nl_question,
        }

        return Response(sql)

def evaluate_sql(pred_sql):
    engine = create_engine('sqlite:///database_app/resume_data.db')
    metadata = MetaData()
    resume_data = Table('resume_data', metadata, autoload_with=engine)
    entities = []
    try:
        connection = engine.connect()
        sql_stmt = text(pred_sql)
        results= connection.execute(sql_stmt)
        ids = [res[0] for res in results]
        query = resume_data.select().where(resume_data.columns.id.in_(ids))
        output =  connection.execute(query)
        entities = output.fetchall()
        entities = [entity._asdict() for entity in entities]
    except SQLAlchemyError as error:
        print(error)
    finally:
        if connection:
            connection.close()
    return entities
