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
from .datasets import SpiderDataset
from sqlalchemy import create_engine, MetaData, Table, text
import sqlite3

from .utils.chat_utils import format_tokens
from .text_generation import TextGenerator

from .configs import (
    text_generation_config,
    spider_config,
    sql_generation_config,
)

from collections import deque, namedtuple

from transformers.trainer_utils import set_seed
from .apps import TextToSqlModel, TopicModel

from .create_resume_db import (
    ENGINE,
    RESUMES,
    METADATA,
)

SCHEMA = """Table resume_data = [*, id, first_name, last_name, work_history, skills, certification, clearance, education]
look for the following values in the education column  = [high_school, Bachelor Of Science, M.S., PhD]
look for the following values in the skills colummn = [python, javascript, java, C/C++, SQL, Tableau, Excel]
look for the following values in the clearance colummn= [NACI, MBI, NACLC, ANACI, BI, SSBI, 'nan']
look for the folloeing values in the certification column certification = [AWS, CISM, google_cloud, CISSP, PMP, NCP-MCI, Microsoft_Azure, no_certification, CISA, 
                            VCP-DCV, CCNP, PHR, SPHR, SHRM]
values for work history is an integer between (0,30)
primary_key is the id column.
"""

dataset = SpiderDataset(spider_config)

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


for idx in range(5):
    dialog[0] += dataset[idx]

examples = [
    [{'role': 'user',
     'content': SCHEMA + '\nQ: ' + 'List the id of the people with atleast 3 years of work experience.'
    },
    {'role': 'assistant',
     'content': 'SELECT * FROM resume_data WHERE work_history >= 3;'
    }],
    [{'role': 'user',
     'content': SCHEMA + '\nQ:' + 'Give me the people with atleast 3 years of work experience.' 
    },
    {'role': 'assistant',
     'content': 'SELECT * FROM resume_data WHERE work_history >= 3;'
    }],
    [{'role': 'user',
     'content': SCHEMA + '\nQ:' + 'List the people with a masters degree'
    },
    {'role': 'assistant',
     'content': "SELECT * FROM resume_data WHERE education = 'M.S.';"
    }]
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
    
class TopicMatchingService(APIView):
    def get(self, request):
        resumeText = request.GET.get('resume_text', '')
        jobDescription = request.GET.get('job_description', '')
        # do something with query here
        match = {
        'score': 0.5,
        }
        return Response(match)

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



def process_input(question):

    def process_column_names(column_name):
        column_name =re.sub('\W+', '_', column_name)
        return 'resume_data.' + column_name

    column_names = ['id', 'first_name', 'last_name', 'clearance', 'certification', 'skill', 'education', 'work_history']
    
    input = question + ' ' + '|' + ' ' + 'resume_data : '
    for column_name in column_names:
        column_name = re.sub('W+', '_', column_name)
        input += ', ' + process_column_names(column_name)

    input = re.sub('resume_data : ,', 'resume_data : ', input)

    return input



def inference(question):
    set_seed(128)

    input_question = process_input(question)
    print(input_question)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    # Initialize tokenizer
    tokenizer = TextToSqlModel.tokenizer
    tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    #initialize model
    model = TextToSqlModel.model
    if torch.cuda.is_available():
        model = model.cuda()

    tokenized_input = tokenizer(
            input_question,
            return_tensors = 'pt',
            max_length = 512,
            padding = "max_length",
            truncation = True,
    )
    encoder_input_ids = tokenized_input["input_ids"]
    encoder_input_attention_mask = tokenized_input["attention_mask"]
    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()

    with torch.no_grad():
        model_output = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = 8,
                num_return_sequences = 8
        )
        predicted_sql = tokenizer.decode(model_output[0], skip_special_tokens = True)

    return re.split(' \| ', predicted_sql)[1]

def get_predicted_result(query):
    engine = create_engine('sqlite:///database_app/resume.db')
    try:
        connection = sqlite3.connect('database_app/resume.db')
        cursor = connection.cursor()
        select_query = query
        cursor.execute(select_query)
        entities = cursor.fetchall()
        cursor.close()
    except sqlite3.Error as error:
        print(error)
    finally:
        if connection:
            connection.close()
    return entities

def get_predicted_result_2(query):
    engine = create_engine('sqlite:///database_app/resume.db', echo = True)
    with engine.connect() as connection:
        statement = text(query)
        result = connection.execute(statement)
    return result

def process_ids(ids):
    return ', '.join(['\''+ str(id) + '\'' for id in ids])

def get_rows(ids):
    engine = create_engine('sqlite:///database_app/resume.db', echo = False)
    conn = engine.connect()
    metadata = MetaData()
    resume_data = Table('resume_data', metadata, autoload_with = engine) 
    query = resume_data.select().where(resume_data.columns.id.in_(ids))
    output =  conn.execute(query)
    results = output.fetchall()
    conn.close()
    return results
