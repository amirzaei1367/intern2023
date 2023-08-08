from dataclasses import dataclass
from pprint import pprint
from typing import List, TypedDict, Literal
import fire
import json
import os

SQL_PROMPT = """
Given below is a sql schema and a question. Based on the schema proved convert the question in to a sql query.
"""
SQL_PROMPT_2 = """
Below is question that should be converted to a sql query based on the schema provided below. Do not provide an explanation.
Sql schema:
table_name = resume_data
columns = [id(int, primary_key), first_name, last_name, work_history(int), certificates, security_clearance, education]
Response:
"""

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

DIALOG = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class ChatLog:
    def __init__(self, sys_msg):
        assert(sys_msg['role'] == 'system'), ('The first role must be a system role')
        self.dialog: DIALOG = [sys_msg]
        self._role = "system" 

    def add_message(self, message: Message):
        if message['role'] == 'user':
            assert(self._role == 'system' or self._role == 'assistant')
            self.dialog.append(message)
            self._role = message['role']
        elif message['role'] == 'assistant':
            assert(self._role == 'user')
            self.dialog.append(message)
            self._role = message['role'] 
        else:
            print(f'error: cannot add a {message["role"]} role when current role is {self._role}')

    def add_dataset_entry(self, dataset, idx):
        input, output = dataset[idx]
        self.add_message(input)
        self.add_message(output)

    def add_user_entry(self, dataset, idx): 
        input, output = dataset[idx]
        self.add_message(input)

    def format_and_tokenize(self, tokenizer):
        dialog = [
            {
                "role": self.dialog[1]["role"],
                "content": B_SYS
                + self.dialog[0]['content']
                + E_SYS
                + self.dialog[1]['content']
            }
            ] + self.dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user', and 'assistant' roles,"
            "starting with user and alternating (u/a/u/a/u.....)"
        )
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} \
                            {(answer['content']).strip()}"
                ) 
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        assert(
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        return [dialog_tokens]
            
    def __len__(self):
        return len(self.dialog)
    
    def __getitem__(self, idx):
        return self.dialog[idx]
    
    def __iter__(self):
        return iter(self.dialog)
    
    def __str__(self):
        return f'{self.dialog}'
