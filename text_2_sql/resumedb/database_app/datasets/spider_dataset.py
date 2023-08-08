import fire
import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List, TypedDict, Literal

#from utils import format_tokens

DEFAULT_PROMPT = """
Below is question that should be converted to a sql query based on the schema provided below. Do not provide an explanation.
"""

Role = Literal["user", "assistant", "system"]

class Message(TypedDict):
    role: Role
    content: str

DIALOG = List[Message]

INITIAL_MESSAGE = Message(
    role = 'system',
    content = DEFAULT_PROMPT
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class SpiderDataset(Dataset):
    def __init__(self, dataset_config, tokenizer=None, partition='train'):
        if partition == 'train':
            self.data = json.load(open(dataset_config.train_path))
        else:
            self.data = json.load(open(dataset_config.test_path))
        with open(dataset_config.prompt_path, 'r') as f:
            self.prompt_dict = json.load(f)

        self.tokenzer = tokenizer
        self.init_msg = INITIAL_MESSAGE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        db_id = data['db_id']
        question = data['question']
        query = data['query']
        schema = self.prompt_dict[db_id]

        user_msg = {
            'role': 'user',
            'content':schema + '\n\n' +'Q: ' + question
        }
        label = {
            'role': 'assistant',
            'content': '\n\n' + 'A: ' + query 
        }

        return [user_msg, label]

    
    def get_db_id(self, idx):
        return self.data[idx]['db_id']
