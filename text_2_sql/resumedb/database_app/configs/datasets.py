from dataclasses import dataclass

@dataclass 
class spider_dataset:
    dataset: str = "spider_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_path: str = '/home/ubuntu/projects/resumedb/database_app/datasets/spider/train_spider.json'
    test_path: str = '/home/ubuntu/projects/resumedb/database_app/datasets/spider/dev.json'
    prompt_path: str = '/home/ubuntu/projects/resumedb/database_app/datasets/spider/prompt_dict.json'
