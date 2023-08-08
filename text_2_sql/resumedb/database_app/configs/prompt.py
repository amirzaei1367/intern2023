from dataclasses import dataclass
from typing import Optional

@dataclass 
class prompt_config:
    prompt_path: str = '/home/ubuntu/projects/resumedb/database_app/datasets/spider/prompt_dict.json'
    prompt_dir: Optional[str] = None

