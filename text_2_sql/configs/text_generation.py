from dataclasses import dataclass

@dataclass 
class text_generation:
    max_new_tokens: int = 128
    do_sample: bool = True
    use_cache: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: int = 1
