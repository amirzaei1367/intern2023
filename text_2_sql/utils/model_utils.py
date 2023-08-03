from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

def load_model(model_path: str, quantization: bool):
    return LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=quantization,
        return_dict=True,
        device_map="auto",
        low_cpu_memory_usage=True
    )
    
def load_tokenizer(model_path: str):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    return tokenizer

def load_peft_model(model, peft_model: str):
    return PeftModel.from_pretrained(model, peft_model)
