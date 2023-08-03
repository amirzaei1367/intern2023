import torch

class TextGeneration:
    def __init__(self, model, tokenizer, config):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config

    def generate(self, tokens):
        self.model.eval()
        with torch.no_grad():
            for idx, chat in enumerate(tokens):
                tokens = torch.tensor(chat)
                tokens = tokens.unsqueeze(0)
                tokens = tokens.to("cuda:0")
                outputs = self.model.generate(
                    tokens,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_k=self.cofig.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    length_penalty=self.config.length_penalty,
                )
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split('[/INST]')
            output_text = output_text[-1].strip()
            output_text = output_text.split("A: ")
            output_text = output_text[-1]
            return {"role": "assistant", "content": output_text}
