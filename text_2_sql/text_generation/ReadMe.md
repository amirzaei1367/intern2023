### Text Generation 

This folder contains the module generate_text generate text. For information about the specific parameters used check the configs folder. The text_generation module 
contains the class TextGenerator which has the following attributes model, token ,config and contains one method generate(). The model uses the huggingface api generate 
method to produce the output tokens. These tokens are then decoded using the tokenizer to produce the output text. 
For more information about generate and the specific details about the model class check the hugging face documentation or click on 
this link:
[https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate](url)
