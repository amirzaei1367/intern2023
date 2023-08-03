### Configs 

This folder contains the configurations for the Llama2 model that was used in this project. The following parameters were used:

1) max_new_tokens: The maximumn numbers of tokens to generate, ignoring the number of tokens in the prompt
2) do_sample: whether or not to use sampling. If False use greedy decoding otherwise
3) use_cache: Whether or not the model should use the past last key/values attentions to speed up decoding
4) top_p: if set to float <1, onlyt the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
5) top_k: The number of highest probability vocabulary tokens to keep for top-k filtering
6) temperature: The value used to modulate the next token probabilites
7) repetition_penalty: the parameter for repetition penalty 1.0 means no penalty.
8) length_penalty: Exponentioal penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.

The default text generation strategy used is multinomial sampling.

For more information about the these paramaters and other potential parameters not mentioned here please go to the offical huggingface documentation or click this link:
[https://huggingface.co/docs/transformers/main_classes/text_generation](url) 
