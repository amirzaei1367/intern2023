# Text-To-SQL 

The following repsository contains the code that was used in the text-to-sql portion of the resume database project. Text-to-sql is the problem of converting natural 
language questions into appropriate sql queries using large language models, which can then be used to query the corresponding database. The goal of this project is to 
provide the means of quering a given database without the need to learn the SQL expression language. The model that was used in this project was Llama 2. In particular
the 7B-chat variant was used. For more information about these models go here:
[https://about.fb.com/news/2023/07/llama-2/](url)

views.py contains the the endpoints for both the TopicExtractionService as well as the TextToSqlService. 

Visit the official Llama2 repository for more information and examples on how to finetune Llama2.

[https://github.com/facebookresearch/llama/tree/main](url)


[https://github.com/facebookresearch/llama-recipes/](url)

The folder resumedb is the django project folder that contains all of the files needed to run the the text to sql and topic modeling service. The weights and the data used for training and evaluation are not included. To download the weights of the model go here [https://huggingface.co/meta-llama/Llama-2-7b-hf](url)

To download the spider dataset used for evaluation go here [https://yale-lily.github.io/spider](url)


To download the ALPACA dataset usded for finetuning the Llama2 base go here [https://github.com/tatsu-lab/stanford_alpaca](url)

