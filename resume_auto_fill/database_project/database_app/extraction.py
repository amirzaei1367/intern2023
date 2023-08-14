from rest_framework.response import Response
import torch
import numpy as np
import re
import glob

import csv
import random
import time
import os
import boto3
import spacy
import sys
sys.path.append("/home/ubuntu/database_project")

s3 = boto3.client('s3', aws_access_key_id='AKIAZ257POCST3TRJZSS', aws_secret_access_key='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', region_name='us-east-2')
nlp = spacy.load("en_core_web_sm")

#Reference apps.py
from database_app.apps import FileExtraction

# Seed initialization to ensure reproducibility
seed = 1989
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def append_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        field_names = ['first_name', 'last_name', 'phone_number', 'email_address', 'physical_address', 'security_clearance', 'certifications', 'skills', 'education', 'work_history']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        # Check if the file is empty and write header if needed
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(data)

def upload_to_s3(file_path):
        # Define the S3 bucket and object name
        bucket_path = ''
        bucket_name = 'resume-database-intern2023' 
        object_name_in_s3 = bucket_path + os.path.basename(file_path)

        # Upload the file to S3
        s3.upload_file(file_path, bucket_name, object_name_in_s3)

def BatchFieldsExtractionService():
    # Expand the directory path
    RESUMES_DIR = '~/database_project/resumes'
    resumes_path = os.path.expanduser(RESUMES_DIR)

    # List all files in the directory
    all_files = glob.glob(os.path.join(resumes_path, '*'))
    # Iterate over each file and extract details
    extracted_data = []
    for file_path in all_files:
        with open(file_path, 'r') as file:
            uploaded_file = file.read()
        extracted_fields = post(uploaded_file)
        print("extracted fields",extracted_fields)
    return extracted_data

def post(data):
        start_time = time.time()  # Start the timer
        # do something with query here
        parsed_resume = extract_fields_from_resume(data)
        print(parsed_resume)
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print("Elapsed Time:", elapsed_time, "seconds")  # Print the elapsed time

        #Append output to csv file
        append_to_csv('resume_fields.csv', parsed_resume)

        # Upload CSV to S3
        upload_to_s3('resume_fields.csv')

        return parsed_resume

def clean_text(text):
        # Filter out special characters that are not alphanumeric, punctuation, '+' or '#'
        filtered_text = re.sub(r"[^a-zA-Z0-9 \t\n\.\+#\/\\\-:\)\(@]", "", text)
        # Replace multiple spaces with single space
        filtered_text = re.sub(r'\s{2,}', ' ', filtered_text)
        return filtered_text

def get_name_with_ner(text):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Take the first name of the recognized person's name
                return [ent.text.split()[0],ent.text.split()[-1]]
        return None
    
def extract_fields_from_resume(resume_text):
        # Prepare the resume text for the model input
        resume_text = clean_text(resume_text)

        prompts = [
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the first name of the person contained in the resume? (Provide just the first name of the person as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the last name of the person from this resume? (Provide just the last name as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the area code and phone number contained in this resume? (There may be none in which case respond with only: 'None') (Provide only the 10 digit phone number with no hyphens or parenthesis) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the email address of the person from this resume? (There may be none in which case respond only: 'None') (Provide just the email address as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the city, state, and/or country of the person from this resume? (There may be none in which case respond only: 'None;') (Provide just the location or address as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What are the security clearances contained in this resume? (There may be none in which case respond only: 'None') (Provide just the security clearance as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What are the certifications contained in this resume? (There may be none in which case respond only: 'None') (Provide just the certifications as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What are the technical skills contained this resume? (Respond with only a list of skils, seperating items with commas) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the education contained in this resume? (Only respond with the School, Degree, and Degree Level) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the employment history contained in this resume? (Provide just the employment formated as 'Company - Job Title') (Seperate entries with commas) (Respond in the form 'Answer: '):"
        ]

        max_new_tokens_values = [20, 20, 20, 20, 20, 20, 50, 150, 200, 150]
        responses = []

        x = 1
        for prompt, tokens in zip(prompts, max_new_tokens_values):
            # Tokenize the prompt
            input_ids = FileExtraction.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to('cuda')
            # Generate the response
            output_ids = FileExtraction.model_8bit.generate(input_ids, max_length=len(input_ids[0]) + tokens)
            # Decode the generated response
            generated_text = FileExtraction.tokenizer.decode(output_ids[0])
            # Append the response to the list, wrapping it inside a list
            responses.append(generated_text.replace(prompt, '').replace('\n', '  ').strip())

            print("Completed response:", x, "/ 10")
            x += 1

        # Extract fields from the generated response
        extracted_fields = {}
        
        field_names = ['first_name', 'last_name', 'phone_number', 'email_address', 'physical_address', 'security_clearance', 'certifications', 'skills', 'education', 'work_history']
        x = 1
        
        #Alternative approach using a basic SpaCy NER implementation to pull first and last names
        '''for generated_response, field_name, prompt in zip(responses, field_names, prompts):
            if field_name == "first_name" or field_name == "last_name":
                first = self.get_name_with_ner(resume_text)[0]
                last = self.get_name_with_ner(resume_text)[-1]
                if field_name == "first_name":
                    extracted_fields[field_name] = first
                else:
                    extracted_fields[field_name] = last
            else:
                answer_start_index = generated_response.rfind('Answer: ') + len('Answer: ')
                answer_end_index = generated_response.find('</s>') if generated_response.find('</s>') != -1 else None
                extracted_text = generated_response[answer_start_index:answer_end_index]
                extracted_fields[field_name] = extracted_text.strip()
                print("Completed extraction:", x, "/ 10")'''

        # Iterate over responses, field names, and prompts simultaneously
        for generated_response, field_name, prompt in zip(responses, field_names, prompts):
            
            # Find the starting index of the answer
            answer_start_index = generated_response.rfind('Answer:') + len('Answer:')
            
            # Determine the ending index of the answer
            answer_end_index = generated_response.find('</s>') if generated_response.find('</s>') != -1 else None
            
            # Extract the answer using the found indices
            extracted_text = generated_response[answer_start_index:answer_end_index]
            
            # Store the extracted answer in the dictionary
            extracted_fields[field_name] = extracted_text.strip()
            
            # Print a status update for the extraction progress
            print("Completed extraction:", x, "/ 10")
            
            x += 1



        return extracted_fields

upload_to_s3('resume_fields.csv')