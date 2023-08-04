from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from django.views import View
import torch
import numpy as np
import docx2txt
from fpdf import FPDF
import fitz
import requests
import json
import re

import csv
from io import BytesIO
import random
import time
import os
import boto3

s3 = boto3.client('s3', aws_access_key_id='AKIAZ257POCST3TRJZSS', aws_secret_access_key='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', region_name='us-east-2')

#Reference apps.py
from .apps import FileExtraction

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

class TextExtractionService(APIView):
    def post(self, request):
        uploaded_file = request.FILES.get('file')
        # Extract text from the uploaded file
        if uploaded_file.name.endswith('.txt'):
            extracted_text = uploaded_file
        else:
            extracted_text = self.convert_file_to_text(uploaded_file)
        return Response(extracted_text)

    # Function to convert a file to text
    def convert_file_to_text(self, uploaded_file):
        file_path = uploaded_file.name
        file = uploaded_file.read()
        # Check file type and extract text accordingly
        if file_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(file)
        elif file_path.endswith('.docx') or file_path.endswith('.doc'):
            text = self.extract_text_from_docx(file)
        else:
            raise ValueError("Unsupported file format")
        return text

    # Extract text from DOCX
    def extract_text_from_docx(self, docx_file):
        text = BytesIO(docx_file)
        text = docx2txt.process(text)
        text = text.replace('\n', ' ')
        return text

    # Extract text from PDF
    def extract_text_from_pdf(self, pdf_content):
        text = ""
        pdf_file = BytesIO(pdf_content)
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")  # Removed read() method here
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text().replace('\n', ' ')
        pdf_document.close()
        return text

class FieldsExtractionService(APIView):
    # Create your views here.
    def post(self, request):
        start_time = time.time()  # Start the timer

        uploaded_file = request.FILES.get('file')

        # Create a new request object with the file parameter
        data = {'file': uploaded_file}

        # Make a POST request to the TextExtractionService view
        response = requests.post("http://192.9.135.43:8000/TextExtractionService", files=data)

        # Extract the text from the response
        extracted_text = response.text

        # do something with query here
        parsed_resume = self.extract_fields_from_resume(extracted_text)
        print(parsed_resume)

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print("Elapsed Time:", elapsed_time, "seconds")  # Print the elapsed time

        #Append output to csv file
        append_to_csv('resume_fields.csv', parsed_resume)

        # Upload CSV to S3
        upload_to_s3('resume_fields.csv')

        return Response(parsed_resume)

    def clean_text(self, text):
        # Filter out special characters that are not alphanumeric, punctuation, '+' or '#'
        filtered_text = re.sub(r"[^a-zA-Z0-9 \t\n\.\+#\/\\\-:\)\(]", "", text)
        # Replace multiple spaces with single space
        filtered_text = re.sub(r'\s{2,}', ' ', filtered_text)
        return filtered_text

    def extract_fields_from_resume(self, resume_text):
        # Prepare the resume text for the model input
        resume_text = self.clean_text(resume_text)

        prompts = [
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What is the first name of the person from this resume? (Provide just the name as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What is the last name of the person from this resume? (Provide just the name as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What is the full 10 digit phone number contained in this resume? (There may be none in which case respond: None) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What is the email address of the person from this resume? (There may be none in which case respond: None) (Provide just the email address as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What is the physical address or general location of the person from this resume? (There may be none in which case respond: None) (Provide just the location or address as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What are the security clearances contained in this resume? (There may be none in which case respond: None) (Provide just the security clearance as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What are the certifications contained in this resume? (There may be none in which case respond: None) (Provide just the certifications as the response) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What are the technical skills contained this resume? (Respond with only a list of skils, seperating items with commas) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What is the education contained in this resume? (Only respond with the School, Degree, and Degree Level) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n{resume_text}\n\nUser: What is the employment history contained in this resume? (Format employment as 'Company - Job Title') (Seperate entries with commas) (Respond in the form 'Answer: '):"
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
        for generated_response, field_name, prompt in zip(responses, field_names, prompts):
            answer_start_index = generated_response.rfind('Answer: ') + len('Answer: ')
            answer_end_index = generated_response.find('</s>') if generated_response.find('</s>') != -1 else None
            extracted_text = generated_response[answer_start_index:answer_end_index]
            extracted_fields[field_name] = extracted_text.strip()
            print("Completed extraction:", x, "/ 10")
            x += 1


        return extracted_fields

class PDFGenerationService(View):
    def get(self, request):
        # Convert dict_keys object to list before accessing by index
        json_str = list(request.GET.keys())[0]
        params_dict = json.loads(json_str)

        fields = {
            "first_name": params_dict.get("first_name", ""),
            "last_name": params_dict.get("last_name", ""),
            "Certifications": params_dict.get("certificates", ""),
            "Clearance": params_dict.get("security_clearance", ""),
            "Education": params_dict.get("education", ""),
            "Work History": params_dict.get("work_history", ""),
            "Skills": params_dict.get("skills", "")
        }
        print("fields:", fields)
        output_path = self.create_pdf_from_output(fields)
        print(f"Output PDF created")

        data = {
            'path': output_path,
        }

        return JsonResponse(data)

    def create_pdf_from_output(self, output):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Set the font to Times New Roman, size 14
        pdf.set_font("Times", size=14)

        pdf_text = ""
        prev_key = None

        for key, value in output.items():
            if prev_key and key != prev_key:
                pdf_text += "\n"

            if key == "first_name":
                print(value)
                pdf_text += "Name: " + "\n"
                pdf_text += value + " "
                continue
            elif key == "last_name":
                pdf_text += value + "\n\n"
                continue
            else:
                pdf_text += key + ":\n"

            if isinstance(value, list):
                for item in value:
                    pdf_text += item + "\n"
            else:
                pdf_text += value + "\n"

            prev_key = key

        pdf.multi_cell(0, 10, txt=pdf_text, align="L")

        output_filename = "output.pdf"
        output_path = os.path.join(os.getcwd(), output_filename)
        pdf.output(output_path)
        print(output_path)

        # Upload to S3
        upload_to_s3(output_path)

        return output_path