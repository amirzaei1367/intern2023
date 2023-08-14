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
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import csv
from io import BytesIO
import random
import time
import os
import boto3
import spacy
import textwrap
from textwrap import fill

s3 = boto3.client('s3', aws_access_key_id='AKIAZ257POCST3TRJZSS', aws_secret_access_key='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', region_name='us-east-2')
nlp = spacy.load("en_core_web_sm")

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


def BatchFieldsExtractionService():
    # Expand the directory path
    RESUMES_DIR = '~/database_project/resumes'
    resumes_path = os.path.expanduser(RESUMES_DIR)

    # List all files in the directory
    all_files = glob.glob(os.path.join(resumes_path, '*'))
    # Iterate over each file and extract details
    extracted_data = []
    for file_path in all_files:
        extracted_fields = requests.post("http://192.9.135.43:8000/FieldsExtractionService", file_path)
        #Append output to csv file
        append_to_csv('resume_fields.csv', extracted_fields)

        # Upload CSV to S3
        upload_to_s3('resume_fields.csv')

    return Response(extracted_data)

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
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text().replace('\n', ' ')
        pdf_document.close()
        return text
495%494
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
        filtered_text = re.sub(r"[^a-zA-Z0-9 \t\n\.\+#\/\\\-:\)\(@]", "", text)
        # Replace multiple spaces with single space
        filtered_text = re.sub(r'\s{2,}', ' ', filtered_text)
        return filtered_text

    def get_name_with_ner(self, text):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Take the first name of the recognized person's name
                return [ent.text.split()[0],ent.text.split()[-1]]
        return None
    
    def extract_fields_from_resume(self, resume_text):
        # Prepare the resume text for the model input
        resume_text = self.clean_text(resume_text)

        prompts = [
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the first name of the person contained in the resume? (Provide just the first name of the person as the response with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the last name of the person from this resume? (Provide just the last name as the response with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the 10-digit phone number contained in this resume? (There may be none in which case respond with only: 'None') (Provide only the 10-digit phone number with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the email address (typically denoted by an '@') contained this resume? (There may be none in which case respond only: 'None') (Provide just the email address as the response with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the city, state, and/or country of the person from this resume? (There may be none in which case respond only: 'None;') (Provide just the location or address as the response with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What are the security clearances contained in this resume? (There may be none in which case respond only with the string: 'None') (Provide just the security clearance as the response with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What are the certifications contained in this resume? (There may be none in which case respond only: 'None') (Provide just the certifications as the response with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What are the technical skills contained this resume? (Respond with only a list of skills, separating items with commas with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the education contained in this resume? (Only respond with the School, Degree, and Degree Level with no additional context) (Respond in the form 'Answer: '):",
            f"LLAMA: Here is a resume:\n"+resume_text+"\n\nUser: What is the employment history contained in this resume? (Provide just the employment formatted as 'Company - Job Title' with no additional context) (Separate entries with commas) (Respond in the form 'Answer: '):"
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

class PDFGenerationService(View):
    def get(self, request):
        json_str = list(request.GET.keys())[0]
        params_dict = json.loads(json_str)
        print(params_dict)
        fields = {
            "first_name": params_dict.get("first_name", ""),
            "last_name": params_dict.get("last_name", ""),
            "Phone Number": params_dict.get("phone_number", ""),
            "Email Address": params_dict.get("email_address", ""),
            "Physical Address": params_dict.get("physical_address", ""),
            "Certifications": params_dict.get("certificates", ""),
            "Clearance": params_dict.get("security_clearance", ""),
            "Education": params_dict.get("education", ""),
            "Work History": params_dict.get("work_history", ""),
            "Skills": params_dict.get("skills", "")
        }

        output_path = self.create_pdf_from_output(fields)
        data = {'path': output_path}
        return JsonResponse(data)

    def create_pdf_from_output(self, output):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'STIXGeneral'

        fig, ax = plt.subplots(figsize=(8.5, 11))

        logo_path = "3747.png"
        logo = mpimg.imread(logo_path)
        logobox = OffsetImage(logo, zoom=0.2)
        logo_ab = AnnotationBbox(logobox, (0.08, 0.96), frameon=False)
        ax.add_artist(logo_ab)
        ax.axvline(x=.5, ymin=0, ymax=1, color='#007ACC', alpha=0.0, linewidth=50)
        plt.axvline(x=.99, color='#000000', alpha=0.5, linewidth=300)
        plt.axhline(y=.88, xmin=0, xmax=1, color='#ffffff', linewidth=3)
        ax.set_facecolor('white')
        plt.axis('off')

        max_width_left = 55   # Char width for the left side text
        max_width_right = 30  # Char width for the skills section on the right side

        # Annotations with titles and details with text wrapping
        def wrapped_text_annotation(text, chars_per_line):
            text = '- ' + text
            text = text.replace(', ', '\n- ')
            text = text.strip('.')
            textwrap.fill(text, chars_per_line)
            return text


        # Assuming y starts from top to bottom (e.g., 1.0 at the top and 0.0 at the bottom)
        current_y_position = 0.84

        # Function to add title and content and return the new y position
        def annotate_section(title, content, current_y, title_fontsize=14, content_fontsize=10, color='#58C1B2'):
            # Calculate the space needed based on the number of lines in the content and title
            num_lines = len(content.split('\n'))
            delta_y = num_lines * 0.02  # Adjust multiplier as needed for spacing
            
            plt.annotate(title, (.01, current_y), weight='bold', fontsize=title_fontsize, color=color)
            
            # Start content below the title
            content_y = current_y - 0.04
            
            for line in content.split('\n'):
                plt.annotate(line, (.01, content_y), weight='regular', fontsize=content_fontsize, color='#000000')
                content_y -= 0.02  # Move further down for the next line

            return content_y - 0.02  # Adjust spacing after the section is over

        # Name
        plt.annotate(output["first_name"] + " " + output["last_name"], (.01, current_y_position), weight='bold', fontsize=20)
        current_y_position -= 0.06  # Adjusting for the space taken up by the name

        # Education
        education_text = wrapped_text_annotation(output["Education"], max_width_left)
        current_y_position = annotate_section('Education:', education_text, current_y_position)

        # Work History
        work_history_text = wrapped_text_annotation(output["Work History"], max_width_left)
        current_y_position = annotate_section('Work History:', work_history_text, current_y_position)

        # Certifications
        certification_text = wrapped_text_annotation(output["Certifications"], max_width_left)
        current_y_position = annotate_section('Certifications:', certification_text, current_y_position)

        # Security Clearance
        security_clearance_text = wrapped_text_annotation(output["Clearance"], max_width_left)
        current_y_position = annotate_section('Security Clearance:', security_clearance_text, current_y_position)


        
        # Starting y-coordinates
        contact_y_start = .96
        skills_y_start = .84

        # Contact details
        contact_details_list = [output["Phone Number"].strip('.'), output["Email Address"].strip('.'), output["Physical Address"].strip('.')]
        for detail in contact_details_list:
            plt.annotate(detail, (.7, contact_y_start), weight='regular', fontsize=9, color='#ffffff')
            contact_y_start -= 0.02  # Move further down for the next line

        # Skills
        plt.annotate('Skills:', (.7, skills_y_start), weight='bold', fontsize=12, color='#ffffff')
        skills_text = wrapped_text_annotation(output["Skills"], max_width_right)
        skills_text = skills_text.strip('.')
        skills_y_start -= 0.02  # Move down for the first skill
        for skill in skills_text.split('\n'):
            plt.annotate(skill, (.7, skills_y_start), weight='regular', fontsize=10, color='#000000')
            skills_y_start -= 0.02  # Move further down for the next skill


        plt.savefig(output["first_name"]+output["last_name"]+'.pdf', format="pdf", dpi=300)
        output_path = os.path.join(os.getcwd(), "output.pdf")

        # Upload to S3
        upload_to_s3(output_path)

        return output_path