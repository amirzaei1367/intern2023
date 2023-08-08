## The code below is for ranking resumes in order of most qualified to least qualified. Lines 63 to 70 for the code for Cosine Similarity ##

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
import pandas as pd
import re  # Import the re module for regular expressions
from sentence_transformers import SentenceTransformer

def extract_resume_keywords(resume_csv_path):
    """
    Extracts keywords from resumes using semantic search.
    """
    extracted_keywords = []

    # Read the CSV file
    df = pd.read_csv(resume_csv_path)

    # Load the pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    for _, row in df.iterrows():
        # Extract keywords using semantic search
        resume_text = str(row['Resume_str'])  # Convert to string explicitly
        keywords = []

        # Define the keyword patterns
        keyword_patterns = {
            'First Name': r"First Name:(.*?)\n",
            'Last Name': r"Last Name:(.*?)\n",
            'Security Clearance': r"Security Clearance:(.*?)\n",
            'Education': r"Education:(.*?)\n",
            'Employment History': r"Employment History:(.*?)\n",
            'Certificates': r"Certificates:(.*?)\n",
            'Summary of Expertise': r"Summary of Expertise:(.*?)\n"
        }

        for keyword, pattern in keyword_patterns.items():
            keyword_match = re.search(pattern, resume_text, re.DOTALL)
            keyword_value = keyword_match.group(1).strip() if keyword_match else ""
            keywords.append((keyword, keyword_value))

        # Append the extracted keywords to the list
        extracted_keywords.append({
            "Resume": resume_text,
            "Keywords": keywords
        })

    return extracted_keywords

def calculate_similarity(description, resume_text, metric='euclidean'):
    """
    Calculates the similarity between a description and a resume using a given distance metric.
    """
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    # Encode the description and resume text into their respective embeddings
    description_embedding = model.encode([description])
    resume_embedding = model.encode([resume_text])

    # Calculate the distance between the embeddings based on the selected metric
    if metric == 'euclidean':
        similarity_scores = -euclidean_distances(description_embedding, resume_embedding)[0][0]
    elif metric == 'cosine':
        similarity_scores = cosine_distances(description_embedding, resume_embedding)[0][0]
    elif metric == 'manhattan':
        similarity_scores = -manhattan_distances(description_embedding, resume_embedding)[0][0]
    else:
        raise ValueError(f"Invalid metric: {metric}. Please choose 'euclidean', 'cosine', or 'manhattan'.")

    return similarity_scores

resume_csv_path = r"C:\Users\nicholas.carter\Downloads\Resume.csv"
resume_keywords = extract_resume_keywords(resume_csv_path)

description = "We are looking for a candidate with programming skills using Python, Java, or C++ and at least 5 years of experience in network security"

# Define the metrics to evaluate
metrics = ['euclidean', 'cosine', 'manhattan']

for metric in metrics:
    print(f"Using {metric} distance:")
    # Calculate similarity between the description and each resume
    for resume in resume_keywords:
        resume_text = resume['Resume']
        similarity_scores = calculate_similarity(description, resume_text, metric=metric)
        resume['Similarity'] = similarity_scores

    # Sort resumes by similarity score in descending order
    resume_keywords.sort(key=lambda x: x['Similarity'], reverse=True)

    # Display the ranked resumes
    for rank, resume in enumerate(resume_keywords, 1):
        print("Rank:", rank)
        print("Resume:", resume['Resume'])
        print("Similarity Score:", resume['Similarity'])
        print("Keywords:", resume['Keywords'])
        print("-" * 40)
