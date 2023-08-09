from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import pandas as pd
import os

nltk.download('punkt')  # Download tokenizer data for NLTK

# Load pre-trained GloVe word embeddings
glove_model = api.load("glove-wiki-gigaword-100")  # You can choose different dimensions (e.g., glove-wiki-gigaword-50)

# Folder path where the PNG images are stored
image_folder_path = "/Users/montrellnelson/Downloads/REI_IMAGES/"

# CSV file path containing serial numbers and corresponding ground truth captions
csv_file_path = "/Users/montrellnelson/Downloads/df_design_search_1000.csv"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Read the CSV file to get the serial numbers and ground truth captions
df = pd.read_csv(csv_file_path)

total_cos_similarity = 0
cos_similarity_list = []
num_pairs = len(df)

ground_truth_captions_dict = {}


# Dictionary to store generated captions for each image
generated_captions_dict = {}


for index, row in df.iterrows():
    serial_number = row['serial_no']
    ground_truth_caption = row['td_desc']

    # Load the image from the folder using the serial number as the image file name
    image_file_name = f"{serial_number}.jpg"  # Assuming the images have the ".jpg" extension
    image_path = os.path.join(image_folder_path, image_file_name)
    raw_image = Image.open(image_path).convert('RGB')

    # Conditional image captioning
    text = ""
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)  # Generate a new caption for each image
    generated_caption = processor.decode(out[0], skip_special_tokens=True)



    # Tokenize the generated and ground truth captions
    generated_caption_tokens = nltk.word_tokenize(generated_caption)
    ground_truth_tokens = nltk.word_tokenize(ground_truth_caption)

    # Get the word embeddings for each word in the captions (if available)
    generated_caption_vectors = [glove_model[word] for word in generated_caption_tokens if word in glove_model]
    ground_truth_vectors = [glove_model[word] for word in ground_truth_tokens if word in glove_model]

    # Calculate the average word embeddings to get the vector representations of the captions
    generated_caption_vector = sum(generated_caption_vectors) / len(generated_caption_vectors) if generated_caption_vectors else None
    ground_truth_vector = sum(ground_truth_vectors) / len(ground_truth_vectors) if ground_truth_vectors else None

    # Calculate Cosine Similarity
    cos_similarity = cosine_similarity([generated_caption_vector], [ground_truth_vector])[0][0] if generated_caption_vector is not None and ground_truth_vector is not None else None

    total_cos_similarity += cos_similarity if cos_similarity is not None else 0
    cos_similarity_list.append((cos_similarity, serial_number))

# Store the generated caption in the dictionary
    generated_captions_dict[serial_number] = generated_caption

# Store the ground truth caption in the dictionary
    ground_truth_captions_dict[serial_number] = ground_truth_caption
    
    # print("Cosine Similarity:", cos_similarity)
    # print("Ground Truth Caption:", ground_truth_caption)
    # print("Generated Caption:", generated_caption)

# Sort the list of cosine similarities in descending order

cos_similarity_list.sort(reverse=True, key=lambda x: x[0])

# # Calculate the average cosine similarity score
average_cos_similarity = total_cos_similarity / num_pairs
print("Average Cosine Similarity:", average_cos_similarity)



#TOP 5

def calculate_cosine_similarity(target_generated_caption, ground_truth_captions_dict):
    # Calculate the vector representation of the target generated caption
    target_generated_caption_tokens = nltk.word_tokenize(target_generated_caption)
    target_generated_caption_vectors = [glove_model[word] for word in target_generated_caption_tokens if word in glove_model]
    target_generated_caption_vector = sum(target_generated_caption_vectors) / len(target_generated_caption_vectors) if target_generated_caption_vectors else None

    # Calculate the cosine similarity with all other ground truth captions
    all_set_cos_similarity_list = []
    for serial_number, ground_truth_caption in ground_truth_captions_dict.items():
        if serial_number != target_serial_number:
            # Tokenize the ground truth caption
            ground_truth_tokens = nltk.word_tokenize(ground_truth_caption)
            ground_truth_vectors = [glove_model[word] for word in ground_truth_tokens if word in glove_model]
            ground_truth_vector = sum(ground_truth_vectors) / len(ground_truth_vectors) if ground_truth_vectors else None

            # Calculate Cosine Similarity
            all_set_cos_similarity = cosine_similarity([target_generated_caption_vector], [ground_truth_vector])[0][0] if target_generated_caption_vector is not None and ground_truth_vector is not None else None

            all_set_cos_similarity_list.append((all_set_cos_similarity, serial_number))

    # Sort the list of cosine similarities in descending order
    all_set_cos_similarity_list.sort(reverse=True, key=lambda x: x[0])

    # Get the top 5 highest cosine similarities
    top_5_similarities = all_set_cos_similarity_list[:5]

    # Get the top 10 highest cosine similarities
    top_10_similarities = all_set_cos_similarity_list[:10]

    # Get the top 20 highest cosine similarities
    top_20_similarities = all_set_cos_similarity_list[:20]

    return top_5_similarities

def evaluate_hits_misses(generated_captions_dict, ground_truth_captions_dict):
    # Dictionary to store the results (hit or miss) for each generated caption
    results_dict = {}

    for target_serial_number, target_generated_caption in generated_captions_dict.items():
        top_5_similarities = calculate_cosine_similarity(target_generated_caption, ground_truth_captions_dict)

        print(f"Top 5 closest ground truth captions for generated caption {target_serial_number}:")
        for similarity, serial_number in top_5_similarities:
            print(f"Ground Truth Serial Number: {serial_number}, Cosine Similarity: {similarity}")
            print("####################################################")

        # Check if the target serial number appears in the top 5
        target_serial_in_top_5 = any(serial_number == target_serial_number for _, serial_number in top_5_similarities)

        # If the target serial number is in the top 5, it's a hit; otherwise, it's a miss
        result = "Hit" if target_serial_in_top_5 else "Miss"

        # Store the result in the results dictionary
        results_dict[target_serial_number] = result

        # If it's a miss, print the additional information
        if result == "Miss":
            print(f"\nTarget Serial Number: {target_serial_number}")
            print(f"Generated Caption: {target_generated_caption}")
            print("Top 5 Ground Truth Captions:")
            for i, (similarity, serial_number) in enumerate(top_5_similarities, start=1):
                ground_truth_caption = ground_truth_captions_dict.get(serial_number, None)
                print(f"{i}. Ground Truth Serial Number: {serial_number}, Cosine Similarity: {similarity}")
                print(f"   Ground Truth Caption: {ground_truth_caption}")
            print("####################################################\n")

    # Print the results for each generated caption
    for serial_number, result in results_dict.items():
        print(f"Serial Number: {serial_number}, Result: {result}")

    total_generated_captions = len(generated_captions_dict)
    total_hits = sum(1 for result in results_dict.values() if result == "Hit")
    accuracy = total_hits / total_generated_captions

    # Print the accuracy
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(total_hits)
    print(total_generated_captions)


# Example usage:
# Assuming you have dictionaries `generated_captions_dict` and `ground_truth_captions_dict` containing generated captions and ground truth captions, respectively.
evaluate_hits_misses(generated_captions_dict, ground_truth_captions_dict)


#####################
# OUTPUT: 
# Accuracy: 6.00% 
# 60
# 1000

# ---------ONE EXAMPLE OF FULL OUTPUT---------------
#Top 5 closest ground truth captions for generated caption 87643300:
#Ground Truth Serial Number: 87849420, Cosine Similarity: 0.9486352801322937
####################################################
#Ground Truth Serial Number: 87712044, Cosine Similarity: 0.9484784007072449
####################################################
#Ground Truth Serial Number: 87892613, Cosine Similarity: 0.9478667974472046
####################################################
#Ground Truth Serial Number: 87807719, Cosine Similarity: 0.9453575015068054
####################################################
#Ground Truth Serial Number: 87897947, Cosine Similarity: 0.9452517032623291
####################################################

#Target Serial Number: 87643300
#Generated Caption: a close up of a cat laying on its back with paw prints
#Top 5 Ground Truth Captions:
#1. Ground Truth Serial Number: 87849420, Cosine Similarity: 0.9486352801322937
    #Ground Truth Caption: the mark consists of outline of a dog with a long curved tail standing on its hind legs with its mouth pointed upwards with the mouth open and its forepaws up against dotted lines. the dotted lines are used to show placement of the mark and are not claimed as a feature of the mark.  silhouettes of dogs; dogs rendered only as outlines, shadows, or silhouettes    dogs; puppies    costumed felines and those with human attributes, including cats of all sizes
#2. Ground Truth Serial Number: 87712044, Cosine Similarity: 0.9484784007072449
   #Ground Truth Caption: the mark consists of a silhouette of a man holding a large arrow-shaped sign in one hand above his head and with his other hand pointing in same direction as the arrow-shaped sign.  silhouettes of men; men depicted as shadows or silhouettes of men    street signs not attached to a support; advertising, signs, alone    rectangles that are completely or partially shaded
#3. Ground Truth Serial Number: 87892613, Cosine Similarity: 0.9478667974472046
   #Ground Truth Caption: the mark consists of two adjacent designs, on the left a stylized design of a leaf with an arc in the middle, on the right a design of a sideways droplet of oil with an arc inside.  more than one leaf, including scattered leaves, bunches of leaves not attached to branches    leaf, single; other leaves
#4. Ground Truth Serial Number: 87807719, Cosine Similarity: 0.9453575015068054
   #Ground Truth Caption: the mark consists of a stylized spade design, with the bottom right in a light blue shade and the remainder in blue.  spades, on playing cards
#5. Ground Truth Serial Number: 87897947, Cosine Similarity: 0.9452517032623291
   #Ground Truth Caption: the mark consists of a three-dimensional configuration of a lollipop candy consisting of three identical and equally distant circles within a single large circle with the backside having a hollow shape with a single bean like shape in the middle. the lollipop stick appearing in dotted lines is not a part of the mark and serves only to show the position of the mark.  suckers, candy; lollipops; candy on a stick, such as suckers    four circles; circles, exactly four circles

#-----------------------------------------------
# TOP 10 
#-----------------------------------------------

def calculate_cosine_similarity(target_generated_caption, ground_truth_captions_dict):
    # ... (previous code)
    # Calculate the vector representation of the target generated caption
    target_generated_caption_tokens = nltk.word_tokenize(target_generated_caption)
    target_generated_caption_vectors = [glove_model[word] for word in target_generated_caption_tokens if word in glove_model]
    target_generated_caption_vector = sum(target_generated_caption_vectors) / len(target_generated_caption_vectors) if target_generated_caption_vectors else None

    # Calculate the cosine similarity with all other ground truth captions
    all_set_cos_similarity_list = []
    for serial_number, ground_truth_caption in ground_truth_captions_dict.items():
        if serial_number != target_serial_number:
            # Tokenize the ground truth caption
            ground_truth_tokens = nltk.word_tokenize(ground_truth_caption)
            ground_truth_vectors = [glove_model[word] for word in ground_truth_tokens if word in glove_model]
            ground_truth_vector = sum(ground_truth_vectors) / len(ground_truth_vectors) if ground_truth_vectors else None

            # Calculate Cosine Similarity
            all_set_cos_similarity = cosine_similarity([target_generated_caption_vector], [ground_truth_vector])[0][0] if target_generated_caption_vector is not None and ground_truth_vector is not None else None

            all_set_cos_similarity_list.append((all_set_cos_similarity, serial_number))

    # Sort the list of cosine similarities in descending order
    all_set_cos_similarity_list.sort(reverse=True, key=lambda x: x[0])

    # Get the top 20 highest cosine similarities
    top_10_similarities = all_set_cos_similarity_list[:10]

    return top_10_similarities


def evaluate_hits_misses(generated_captions_dict, ground_truth_captions_dict):
    # ... (previous code)
    # Dictionary to store the results (hit or miss) for each generated caption
    results_dict = {}

    for target_serial_number, target_generated_caption in generated_captions_dict.items():
        top_10_similarities = calculate_cosine_similarity(target_generated_caption, ground_truth_captions_dict)

        print(f"Top 10 closest ground truth captions for generated caption {target_serial_number}:")
        for i, (similarity, serial_number) in enumerate(top_10_similarities, start=1):
            print(f"{i}. Ground Truth Serial Number: {serial_number}, Cosine Similarity: {similarity}")
            print("####################################################")

        # Check if the target serial number appears in the top 10
        target_serial_in_top_10 = any(serial_number == target_serial_number for _, serial_number in top_10_similarities)

        # If the target serial number is in the top 10, it's a hit; otherwise, it's a miss
        result = "Hit" if target_serial_in_top_10 else "Miss"

        # Store the result in the results dictionary
        results_dict[target_serial_number] = result

    # Print the results for each generated caption
    for serial_number, result in results_dict.items():
        print(f"Serial Number: {serial_number}, Result: {result}")

        total_generated_captions = len(generated_captions_dict)
        total_hits = sum(1 for result in results_dict.values() if result == "Hit")
        accuracy = total_hits / total_generated_captions

    # Print the accuracy
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(total_hits)
    print(total_generated_captions)

# Example usage:
# Assuming you have dictionaries `generated_captions_dict` and `ground_truth_captions_dict` containing generated captions and ground truth captions, respectively.
evaluate_hits_misses(generated_captions_dict, ground_truth_captions_dict)

#OUTPUT

#Top 10 closest ground truth captions for generated caption 87643300:
#1. Ground Truth Serial Number: 87849420, Cosine Similarity: 0.9486352801322937
####################################################
#2. Ground Truth Serial Number: 87712044, Cosine Similarity: 0.9484784007072449
####################################################
#3. Ground Truth Serial Number: 87892613, Cosine Similarity: 0.9478667974472046
####################################################
#4. Ground Truth Serial Number: 87807719, Cosine Similarity: 0.9453575015068054
####################################################
#5. Ground Truth Serial Number: 87897947, Cosine Similarity: 0.9452517032623291
####################################################
#6. Ground Truth Serial Number: 87711815, Cosine Similarity: 0.9446431994438171
####################################################
#7. Ground Truth Serial Number: 87916279, Cosine Similarity: 0.9446308016777039
####################################################
#8. Ground Truth Serial Number: 87961167, Cosine Similarity: 0.942855954170227
####################################################
#9. Ground Truth Serial Number: 87780110, Cosine Similarity: 0.9428118467330933
####################################################
#10. Ground Truth Serial Number: 87645122, Cosine Similarity: 0.9425172209739685

#Accuracy: 8.30%
#83
#1000

#----------------------------------------------
#TOP 20
#----------------------------------------------


def calculate_cosine_similarity(target_generated_caption, ground_truth_captions_dict):
    # ... (previous code)
    # Calculate the vector representation of the target generated caption
    target_generated_caption_tokens = nltk.word_tokenize(target_generated_caption)
    target_generated_caption_vectors = [glove_model[word] for word in target_generated_caption_tokens if word in glove_model]
    target_generated_caption_vector = sum(target_generated_caption_vectors) / len(target_generated_caption_vectors) if target_generated_caption_vectors else None

    # Calculate the cosine similarity with all other ground truth captions
    all_set_cos_similarity_list = []
    for serial_number, ground_truth_caption in ground_truth_captions_dict.items():
        if serial_number != target_serial_number:
            # Tokenize the ground truth caption
            ground_truth_tokens = nltk.word_tokenize(ground_truth_caption)
            ground_truth_vectors = [glove_model[word] for word in ground_truth_tokens if word in glove_model]
            ground_truth_vector = sum(ground_truth_vectors) / len(ground_truth_vectors) if ground_truth_vectors else None

            # Calculate Cosine Similarity
            all_set_cos_similarity = cosine_similarity([target_generated_caption_vector], [ground_truth_vector])[0][0] if target_generated_caption_vector is not None and ground_truth_vector is not None else None

            all_set_cos_similarity_list.append((all_set_cos_similarity, serial_number))

    # Sort the list of cosine similarities in descending order
    all_set_cos_similarity_list.sort(reverse=True, key=lambda x: x[0])

    # Get the top 20 highest cosine similarities
    top_20_similarities = all_set_cos_similarity_list[:20]

    return top_20_similarities


def evaluate_hits_misses(generated_captions_dict, ground_truth_captions_dict):
    # ... (previous code)
    # Dictionary to store the results (hit or miss) for each generated caption
    results_dict = {}

    for target_serial_number, target_generated_caption in generated_captions_dict.items():
        top_20_similarities = calculate_cosine_similarity(target_generated_caption, ground_truth_captions_dict)

        print(f"Top 20 closest ground truth captions for generated caption {target_serial_number}:")
        for i, (similarity, serial_number) in enumerate(top_20_similarities, start=1):
            print(f"{i}. Ground Truth Serial Number: {serial_number}, Cosine Similarity: {similarity}")
            print("####################################################")

        # Check if the target serial number appears in the top 20
        target_serial_in_top_20 = any(serial_number == target_serial_number for _, serial_number in top_20_similarities)

        # If the target serial number is in the top 20, it's a hit; otherwise, it's a miss
        result = "Hit" if target_serial_in_top_20 else "Miss"

        # Store the result in the results dictionary
        results_dict[target_serial_number] = result

    # Print the results for each generated caption
    for serial_number, result in results_dict.items():
        print(f"Serial Number: {serial_number}, Result: {result}")

        total_generated_captions = len(generated_captions_dict)
        total_hits = sum(1 for result in results_dict.values() if result == "Hit")
        accuracy = total_hits / total_generated_captions

    # Print the accuracy
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(total_hits)
    print(total_generated_captions)

# Example usage:
# Assuming you have dictionaries `generated_captions_dict` and `ground_truth_captions_dict` containing generated captions and ground truth captions, respectively.
evaluate_hits_misses(generated_captions_dict, ground_truth_captions_dict)

#OUTPUT

#Top 20 closest ground truth captions for generated caption 87643300:
#1. Ground Truth Serial Number: 87849420, Cosine Similarity: 0.9486352801322937
####################################################
#2. Ground Truth Serial Number: 87712044, Cosine Similarity: 0.9484784007072449
####################################################
#3. Ground Truth Serial Number: 87892613, Cosine Similarity: 0.9478667974472046
####################################################
#4. Ground Truth Serial Number: 87807719, Cosine Similarity: 0.9453575015068054
####################################################
#5. Ground Truth Serial Number: 87897947, Cosine Similarity: 0.9452517032623291
####################################################
#6. Ground Truth Serial Number: 87711815, Cosine Similarity: 0.9446431994438171
####################################################
#7. Ground Truth Serial Number: 87916279, Cosine Similarity: 0.9446308016777039
####################################################
#8. Ground Truth Serial Number: 87961167, Cosine Similarity: 0.942855954170227
####################################################
#9. Ground Truth Serial Number: 87780110, Cosine Similarity: 0.9428118467330933
####################################################
#10. Ground Truth Serial Number: 87645122, Cosine Similarity: 0.9425172209739685
####################################################
#11. Ground Truth Serial Number: 88176357, Cosine Similarity: 0.9420943856239319
####################################################
#12. Ground Truth Serial Number: 88176363, Cosine Similarity: 0.9420943856239319
####################################################

#Accuracy: 11.30%
#113
#1000
