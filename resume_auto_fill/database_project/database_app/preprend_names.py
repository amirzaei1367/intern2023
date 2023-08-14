import os
import random

def get_random_name():
    first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Katie", "Michael", "Sarah", "David", "Amanda"]
    last_names = ["Smith", "Doe", "Brown", "Johnson", "Martin", "Lee", "Walker", "Rodriguez", "Garcia", "Martinez"]

    first_name = random.choice(first_names)
    last_name = random.choice(last_names)

    return f"{first_name} {last_name}"

def prepend_name_to_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    name = get_random_name()
    updated_content = name + "\n\n" + content

    with open(file_path, 'w') as f:
        f.write(updated_content)

def main(directory):
    # List all files in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a text file
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            prepend_name_to_file(file_path)

    print(f"Added random names to all text files in {directory}")

if __name__ == '__main__':
    dir_path = input("Enter the directory path: ")
    main(dir_path)
