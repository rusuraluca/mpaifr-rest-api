import os
import shutil
from sklearn.model_selection import train_test_split

# Define the source directory and target directories
source_dir = '../agedb'
train_dir = 'agedb_training'
validate_dir = 'agedb_validation'

# Create training and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validate_dir, exist_ok=True)

# Initialize a dictionary to hold the mapping of names to images
person_images = {}

# List all files in the agedb directory
for file_name in os.listdir(source_dir):
    if file_name.endswith('.jpg'):
        # Extract name, age, and sex from the file name
        parts = file_name.split('_')
        name = parts[1]
        age = parts[2]
        gender = parts[3].split('.')[0][-1]  # Extracts 'f' or 'm' from the file name
        person_key = f"{name}_{gender}"

        # Add the image to the person's list
        if person_key not in person_images:
            person_images[person_key] = []
        person_images[person_key].append(file_name)

# Split data into training and validation sets
for person_key, images in person_images.items():
    if len(images) > 1:
        # Only split if there are at least 2 images
        train_images, validate_images = train_test_split(images, test_size=0.2, random_state=42)
    else:
        # If only one image, keep it in the training set
        train_images = images
        validate_images = []

    # Create subdirectories for each person in training and validation directories
    train_subdir = os.path.join(train_dir, person_key)
    validate_subdir = os.path.join(validate_dir, person_key)
    os.makedirs(train_subdir, exist_ok=True)
    if validate_images:
        os.makedirs(validate_subdir, exist_ok=True)

    # Move the training images
    for i, image in enumerate(train_images):
        age = image.split('_')[2]
        target_file_name = f"{str(i).zfill(3)}_{age}.jpg"
        shutil.move(os.path.join(source_dir, image), os.path.join(train_subdir, target_file_name))

    # Move the validation images, if any
    for i, image in enumerate(validate_images):
        age = image.split('_')[2]
        target_file_name = f"{str(i).zfill(3)}_{age}.jpg"
        shutil.move(os.path.join(source_dir, image), os.path.join(validate_subdir, target_file_name))

print("Dataset organization complete.")
