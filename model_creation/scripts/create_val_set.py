import os
import random
import shutil

# Set the percentage of files to move to the validation set
validation_percentage = 0.05


# Get a list of all the files in the train_data folder
files = os.listdir("model_creation/train_data/rain")

# Calculate the number of files to move to the validation set
num_validation_files = int(validation_percentage * len(files))

# Randomly select the files to move to the validation set
validation_files = random.sample(files, num_validation_files)

# Move the selected validation files to the validation_data folder
for file_name in validation_files:
    source_path = os.path.join("model_creation/train_data/rain", file_name)
    dest_path = os.path.join("model_creation/test_data/rain", file_name)
    shutil.move(source_path, dest_path)