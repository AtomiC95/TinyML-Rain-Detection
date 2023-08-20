import os
import random
import shutil

# Set the percentage of files to move to the validation and test sets
validation_percentage = 0.2
test_percentage = 0.1

classes = ["heavy_rain", "light_rain", "no_rain", "medium_rain"]

for cls in classes:
    files = os.listdir(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_data/{cls}")

    # Calculate the number of files to move to the validation and test sets
    num_validation_files = int(validation_percentage * len(files))
    num_test_files = int(test_percentage * len(files))

    # Randomly select the files for validation set
    validation_files = random.sample(files, num_validation_files)
    
    # Remove the validation files from the 'files' list to prevent them from being sampled for the test set
    for file in validation_files:
        files.remove(file)

    # Move the selected validation files to the validation_data folder
    for file_name in validation_files:
        source_path = os.path.join(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_data/{cls}", file_name)
        dest_path = os.path.join(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/val_data/{cls}", file_name)
        shutil.move(source_path, dest_path)
    
    # Now sample files for the test set from the remaining files
    test_files = random.sample(files, num_test_files)

    # Move the selected test files to the test_data folder
    for file_name in test_files:
        source_path = os.path.join(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_data/{cls}", file_name)
        dest_path = os.path.join(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/test_data/{cls}", file_name)
        shutil.move(source_path, dest_path)
