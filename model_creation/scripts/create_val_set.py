import os
import random
import shutil

# Set the percentage of files to move to the validation set
validation_percentage = 0.2


classes = ["heavy_rain", "light_rain", "no_rain", "medium_rain"]
for cls in classes:
    files = os.listdir(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_data/{cls}")

    # Calculate the number of files to move to the validation set
    num_validation_files = int(validation_percentage * len(files))

    # Randomly select the files to move to the validation set
    validation_files = random.sample(files, num_validation_files)

    # Move the selected validation files to the validation_data folder
    for file_name in validation_files:
        source_path = os.path.join(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_data/{cls}", file_name)
        dest_path = os.path.join(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/val_data/{cls}", file_name)
        shutil.move(source_path, dest_path)