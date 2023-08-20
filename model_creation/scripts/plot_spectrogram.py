import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def plot_image(img_path, subplot_position):
    img = mpimg.imread(img_path)
    plt.subplot(2, 2, subplot_position)  # 2 rows, 2 columns for 4 subplots
    plt.imshow(img, aspect='auto', origin='lower', extent=[0, 1, 20, 4000])
    # Extract the parent directory name and set it as the title
    parent_directory_name = os.path.basename(os.path.dirname(img_path))
    plt.title(parent_directory_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
images = [
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_spec/heavy_rain/chunk_5_log.jpg",
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_spec/light_rain/chunk_12_log.jpg",
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_spec/medium_rain/chunk_5_log.jpg",
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_spec/no_rain/chunk_9_log.jpg"
]


plt.figure(figsize=(10, 8))

for idx, img in enumerate(images, 1):
    plot_image(img, idx)

plt.tight_layout()
plt.show()



