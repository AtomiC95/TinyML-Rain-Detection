import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Load the TFLite Model
LITE_MODEL_PATH = "/home/nikolas/git/TinyML-Rain-Detection/model_creation/models/lite_model.tflite"
interpreter = tf.lite.Interpreter(model_path=LITE_MODEL_PATH)
interpreter.allocate_tensors()

# 2. Prepare the Test Data
TEST_DATA_ROOT = "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/test_spec"
IMG_WIDTH_HEIGHT = (126, 65)
BATCH_SIZE = 1

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=TEST_DATA_ROOT,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_WIDTH_HEIGHT,
    color_mode='grayscale',
    shuffle=False  # Important for matching predictions to labels later
)

test_ds = test_ds.map(lambda x, y: (x * 255.0, y))

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

all_predictions = []

for images, _ in test_ds:
    for img in images:
        # Scale the image to [0, 255] and cast to UINT8
        #img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        
        # Preprocess the image if necessary (e.g., resizing if the TFLite model expects a different size)
        img = tf.expand_dims(img, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])
        all_predictions.extend(predictions)

# 4. Generate Predictions
all_predictions = np.argmax(all_predictions, axis=1)
true_labels = np.concatenate([y for x, y in test_ds])
true_labels = np.argmax(true_labels, axis=1)

# 5. Create the Confusion Matrix
# Assuming you have class names in a list called class_names or define it based on your data structure
class_names = ["heavy_rain", "light_rain", "medium_rain", "no_rain"]  # Modify this accordingly

plt.figure(figsize=(10, 7))
matrix = confusion_matrix(true_labels, all_predictions)
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Compute accuracy
accuracy = np.mean(all_predictions == true_labels) * 100
print(f"Test accuracy with TFLite model: {accuracy:.2f}%")

