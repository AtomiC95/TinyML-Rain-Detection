import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

#Load the TFLite Model
LITE_MODEL_PATH = "/home/nikolas/git/TinyML-Rain-Detection/model_creation/models/lite_model.tflite"
interpreter = tf.lite.Interpreter(model_path=LITE_MODEL_PATH)
interpreter.allocate_tensors()

#Prepare the Test Data
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
    shuffle=False 
)


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

all_predictions = []

for images, _ in test_ds:
    for img in images:
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img_array, axis=0)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])
        all_predictions.extend(predictions)


all_predictions = np.argmax(all_predictions, axis=1)
true_labels = np.concatenate([y for x, y in test_ds])
true_labels = np.argmax(true_labels, axis=1)


class_names = ["heavy_rain", "light_rain", "medium_rain", "no_rain"] 

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

