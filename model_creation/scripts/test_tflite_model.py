import tensorflow as tf
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

input_shape = (126,65)

# Load TFLite model
interpreter = tf.lite.Interpreter("./model_creation/models/lite_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    'TinyML-Rain-Detection/model_creation/model_data/test_spec',
    target_size=input_shape,
    batch_size=4,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

predicted_labels = []
true_labels = []

for i in range(len(test_generator)):
    images, labels = test_generator[i]
    
    true_label_batch = np.argmax(labels, axis=1)
    
    for image, true_label in zip(images, true_label_batch):
        # Expand the dimensions of the image to simulate a batch of size 1
        image = np.expand_dims(image, axis=0)
        
        # Set the tensor
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run the inference
        interpreter.invoke()

        # Retrieve the output from the tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(output_data)
        
        predicted_labels.append(predicted_label)
        true_labels.append(true_label)

# Define the class names if you have them
cm = confusion_matrix(true_labels, predicted_labels)

class_names = ["heavy_rain", "light_rain", "medium_rain", "no_rain"]

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix des TF Lite Modells')
plt.show()
