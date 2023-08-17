from pathlib import Path
import tensorflow as tf
import os
import random

DATA_ROOT = Path("/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_spec")

BATCH_SIZE=1
IMG_WIDTH_HEIGHT=(126,65)
IMG_CHANNELS = 1
INPUT_IMG_SHAPE=IMG_WIDTH_HEIGHT + (IMG_CHANNELS,)

KERAS_MODEL_PATH = "/home/nikolas/git/TinyML-Rain-Detection/model_creation/models/final_model_test.pb"
LITE_MODEL_PATH = "/home/nikolas/git/TinyML-Rain-Detection/model_creation/models/lite_model.tflite"
#NUM_EXAMPLES_IN_REP_DATASET = 400

def representative_dataset():
    NUM_EXAMPLES_FROM_EACH_CLASS = 91
    
    class_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        # List all image files in the directory
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
        
        # Randomly select a subset of images
        selected_images = random.sample(image_files, NUM_EXAMPLES_FROM_EACH_CLASS)
        
        for img_file in selected_images:
            img_path = os.path.join(class_dir, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=IMG_WIDTH_HEIGHT)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            data = tf.expand_dims(img_array, axis=0)
            #data = tf.cast(data, tf.float32)
            yield [data]


# initialize converter
converter = tf.lite.TFLiteConverter.from_saved_model(KERAS_MODEL_PATH)

# # define microcontroller optimization such as 8-bit quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset
# #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.float32
# converter.inference_output_type = tf.float32

# apply conversion
tflite_model = converter.convert()

# save tf_lite model
with open(LITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)


