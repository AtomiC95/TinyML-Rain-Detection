from pathlib import Path
import tensorflow as tf
import numpy

DATA_ROOT = Path("model_creation/model_data/train_spec_dir")

BATCH_SIZE=8
IMG_WIDTH_HEIGHT=(17,129)
IMG_CHANNELS = 1
INPUT_IMG_SHAPE=IMG_WIDTH_HEIGHT + (IMG_CHANNELS,)

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory= DATA_ROOT,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_WIDTH_HEIGHT,
    color_mode='grayscale',
    )


KERAS_MODEL_PATH = "model_creation/models/peace_sign_model"
#assert KERAS_MODEL_PATH.exists(), "The path to your Keras model does not exist!"
LITE_MODEL_PATH = "model_creation/models/lite_model.tflite"
FULL_QUANTIZATION = False
NUM_EXAMPLES_IN_REP_DATASET = 100

def representative_dataset():
    """
    len(train_ds.as_numpy_iterator) == 100
    len(batch) == 2
    batch[0].shape == (32, 240, 240, 3)
    batch[1].shape == (32, 3)
    """
    img_count = 0
    for batch in train_ds.as_numpy_iterator():
        for img in batch[0]:
            data = tf.expand_dims(img, axis=0) # re-wrap into a batch of 1 img
            data = data.numpy()
            data = data.astype(numpy.float32)
            img_count += 1
            yield [data]
            
            if img_count == NUM_EXAMPLES_IN_REP_DATASET: break
        
        if img_count == NUM_EXAMPLES_IN_REP_DATASET: break

# initialize converter
converter = tf.lite.TFLiteConverter.from_saved_model(KERAS_MODEL_PATH)

# define microcontroller optimization such as 8-bit quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
if FULL_QUANTIZATION:
    # whether to fully quantize
    # in the TinyML book this is missing
    # the TF documentation recommends this

    # NOTE: uint8 is deprecated in tflite-micro 
    # https://github.com/tensorflow/tflite-micro/issues/216
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
tflite_quant_model = converter.convert()

# apply conversion
tflite_model = converter.convert()

# save tf_lite model
with open(LITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)


