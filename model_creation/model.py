import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input

input_shape = (161,1025,1)
inputs = Input(shape=(161, 1025,1))

num_classes = 2

batch_size = 8

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    '/home/nikolas/git/bachelorthesis/model_creation/train_spec_dir',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    '/home/nikolas/git/bachelorthesis/model_creation/val_spec_dir',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)




# base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# for layer in base_model.layers:
#     layer.trainable = False

# x = base_model.output
x = tf.keras.layers.Conv2D(64,(3,3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(32,(3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(16,(3,3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)

predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(train_generator, epochs=5, steps_per_epoch=len(train_generator), validation_data=val_generator, validation_steps=len(val_generator))

model.save('peace_sign_model')

