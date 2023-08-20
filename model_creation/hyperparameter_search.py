import tensorflow as tf
from keras_tuner.tuners import RandomSearch
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
from seed_config import set_seed

set_seed() 

model_input_shape = (126, 65, 1)
inputs = tf.keras.layers.Input(shape=model_input_shape, dtype=tf.float32)
input_shape = (126,65)


def build_model(hp):
    # Define hyperparameters for model architecture
    hp_units1 = hp.Int('units1', min_value=1, max_value=8, step=1)
    hp_units2 = hp.Int('units2', min_value=1, max_value=4, step=1)


    x = tf.keras.layers.Conv2D(hp_units1, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(hp_units2, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)

    # Define hyperparameters for optimizer
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-07, max_value=1e-4)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    '/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_spec',
    target_size=input_shape,
    batch_size=4,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

val_generator = val_datagen.flow_from_directory(
    '/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/val_spec',
    target_size=input_shape,
    batch_size=4,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)
# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=3)
early_stopping_callback_acc = EarlyStopping(monitor='accuracy', patience=3)

# Define the TensorBoard callback
log_dir = 'model_creation/tensorboard_log'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=200,
    directory='model_creation/tuner_dir',
    project_name='rain_detection'
)


history = tuner.search(train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        epochs=20,
                        callbacks=[early_stopping_callback, early_stopping_callback_acc])


best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

tuner.results_summary

print("Best Hyperparameters:")
for name, value in best_hyperparameters.values.items():
    print(f"{name}: {value}")
print(f"Validation Accuracy: {best_model.evaluate(val_generator)[1]}")
