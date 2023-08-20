import tensorflow as tf
from keras_tuner.tuners import RandomSearch
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from seed_config import set_seed

set_seed() 


model_input_shape = (126, 65, 1)
inputs = tf.keras.layers.Input(shape=model_input_shape, dtype=tf.float32)
input_shape = (126,65)

def plot_accuracy_vs_epoch(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = epochs = np.arange(1, len(val_acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Accuracy (Training)')
    plt.plot(epochs, val_acc, 'g--', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoche')
    plt.grid('minor')

def plot_loss_vs_epoch(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = epochs = np.arange(1, len(val_loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Loss (Training)')
    plt.plot(epochs, val_loss, 'g--', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoche')
    plt.grid('minor')

def model():

    x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax', kernel_initializer='glorot_uniform')(x)

    model = tf.keras.models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=7.0887e-05)


    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model



train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    '/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/test_spec',
    target_size=input_shape,
    batch_size=1,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

train_generator = train_datagen.flow_from_directory(
    '/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_spec',
    target_size=input_shape,
    batch_size=1,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

val_generator = val_datagen.flow_from_directory(
    '/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/val_spec',
    target_size=input_shape,
    batch_size=1,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=3)

my_model = model()

history = my_model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        epochs=20,
                        callbacks=[early_stopping_callback])



plot_accuracy_vs_epoch(history)
plot_loss_vs_epoch(history)
plt.show()

my_model.save('/home/nikolas/git/TinyML-Rain-Detection/model_creation/models/final_model.pb')


# Make predictions on the test set
predictions = my_model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Generate the confusion matrix
confusion_mtx = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(confusion_mtx)

# Plot the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt='b',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

accuracy = np.sum(predicted_classes == true_classes) / len(true_classes) * 100
print(f"Test accuracy: {accuracy:.2f}%")