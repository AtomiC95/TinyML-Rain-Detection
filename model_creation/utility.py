import numpy as np
import matplotlib.pyplot as plt
import itertools

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Provided confusion matrix
provided_matrix = np.array([
    [8, 1, 2, 1],
    [2, 8, 2, 0],
    [0,1, 11, 0],
    [0, 1, 0, 11]
])

# Classes for the confusion matrix
classes = ['heavy_rain', 'light_rain', 'medium_rain', 'no_rain']

# Plot the provided confusion matrix
plot_confusion_matrix(provided_matrix, classes)
plt.show()

# import tensorflow
# from keras.utils import plot_model


# model = tensorflow.keras.models.load_model("/home/nikolas/git/TinyML-Rain-Detection/model_creation/models/final_model.pb")
# model.summary()
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

