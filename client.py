import os

import flwr as fl
import tensorflow as tf


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



# Define the paths to the dataset and label files
train_dir = 'D:\Sowmiya\Data_preprocessing\Final_dataset\data_train_300'
val_dir = 'D:\Sowmiya\Data_preprocessing\Final_dataset\data_val'
test_dir = 'D:\Sowmiya\Data_preprocessing\Final_dataset\data_test'

train_labels_file = 'D:\Sowmiya\Data_preprocessing\Final_dataset\_train_300.txt'
val_labels_file = 'D:\Sowmiya\Data_preprocessing\Final_dataset\_val_64.txt'
test_labels_file = 'D:\Sowmiya\Data_preprocessing\Final_dataset\_test_64.txt'



input_shape = (224, 224, 3)

def load_dataset(directory, labels_file):
    image_paths = []
    labels = []

    # Load image paths and labels from the labels file
    with open(labels_file, 'r') as file:
        for line in file:
            line_parts = line.split()
            image_file = line_parts[0]
            label = int(line_parts[1])
            image_path = os.path.join(directory, image_file)
            image_paths.append(image_path)
            labels.append(label)

    # Load images and convert them to numpy arrays
    images = []
    for image_path in image_paths:
        image = load_img(image_path, target_size=input_shape[:2])
        image = img_to_array(image)
        images.append(image)

    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')

    return images, labels


# In[58]:


train_images, train_labels = load_dataset(train_dir, train_labels_file)
val_images, val_labels = load_dataset(val_dir, val_labels_file)
test_images, test_labels = load_dataset(test_dir, test_labels_file)


# In[59]:




train_images /= 255.0
val_images /= 255.0
test_images /= 255.0
num_classes = 8

# Convert labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels - 1, num_classes)
val_labels = tf.keras.utils.to_categorical(val_labels - 1, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels - 1, num_classes)





import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
input_shape = (224, 224, 3)
# Load the pre-trained DenseNet121 model
densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers so that they are not updated during training
for layer in densenet_model.layers:
    layer.trainable = False

# Create a new model
model = Sequential()

# Add the pre-trained DenseNet121 model as a layer
model.add(densenet_model)

# Add a global average pooling layer to reduce the spatial dimensions
model.add(GlobalAveragePooling2D())

# Add a new fully connected layer
model.add(Dense(128, activation='relu'))

# Add the final output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[63]:


# Train the model



import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class MarsClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_images, train_labels, epochs=50, batch_size=32, validation_data=(val_images, val_labels))
        return model.get_weights(), len(train_images), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        test_loss, test_accuracy = model.evaluate(test_images, test_labels)
        
        # Predict on test data
        test_predictions = model.predict(test_images)
        test_predictions = np.argmax(test_predictions, axis=1)

        # Calculate additional metrics: precision, recall, and F1 score
        precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, test_predictions, average='weighted')
        accuracy = accuracy_score(test_labels, test_predictions)

        return test_loss, len(test_images), {"accuracy": test_accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MarsClient())
