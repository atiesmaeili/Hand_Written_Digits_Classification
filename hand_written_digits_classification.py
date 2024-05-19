# -*- coding: utf-8 -*-
#"""Hand written digits classification

#Mount Goggle Drive


from google.colab import drive
import zipfile
drive.mount('/content/drive')

#Sklearn Classification"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image

data_directory = '/content/drive/MyDrive/Lab_AI/images'

# Prepare training data
train_data = []
train_labels = []

for class_name in os.listdir(os.path.join(data_directory, 'train')):
    class_directory = os.path.join(data_directory, 'train', class_name)
    for img_name in os.listdir(class_directory):
        img_path = os.path.join(class_directory, img_name)
        img = image.load_img(img_path, target_size=(28, 28))
        img_array = image.img_to_array(img)
        train_data.append(img_array)
        train_labels.append(class_name)

# Convert data and labels to NumPy arrays
X_train = np.array(train_data)
y_train = np.array(train_labels)

# Prepare testing data
test_data = []
test_labels = []

for class_name in os.listdir(os.path.join(data_directory, 'test')):
    class_directory = os.path.join(data_directory, 'test', class_name)
    for img_name in os.listdir(class_directory):
        img_path = os.path.join(class_directory, img_name)
        img = image.load_img(img_path, target_size=(28, 28))
        img_array = image.img_to_array(img)
        test_data.append(img_array)
        test_labels.append(class_name)

# Convert data and labels to NumPy arrays
X_test = np.array(test_data)
y_test = np.array(test_labels)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Flatten the image data
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Create and train a logistic regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_flatten, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test_flatten)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""#CNN"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Define the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/sssssss/train',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=True  # Shuffle the data for better training
)

test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/sssssss/train',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=False  # Do not shuffle the test data
)

# Get the number of classes
num_classes = len(train_generator.class_indices)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Use num_classes as the output units

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // 32
)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
