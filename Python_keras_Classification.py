# Python tutorial using Keras with tensorflow backend support for a multi-hidden layer Artificial Deep Neural Network (DNN) for classification on the fashion MNIST dataset.
# Deep learning is an artificial intelligence function that imitates the workings of the human brain in processing data and creating patterns for use in decision making.
# Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled.

# Import python libraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# check the tensorflow version currently running
#print(tf.__version__)

# Import the MNIST fashion dataset
theData = keras.datasets.fashion_mnist
# Split up the training and the testing sub datasets
(train_images, train_labels), (test_images, test_labels) = theData.load_data()
# Set the names for the label dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Analyze the training images dataset's shape
#print('Training Images Shape: ', train_images.shape, '\n')
# Analyze the length of the training dataset labels
#print('Training Images Length: ', len(train_labels), '\n')
# Analyze the training labels dataset
#print('Training Labels: ', train_labels, '\n')
# Analyze the testing labels dataset
#print('Testing Labels: ', test_labels, '\n')
# Analyze the testing images dataset's shape
#print('Testing Images Shape: ', test_images.shape, '\n')
# Analyze the length of the testing dataset labels
#print('Testing Image Length: ', len(test_labels), '\n')

'''
# Plot the 1st image of the training dataset for visualization
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''
'''
# Plot the first 25 images with label from the training dataset for visualization
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

# Preprocess & Normalize the training and testing datasets
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the sequential DNN model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Define the compiler for the sequential DNN model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the defined sequential DNN model
model.fit(train_images, train_labels, epochs=5)
# Evaluate the sequential DNN model
test_loss, test_acc = model.evaluate(test_images, test_labels)
# Analyze the testing accuracy of the defined sequential DNN model
print('\n Test accuracy: ', test_acc, '\n')
# Analyze the model's summary
print("Model's Summary", model.summary(), '\n')

# Define the model's prediction using the predict function
predictions = model.predict(test_images)
# Analyze the model's prediction on the first image
print("Model's Prediction on the First Image Matrix: ", predictions[0], '\n')
# Analyze the highest level of confidence value of the first image
print("Model's Prediction on the First Image Label: ", np.argmax(predictions[0]), '\n')
# Analyze the label of the First Image of the dataset
print("First Image Label: ", test_labels[0], '\n')

# Plot the first ten images of the dataset with there predictions and labels for visualization
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

'''
# plot the first image with its corresponding prediction and label for visualization
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
'''
'''
# plot the thirteen image with its corresponding prediction and label for visualization
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
'''
'''
# Plot the first X test images, their predicted label, and the true label for visualization
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
'''

# Make a prediction using the fully trained model on a single image
# Grab an image from the test dataset
img = test_images[0]
# Anaylze the single image shape
print("1st Image Shape: ", img.shape, '\n')
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
# Anaylze the single image shape after creating a new dataset
print("1st Image Shape after new dataset: ", img.shape, '\n')

# Predict the image using the fully trained model
predictions_single = model.predict(img)
# Anaylze the prediction matrix of the single image
print("1st Image prediction Matrix: ", predictions_single, '\n')

# plot the prediction of the single image for visualization
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

# Predict the label of the single image
prediction_result = np.argmax(predictions_single[0])
# Anaylze the prediction of the single image
print("1st Image Prediction Label", prediction_result, '\n')

