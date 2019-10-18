# Python tutorial using Keras with tensorflow backend support for a multi-hidden layer Artificial Deep Neural Network (DNN) for regression on the auto mpg dataset.
# Deep learning is an artificial intelligence function that imitates the workings of the human brain in processing data and creating patterns for use in decision making.
# Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled.
# Keras is an open-source high-level neural-network library written in Python.

# Import python libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import the auto mpg dataset from the UCI machine learning repository
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# Confirm the dataset was imported correctly 
#print(dataset_path)

# Create the column names for the conversion of the dataset to pandas format
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
# Create the pandas dataframe with the auto mpg dataset
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
# Copy the recently created dataset and rename it
dataset = raw_dataset.copy()

# Analyze the whole auto mpg panda dataframe
#print(dataset)
# Analyze the head of the auto mpg panda dataframe
#print(dataset.head())
# Analyze the tail of the auto mpg panda dataframe
#print(dataset.tail())
# Analyze the whole dataframe and count how many instances have unknown data
#print(dataset.isna().sum())
# Drop the rows in the dataframe that contain unknown data
dataset = dataset.dropna()
# Analyze the whole dataframe and confirm no rows with unknown data remain
#print(dataset.isna().sum())

# Pull the column that contains the Origin label
origin = dataset.pop('Origin')
# Analyze the Origin column
#print(origin)
# Convert each entry in the origin column from categorical to numeric by one-hot encoding
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
# Analyze the tail of the auto mpg dataframe
#print(dataset.tail())

# Split the whole dataset into seperate training and testing datasets for the regression ANN model
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Analyze the distribution plots of the training dataset
#sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
#plt.show()

# Analyze the overall statistics of the training dataset
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
#print(train_stats)

# Separate the target label from the feature dataset
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize the training and testing dataset
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
# Analyze the normalized training and testing datasets
#print(normed_train_data)
#print(normed_test_data)

# Define the build_model function for the ANN model
def build_model():
  model = keras.Sequential([layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]), layers.Dense(64, activation=tf.nn.relu), layers.Dense(1)])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])
  return model
# Create the ANN model using the build_model function
model = build_model()
#model.summary()

# Test the trained model with 10 samples of the testing data
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
#print(example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
EPOCHS = 1000

# Analyze the model's training progress
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())

# Analyze the mean abs error and the mean square error
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()
#plot_history(history)

# Analyze the model with the test dataset and predict an outcome
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("\nTesting set Mean Abs Error: {:5.2f} MPG".format(mae))

# Predict the MPG values using the testing dataset
test_predictions = model.predict(normed_test_data).flatten()
'''
# Analyze the MPG values
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()
'''
'''
# Analyze the predicted error
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()
'''


