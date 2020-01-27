# Python tutorial using keras with tensorflow backend support for a multi-hidden layer Artificial Deep Neural Network (DNN) on the pima indians diabetes dataset.
# Deep learning is an artificial intelligence function that imitates the workings of the human brain in processing data and creating patterns for use in decision making.
# Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled.
# Python is an interpreted, high-level, general-purpose programming language.
# Keras is an open-source high-level neural-network library written in Python.
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# Import python libraries
from numpy import loadtxt
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import os

# Import the Pima Indians diabetes dataset
theData = loadtxt('pima-indians-diabetes.csv', delimiter=',')
#print("Data's full values: ", theData, '\n')

# Split the Dataset into the X (Features) and Y Datasets (Labels)
X_dataset = theData[:,0:8]
Y_dataset = theData[:,8]
#print("The X Dataset: ", X_dataset, '\n')
#print("The Y Dataset: ", Y_dataset, '\n')

# Create the sequential DNN model
theModel = Sequential()
# Create the 1st DNN Hidden Layer with 12 nodes (8 Input Layer nodes): uniform (kernel_initalizer) and RELU (Activation Function)
theModel.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
# Create the 2nd DNN Hidden Layer with 8 nodes: uniform (kernel_initalizer) and RELU (Activation Function)
theModel.add(Dense(8, kernel_initializer='uniform', activation='relu'))
# Create the 3rd DNN Hidden Layer with 4 nodes: uniform (kernel_initalizer) and RELU (Activation Function)
theModel.add(Dense(4, kernel_initializer='uniform', activation='relu'))
# Create the 4nd DNN Hidden Layer with 1 node: uniform (kernel_initalizer) and Sigmoid (Activation Function)
theModel.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Plot the sequential DNN model
#plot_model(theModel, to_file='theModel_plot.png', show_shapes=True, show_layer_names=True)
# Get a summary of the sequential DNN model
print(theModel.summary())

# Compile the DNN model using binary crossentropy loss function, adam optimizer function, and set the metrics to accuracy
theModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the DNN model with 150 epochs, 10 batch size, and 0.33 test dataset size
#theModel.fit(X_dataset, Y_dataset, epochs=150, batch_size=10, verbose=0)
history = theModel.fit(X_dataset, Y_dataset, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# Predict the labels using the X dataset
thePredictions = theModel.predict_classes(X_dataset)

# Evaluate the trained model using the X and Y dataset
_, accuracy = theModel.evaluate(X_dataset, Y_dataset, verbose=0)
# Analyze the trained model's accuracy
print('Accuracy: %.2f' % (accuracy*100), '\n')

# Analyze the first five dataset's labels compaired to the trained model's predictions
for i in range(5):
	print("%s => %d (expected %d)" % (X_dataset[i].tolist(), thePredictions[i], Y_dataset[i]), '\n')

# list all data used in history
print(history.history.keys())

# Plot the model's accuracy for visualization
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# Plot the model's loss cost for visualization
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

'''
# save model and architecture to a single file
theModel.save("theModel.h5")
print("Saved theModel to disk")

# load the saved model
theModel = load_model('theModel.h5')
'''

'''
# serialize model to JSON
model_json = theModel.to_json()
with open("theModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
theModel.save_weights("theModel.h5")
print("Saved theModel to disk")
 
# load json and create model
json_file = open('theModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("theModel.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_dataset, Y_dataset, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''

'''
# serialize model to YAML
model_yaml = theModel.to_yaml()
with open("theModel.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
theModel.save_weights("theModel.h5")
print("Saved theModel to disk")
 
# load YAML and create model
yaml_file = open('theModel.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("theModel.h5")
print("Loaded theModel from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_dataset, Y_dataset, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''

