# Python tutorial using scikit-learn with a multi-hidden layer Artificial Deep Neural Network (DNN) on the iris dataset
# Deep learning is an artificial intelligence function that imitates the workings of the human brain in processing data and creating patterns for use in decision making.
# Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled.

# Import python libraries
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the iris dataset
dataSet = datasets.load_iris()

# Analyze the feature set of the data
#print("features: ", dataSet.feature_names, "\n")

# Analyze the target set of the data
#print("Labels: ", dataSet.target_names, "\n")

# Analyze the dataset's shape 
#print("Data's Shape: ", dataSet.data.shape, "\n")

# Analyze the first five entires of the dataset's values
#print("Data's First Five Values: ", dataSet.data[0:5], "\n")

# Analyze the target set of the data
#print("Data's Target Values: ", dataSet.target, "\n")

# Split the whole dataset into a seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.20)

# Create a Scaler to transform data into standardization form for better classication results
theScaler = StandardScaler()
# Train the Scaler to the X_train dataset
theScaler.fit(X_train)

# Transform the X_train dataset with the Scaler
X_train = theScaler.transform(X_train)
# Transform the X_test dataset with the Scaler
X_test = theScaler.transform(X_test)

# Create the DNN Model with multi-hidden layer and 1000 iterations and default learning rate
theModel = MLPClassifier(activation='relu', hidden_layer_sizes=(5, 7, 10, 15, 5), max_iter=1000)
# Train the newly created DNN Model using both X_Train and Y_Train datasets
theModel.fit(X_train, y_train)

# Predict the testing dataset using the recently trained DNN model
theModel_Predict = theModel.predict(X_test)

# Analyze the DNN model's Accuracy, confusion_matrix, and classification report
print("The Model's Accuracy: ", accuracy_score(y_test, theModel_Predict), "\n")
print("The Model's Confusion Matrix: \n", confusion_matrix(y_test, theModel_Predict), "\n")
print("The Model's Classicication Report: \n", classification_report(y_test, theModel_Predict), "\n")

# Analyze the DNN model's weight matrices and Bias vectors per index of layer
#print("The Models Weight Matrices for Input Layer: \n", theModel.coefs_[0], "\n")
#print("The Models Bias Vectors for Input Layer: \n", theModel.intercepts_[0], "\n")

#print("The Models Weight Matrices for Hidden Layer (0): \n", theModel.coefs_[1], "\n")
#print("The Models Bias Vectors for Hidden Layer (0): \n", theModel.intercepts_[1], "\n")

#print("The Models Weight Matrices for Hidden Layer (1): \n", theModel.coefs_[2], "\n")
#print("The Models Bias Vectors for Hidden Layer (1): \n", theModel.intercepts_[2], "\n")

#print("The Models Weight Matrices for Hidden Layer (2): \n", theModel.coefs_[3], "\n")
#print("The Models Bias Vectors for Hidden Layer (2): \n", theModel.intercepts_[3], "\n")

#print("The Models Weight Matrices for Hidden Layer (3): \n", theModel.coefs_[4], "\n")
#print("The Models Bias Vectors for Hidden Layer (3): \n", theModel.intercepts_[4], "\n")

#print("The Models Weight Matrices for Output Layer: \n", theModel.coefs_[5], "\n")
#print("The Models Bias Vectors for Output Layer: \n", theModel.intercepts_[5], "\n")

