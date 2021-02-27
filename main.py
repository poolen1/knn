import pandas as pd
import numpy as np
from knn import KNN

# Read in data from files
training_set = pd.read_csv("data/MNIST_training.csv")
test_set = pd.read_csv("data/MNIST_test.csv")

# Model training
# =================================

y_list = test_set.iloc[:, 0]
X_list = test_set.drop('label', axis=1)
y = np.array(y_list)
X = np.array(X_list)
test_data = (y, X)
ground_truth = y

# training_set = np.array(training_set)

classifier = KNN(training_set, test_data, 10)
predictions = classifier.classify_data()
prediction_percent = classifier.evaluate(ground_truth, predictions)

print("Accuracy: ", prediction_percent, "%")
