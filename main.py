import pandas as pd
from sklearn.decomposition import PCA
from knn import KNN
import matplotlib.pyplot as plt
import numpy as np

# Read in data from files
training_set = pd.read_csv("data/MNIST_training.csv")
test_set = pd.read_csv("data/MNIST_test.csv")

#  Function to project multi-D data onto 2D plane
def project_data(data):
    X = data.drop('label', axis=1)
    pca = PCA(n_components=2)
    pca.fit(X)
    PCAX = pca.transform(X)
    return PCAX


# Model training
# =================================

y_list = test_set.iloc[:, 0]
X_list = test_set.drop('label', axis=1)
y = np.array(y_list)
X = np.array(X_list)
test_data = (y, X)
ground_truth = y

training_set = np.array(training_set)

classifier = KNN(training_set, test_data, 1)
predictions = classifier.classify_data()
prediction_percent = classifier.evaluate(ground_truth, predictions)
