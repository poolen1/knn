import pandas as pd
from sklearn.decomposition import PCA
from knn import KNN
import matplotlib.pyplot as plt
import numpy

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
# Project data onto 2 dimensions
# y = training_set.iloc[:, 0]
# X = training_set.drop('label', axis=1)
# training_data = (y, X)
# training_PCA = project_data(training_set)
# training_data = (y, training_PCA)

y = test_set.iloc[:, 0]
X = training_set.drop('label', axis=1)
test_data = (y, X)
# test_PCA = project_data(test_set)
# test_data = (y, test_PCA)

classifier = KNN(training_set, test_data, 1)