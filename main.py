import pandas as pd
import numpy as np
from knn import KNN

from datetime import datetime

# Read in data from files
# =================================
training_set = pd.read_csv("data/MNIST_training.csv")
test_set = pd.read_csv("data/MNIST_test.csv")

# Data Processing
# =================================

y_list = test_set.iloc[:, 0]
X_list = test_set.drop('label', axis=1)
y = np.array(y_list)
X = np.array(X_list)
test_data = (y, X)
ground_truth = y

# KNN Prediction
# =================================

start = datetime.now()
classifier = KNN(training_set, test_data, 3, 2)
predictions = classifier.classify_data()

# Prediction Accuracy
# =================================

prediction_percent = classifier.evaluate(ground_truth, predictions)
print("Accuracy: ", prediction_percent, "%")
print("Runtime: ", datetime.now() - start)

# best: k=3, p=2 (Euclidean distance), 82%
# Meta-parameter Analysis
# Loop through k in range (1,10) and p in range (1.0, 2.0)
# =================================
"""
metadata = []
for i in range(1, 11):
    for j in range(10, 21):
        k = i
        p = j * 0.1
        start = datetime.now()
        classifier = KNN(training_set, test_data, k, p)
        predictions = classifier.classify_data()
        prediction_percent = classifier.evaluate(ground_truth, predictions)
        stats = (k, p, prediction_percent)
        metadata.append(stats)
        print("K, P, Accuracy: ", k, p, prediction_percent, "%")
        print("Runtime: ", datetime.now() - start)

metadata.sort(key=lambda x: x[2], reverse=True)
print(metadata)
# print("Accuracy: ", prediction_percent, "%")
# print("Runtime: ", datetime.now() - start)
"""
