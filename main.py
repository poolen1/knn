import pandas as pd
import numpy

# f = open("data/MNIST_training.csv", "r")
training_set = pd.read_csv("data/MNIST_training.csv")
test_set = pd.read_csv("data/MNIST_test.csv")

print(training_set)