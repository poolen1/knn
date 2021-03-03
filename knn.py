import numpy as np


# =================================
class KNN:
    def __init__(self, training, test, k, p):
        self.k = k  # Number of NN chosen
        self.training_data = training
        self.test_data = test
        self.p = p  # Minkowski distance parameter

    # Calculate distances of all training points
    # from query data point
    # =================================
    def minkowski_distance(self, query):
        diffs = []  # List of (training_item, Minkowski_Val)
        training_labels = self.training_data.iloc[:, 0]
        training_items = np.array(self.training_data.drop('label', axis=1))
        for i in range(len(query)):
            the_sum = sum(abs(query - training_items[i]) ** self.p)
            mink_value = the_sum ** (1/float(self.p))
            # Distance tuple contains distance value
            # and the label of the associated training data point
            diff = (mink_value, training_labels[i])
            diffs.append(diff)
        return diffs

    # Calculate distances, sort and slice
    # distances array to return k (distance, label) tuples
    # =================================
    def get_neighbors(self, query):
        diffs = self.minkowski_distance(query)
        diffs.sort(key=lambda x: x[0])
        nn = diffs[:self.k]
        return nn

    # Instantiate ballot with label tuple (label, number_of_votes)
    # Number of votes initialized with 0
    # =================================
    @staticmethod
    def gather_categories():
        labels = []
        for i in range(10):
            label = (i, 0)
            labels.append(label)
        return labels

    # Simple majority of query's KNN's label values
    # creates query's predicted label
    # =================================
    def vote(self, knn):
        max_val = 0
        category = 0
        # Labels array: contains set of categories,
        # and value that represents
        # number of NN with the same category
        labels = np.array(self.gather_categories())

        # Vote: iterate NN and increment their
        # categories in the labels array
        for item in knn:
            for label in labels:
                if item[1] == label[0]:
                    label[1] += 1

        # Iterate through labels after
        # polls have closed to find
        # chosen prediction
        for item in labels:
            if item[1] > max_val:
                max_val = item[1]
                category = item[0]

        return category

    # Iterate test data rows to get
    # label predictions for each item
    # =================================
    def classify_data(self):
        categories = []
        for item in self.test_data[1]:
            knn = self.get_neighbors(item)
            categories.append(self.vote(knn))
        return categories

    # Compare array of predicted labels
    # with ground truth labels,
    # output percent accuracy
    # =================================
    @staticmethod
    def evaluate(ground_truth, predictions):
        total_correct = 0
        total_items = len(predictions)
        for i in range(0, total_items):
            if ground_truth[i] == predictions[i]:
                total_correct += 1
        correct_percent = (total_correct/total_items) * 100
        return correct_percent
