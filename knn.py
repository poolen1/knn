import numpy as np


class KNN:
    def __init__(self, training, test, k):
        self.k = k
        self.training_data = training
        self.test_data = test
        self.p = 1

    def minkowski_distance(self, query):
        diffs = []  # List of (training_item, Minkowski_Val)
        the_sum = 0  # Sum of the difference of feature dimensions
        training_labels = self.training_data.iloc[:, 0]
        training_items = self.training_data.drop('label', axis=1)
        training_items = np.array(training_items)
        j = 0
        # for item in training_items:
        #     j += 1
        for i in range(len(query)):
            the_sum = sum(abs(query - training_items[i]) ** self.p)
            # print("the_sum: ", i, " ", the_sum)
            # the_sum += x
            mink_value = the_sum ** (1/float(self.p))
            # print("mink: ", i, " ", mink_value)
            # print("before: ", training_items[i][0])
            # print("label i: ", training_labels[i])
            # training_items[i] = np.insert(training_items[i], 0, training_labels[i])
            # print("After", training_items[i][0])
            diff = (mink_value, training_labels[i])
            # print(diff[1])
            diffs.append(diff)
        return diffs

    def get_neighbors(self, query):
        diffs = self.minkowski_distance(query)

        # print(diffs)
        diffs.sort(key=lambda x: x[0])
        nn = diffs[:self.k]
        # print("nn: ", nn[0][1])
        return nn

    @staticmethod
    def gather_categories():
        labels = []
        for i in range(10):
            label = (i, 0)
            labels.append(label)
        return labels

    def vote(self, knn):
        max_val = 0
        category = 0
        # Labels array: contains set of categories, and value that represents
        # number of NN with the same category
        labels = np.array(self.gather_categories())

        # Vote: iterate NN and increment their categories in the labels array
        for item in knn:
            for label in labels:
                # print("item: ", item)
                if item[1] == label[0]:
                    label[1] += 1

        for item in labels:
            if item[1] > max_val:
                max_val = item[1]
                category = item[0]
        return category

    def classify_data(self):
        categories = []
        i = 0
        for item in self.test_data[1]:
            i += 1
            knn = self.get_neighbors(item)
            categories.append(self.vote(knn))
        return categories

    @staticmethod
    def evaluate(ground_truth, predictions):
        total_correct = 0
        total_items = len(predictions)
        # print("ground: ", ground_truth)
        # print("predictions: ", predictions)
        for i in range(0, total_items):
            if ground_truth[i] == predictions[i]:
                total_correct += 1
        correct_percent = (total_correct/total_items) * 100
        return correct_percent
