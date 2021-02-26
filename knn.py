import numpy as np


class KNN:
    def __init__(self, training, test, k):
        self.k = k
        self.training_data = training
        self.test_data = test
        self.p = 2

    def minkowski_distance(self, query):
        diffs = []  # List of (training_item, Minkowski_Val)
        the_sum = 0  # Sum of the difference of feature dimensions
        for item in self.training_data:
            for i in range(0, len(query)):
                x = abs(query[i] - item[i]) ** self.p
                the_sum += x
            mink_value = the_sum ** (1/self.p)
            diff = (mink_value, item)
            diffs.append(diff)
        return diffs

    def get_neighbors(self, query):
        diffs = self.minkowski_distance(query)
        diffs.sort()
        nn = diffs[:self.k]
        # print(len(nn))
        return nn

    @staticmethod
    def gather_categories():
        labels = []
        for i in range(10):
            label = (i, 0)
            labels.append(label)
        # print(labels)
        return labels

    def vote(self, knn):
        max_val = 0
        category = 0
        labels = np.array(self.gather_categories())

        # print(knn)

        for item in knn:
            for label in labels:
                print(item[1][0], label[0])
                if item[1][0].any() == label[0]:
                    labels[item[1][0]] += 1
                # print(labels[item[1]])
            # if knn in item[0]:
            #    labels[knn[0]] += 1
        print(labels)

        for item in labels:
            if item[1] > max_val:
                max_val = item[1]
                category = item[0]
        return category

    def classify_data(self):
        categories = []
        # print(self.test_data[1])
        for item in self.test_data[1]:
            # print(item)
            knn = self.get_neighbors(item)
            categories.append(self.vote(knn))
        return categories

    def evaluate(self, ground_truth, predictions):
        total_correct = 0
        total_items = len(predictions)
        for i in range(0, total_items):
            if ground_truth[i] == predictions[i]:
                total_correct += 1
        correct_percent = (total_correct/total_items) * 100
        return correct_percent
