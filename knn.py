
class KNN:
    def __init__(self, training, test, k):
        self.K = k
        self.training_data = training
        self.test_data = test
        self.p = 2

    def minkowski_distance(self, query):
        diffs = []  # List of (training_item, Minkowski_Val)
        the_sum = 0  # Sum of the difference of feature dimensions
        x = 0  # Difference btw query and example item
        mink_value = 0
        for item in self.training_data:
            for i in range(0, len(item)):
                x = abs(query[i] - item[i]) ** self.p
                the_sum += x
            mink_value = the_sum ** (1/self.p)
            diffs.append(item, mink_value)
        return diffs

    def get_neighbors(self, query):
        nn = [] * self.k
        diffs = self.minkowski_distance(query)
        diffs.sort(reverse=True)
        nn = diffs[0:self.k]
        return nn

    @staticmethod
    def gather_categories():
        labels = {}*10
        i = 0
        for label in labels:
            label = {i, 0}
        return labels

    def vote(self, knn):
        max_val = 0
        category = 0
        labels = self.gather_categories
        for item in knn:
            if knn[0] in item[0]:
                labels[knn[0]] += 1
        for item in labels:
            if item[1] > max_val:
                max_val = item[1]
                category = item[0]
        return category

    def classify_data(self, data):
        categories = []
        for item in self.test_data:
            knn = self.get_neighbors()
            categories.append(self.vote(knn))
        pass

    def evaluate(self):
        pass
