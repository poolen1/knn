
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

    def gather_categories(self):
        pass

    def vote(self):
        pass

    def classify_data(self, data):
        knn = self.get_neighbors()
        pass
