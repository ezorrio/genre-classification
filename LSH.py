# class for performing LHS
import numpy as np
import math


class RandomHash:
    @staticmethod
    def construction_method(number):
        value = 0
        if number < 1 / 6:
            value = -1
        elif number > 5 / 6:
            value = 1
        return math.sqrt(3) * value

    @staticmethod
    def random_matrix(shape):
        random_matrix = np.random.rand(shape)
        return map(RandomHash.construction_method, random_matrix)

    def __init__(self, data_size, hash_length):
        self.random_matrix = self.random_matrix((hash_length, data_size))
        self.buckets = dict()

    def hash(self, data):
        return np.array2string((np.dot(data, self.random_matrix.T) > 0).astype('int'))

    def __setitem__(self, data, label):
        hash_value = self.hash(data)
        self.buckets[hash_value] = self.buckets.get(hash_value, list()) + [label]

    def __getitem__(self, data):
        hash_value = self.hash(data)
        return self.buckets.get(hash_value, [])


class LSH:
    def __init__(self):
        return
