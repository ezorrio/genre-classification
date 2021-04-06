# class for performing LHS
import numpy as np
import math


class RandomHash:
    @staticmethod
    def random_matrix(rows, cols):
        random_matrix = np.random.rand(rows, cols)
        random_matrix[random_matrix < 1 / 6] = -1
        random_matrix[random_matrix > 5 / 6] = 1
        random_matrix[(1 / 6 < random_matrix) & (random_matrix < 5 / 6)] = 0
        return random_matrix

    def __init__(self, data_size, hash_length):
        self.random_matrix = self.random_matrix(data_size, hash_length)
        self.buckets = dict()

    def hash(self, data):
        return np.array2string((np.dot(self.random_matrix, data.T) > 0).astype('int'))

    def add(self, data, label):
        hash_value = self.hash(data)
        self.buckets[hash_value] = self.buckets.get(hash_value, list()) + [label]

    def get(self, data):
        hash_value = self.hash(data)
        return self.buckets.get(hash_value, [])


class LSH:
    def __init__(self, data_size, hashes_count=4, hash_length=8):
        self.hashes_count = hashes_count
        self.hash_length = hash_length
        self.data_size = data_size
        self.hashes = [RandomHash(data_size=data_size, hash_length=hash_length) for _ in range(self.hashes_count)]

    def hash_data(self, data):
        for random_hash in self.hashes:
            for features, label in data:
                print(features)
                random_hash.add(features, label)

    def get(self, features):
        return [random_hash.get(features) for random_hash in self.hashes]
