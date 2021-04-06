from FMA import *
from LSH import *


class MusicSearch:
    def __init__(self, data_path):
        self.data = FMA(data_path)
        self.lsh = LSH(self.data.features.shape[1])

    def train(self):
        for item in self.data.get_training_data():
            self.lsh.hash_data(item)

    def test(self):
        return
