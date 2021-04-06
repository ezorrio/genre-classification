from FMA import *
from LSH import *


class MusicSearch:
    def __init__(self, data_path):
        self.data = FMA(data_path)
        self.lsh = LSH(self.data.features.shape[1])

    def train(self):
        for item in self.data.get_training_data():
            self.lsh.hash_data(item)

    '''
    Accepts track_id. Should extract features of it and return predicted genre based on LSH
    '''
    def predict(self, track_id):
        # find the track. Pick
        return

    '''
    Takes test data subset, compares with trained data and outputs accuracy and other metrics    
    '''
    def test(self):
        training = self.data.get_training_data()
        test = self.data.get_test_data()
        validation = self.data.get_validation_data()
        return
