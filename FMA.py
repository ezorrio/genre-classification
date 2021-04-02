# class for storing and manipulating data.
import pandas as pd


class FMA:
    def __init__(self, path):
        tracks = pd.read_csv(f"{path}/tracks.csv", index_col=0, header=[0, 1])
        features = pd.read_csv(f"{path}/features.csv", index_col=0, header=[0, 1])
        self.tracks = tracks[tracks['set']['subset'] == 'small']
        self.features = features[features.index.isin(self.__get_track_ids(self.tracks))]

    @staticmethod
    def __get_track_ids(data):
        return data.index.values

    def __get_split(self, criteria):
        subset_tracks = self.tracks[self.tracks['set']['split'] == criteria]
        return subset_tracks, self.features[self.features.index.isin(self.__get_track_ids(subset_tracks))]

    def get_training_data(self):
        """
        Returns a tuple (tracks, features) of training data
        """
        return self.__get_split("training")

    def get_test_data(self):
        """
        Returns a tuple (tracks, features) of test data
        """
        return self.__get_split("test")

    def get_validation_data(self):
        """
        Returns a tuple (tracks, features) of validation data
        """
        return self.__get_split("validation")
