# class for storing and manipulating data.
import pandas as pd


class FMA:
    def __init__(self, path):
        tracks = pd.read_csv(f"{path}tracks.csv", index_col=0, header=[0, 1])
        features = pd.read_csv(f"{path}features.csv", index_col=0, header=[0, 1, 2])
        self.tracks = tracks[tracks['set']['subset'] == 'small']
        self.features = features.loc[self.tracks.index, 'mfcc'] # already limiting to mfcc, so that random matrix gets initialised in correct dim

    def __get_split(self, criteria):
        subset_tracks = self.tracks[self.tracks['set']['split'] == criteria]
        features = self.features.loc[subset_tracks.index]
        labels = self.tracks.loc[subset_tracks.index, ('track', 'genre_top')]
        print(f"{features.shape}", 'fma features shape', {criteria})
        return features, labels

    def get_training_data(self):
        """
        Returns a tuple (features, labels) of training data
        """
        return self.__get_split("training")

    def get_test_data(self):
        """
        Returns a tuple (features, labels) of test data
        """
        return self.__get_split("test")

    def get_validation_data(self):
        """
        Returns a tuple (features, labels) of validation data
        """
        return self.__get_split("validation")
