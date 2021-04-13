# class for storing and manipulating data.
import pandas as pd


class FMA:
    def __init__(self, path, subset='small', feature_fields=None):
        tracks = pd.read_csv(f"{path}tracks.csv", index_col=0, header=[0, 1])
        features = pd.read_csv(f"{path}features.csv", index_col=0, header=[0, 1, 2])
        if feature_fields is None:
            feature_fields = ['mfcc']
        self.tracks = tracks[tracks['set']['subset'] == subset]
        self.features = features.loc[self.tracks.index, feature_fields]  # already limiting to mfcc, so that random matrix gets initialised in correct dim
        self.training = None
        self.test = None
        self.validation = None
        self.trainingWithValidation = None

    def __get_split(self, criteria):
        subset_tracks = self.tracks[self.tracks['set']['split'] == criteria]
        features = self.features.loc[subset_tracks.index]
        labels = self.tracks.loc[subset_tracks.index, ('track', 'genre_top')]
        return features, labels

    def get_training_data(self):
        """
        Returns a tuple (features, labels) of training data
        """
        if self.training is None:
            self.training = self.__get_split("training")
        return self.training

    def get_test_data(self):
        """
        Returns a tuple (features, labels) of test data
        """
        if self.test is None:
            self.test = self.__get_split("test")
        return self.test

    def get_validation_data(self):
        """
        Returns a tuple (features, labels) of validation data
        """
        if self.validation is None:
            self.validation = self.__get_split("validation")
        return self.validation

    def get_training_with_validation_data(self):
        """
        Returns a tuple (features, labels) of training and validation data
        """
        if self.trainingWithValidation is None:
            features = pd.concat([self.get_training_data()[0], self.get_validation_data()[0]])
            labels = pd.concat([self.get_training_data()[1], self.get_validation_data()[1]])

            self.trainingWithValidation = features, labels

        return self.trainingWithValidation
