# %%
from FMA import *
from LSH import *
from tqdm import tqdm

class MusicSearch:
    def __init__(self, data_path, n, l, subset='small', feature_fields=None, measure='Cosine', k=5, magic_number=800):
        if feature_fields is None:
            feature_fields = ['mfcc']
        self.data = FMA(data_path, feature_fields=feature_fields, subset=subset)
        self.lsh = LSH(self.data.features.shape[1], n, l)
        # holds a reference to a set from FMA. For internal usage only
        self._training_set = None
        self._test_set = None
        self._measure = measure
        self._k = k
        self._magic_number = magic_number

    def train(self):
        self._training_set = self.data.get_training_data()
        for item in self._training_set:
            self.lsh.hash_data(item)

    def test(self):
        self._test_set = self.data.get_test_data()
        self.print_classification_results(self._test_set)

    def train_with_validation(self):
        self._training_set = self.data.get_training_with_validation_data()
        for item in self._training_set:
            self.lsh.hash_data(item)

    def test_with_validation(self):
        self._test_set = self.data.get_validation_data()
        self.print_classification_results(self._test_set)

    def find_similar_tracks(self, feature):
        """ takes a feature vector which is hashed in every hash table 
         and returns track_ids of similar tracks """ 
        result = set()
        for hash_table in self.lsh.hashes:
            result.update(hash_table.get(feature))

        return list(result)

    def calculate_similarity(self, feature, track_id):
        index = np.where(self._training_set[0].index == track_id)[0][0]
        training_feature = self._training_set[0].iloc[index]

        if self._measure == "Cosine":
            return self.cosine_similarity(feature, training_feature)

        elif self._measure == "Euclidean":
            return self.euclidean_similarity(feature, training_feature)

        else:
            raise Exception("Invalid similarity measure.\n")

    def k_neighbors(self, feature):
        """ Returns list of track_ids of knn for given feature vector. 
            self._magic_number refers to the size of the random subset of similar tracks, 
            needed for the approximation of the knn problem. 
            """
        similar_tracks = self.find_similar_tracks(feature)

        k_neighbors = []
        if not similar_tracks:
            return k_neighbors

        for track_id in np.random.choice(similar_tracks, np.min(self._magic_number, len(similar_tracks)),
                                         replace=True):  # random subset of similar tracks
            k_neighbors.append((track_id, self.calculate_similarity(feature, track_id)))

        ''' (track_id, similarity)-pairs are sorted via the similarity and only
            k-most similar tracks are returned '''
            
        if self._measure == "Cosine":  # ideally 1 --> sorted descending
            k_neighbors = sorted(k_neighbors, key=lambda l: l[1], reverse=True)[:self._k]

        elif self._measure == "Euclidean":  # ideally 0 --> sorted ascending
            k_neighbors = sorted(k_neighbors, key=lambda l: l[1], reverse=False)[:self._k]

        k_neighbors = [neighbor[0] for neighbor in k_neighbors]  # only return the track_ids

        return k_neighbors

    def predict_genre(self, feature):
        """ predicts genre for given feature vector """
        
        k_neighbors = self.k_neighbors(feature)
        indices = [np.where(self._training_set[0].index == track_id)[0][0] for track_id in k_neighbors]
        genres_of_k_neighbors = [self._training_set[1].iloc[index] for index in indices]

        if genres_of_k_neighbors:
            return self.most_common(genres_of_k_neighbors)
        else:
            print("No similar tracks found.")
            return

    def classification_score(self, test):
        """ Returns a dictionary containing the absolute number of correct 
            predictions per genre """
        scores_per_genres = {}

        for track_id, feature in tqdm(test[0].iterrows(), total=test[0].shape[0]):
            predicted_genre = self.predict_genre(feature)
            id = np.where(test[0].index == track_id)[0][0]
            true_genre = test[1].iloc[id]

            # Creates/calculates the dic-entries
            if true_genre == predicted_genre:
                if true_genre not in scores_per_genres:
                    scores_per_genres[true_genre] = 1
                else:
                    scores_per_genres[true_genre] += 1

        return scores_per_genres

    def print_classification_results(self, test):
        scores_per_genres = self.classification_score(test)

        print('Classification Accuracy per genre:\n')

        for genre_score in scores_per_genres:
            ''' for FMA "small" dataset the absolute number of correct prediction 
                equals the percentage values since there are 100 songs per genre.'''
            print(f'{genre_score}: {scores_per_genres[genre_score]}%')

        overall_score = np.average([scores_per_genres[count] for count in scores_per_genres])
        print('-----------------------------------------')
        print(f'Overall classification accuracy: {overall_score}%')

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def euclidean_similarity(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def most_common(collection):
        return max(set(collection), key=collection.count)

# %%
