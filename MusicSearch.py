#%%
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
        
        self.print_classification_results(training, validation)
        
        return

    """ Added by Chris on 09.04 """

    def find_similar_tracks(self, feature):
                
        result = set()
        for hash_table in self.lsh.hashes:
            result.update(hash_table.get(feature))

        return list(result)
        
    def calculate_similarity(self, training, feature, track_id, measure="Cosine"):
        
        index = np.where(training[0].index == track_id)[0][0]
        training_feature = training[0].iloc[index]
        
        if measure == "Cosine":
            return self.cosine_similarity(feature, training_feature)
        
        elif measure == "Euclidean":
            return self.euclidean_similarity(feature, training_feature)
        
        else:
            print("Invalid similarity measure.\n")
            return
        
    def k_neighbors(self, training, feature, measure='Cosine', k=5,):
        # returns list of track_ids of knn
        
        similar_tracks = self.find_similar_tracks(feature)

        k_neighbors = []
        for track_id in np.random.choice(similar_tracks, 800, replace=True): # random subset of similar tracks
            k_neighbors.append((track_id, self.calculate_similarity(training, feature, track_id, measure)))
        
        if measure == "Cosine": # ideally 1 --> sorted descending
            k_neighbors = sorted(k_neighbors, key=lambda l:l[1], reverse=True)[:k]
        
        elif measure == "Euclidean": # ideally 0 --> sorted ascending
            k_neighbors = sorted(k_neighbors, key=lambda l:l[1], reverse=False)[:k]
        
        k_neighbors = [neighbor[0] for neighbor in k_neighbors] # only return the track_ids

        return k_neighbors

    def predict_genre(self, training, feature):
        # predicts genre for given feature vector
        
        k_neighbors = self.k_neighbors(training, feature)
        indices = [np.where(training[0].index == track_id)[0][0] for track_id in k_neighbors]
        genres_of_k_neighbors = [training[1].iloc[index] for index in indices] 
    
        if genres_of_k_neighbors != []:
            return self.most_common(genres_of_k_neighbors)
        else:
            print("No similar tracks found.")      
            return
    
    def classification_score_per_genre(self, training, test):

        scores_per_genres = {'Hip-Hop' : 0, 'Pop' : 0, 'Folk' : 0, 'Rock' : 0, 'Experimental' : 0,
                'International' : 0, 'Electronic' : 0, 'Instrumental' : 0}
        
        for track_id, feature in test[0].iterrows():
            
            predicted_genre = self.predict_genre(training, feature)
            id = np.where(test[0].index == track_id)[0][0]
            true_genre = test[1].iloc[id]

            if true_genre == predicted_genre:
                scores_per_genres[true_genre] += 1
            
        return scores_per_genres
            
    def print_classification_results(self, training, test):
             
        scores_per_genres = self.classification_score_per_genre(training, test)

        print('Classification Accuracy per genre:\n')

        for genre_score in scores_per_genres:    
            print(f'{genre_score}: {scores_per_genres[genre_score]}%')        

        overall_score = np.average([scores_per_genres[count] for count in scores_per_genres])
        print('-----------------------------------------')
        print(f'Overall classification accuracy: {overall_score}%')

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def euclidean_similarity(self, vec1, vec2):
        return np.linalg.norm(vec1-vec2)

    def most_common(self, List):
        return max(set(List), key = List.count)

# %%
