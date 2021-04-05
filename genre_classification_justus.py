import pandas as pd
import numpy as np



'''
Importing and Splitting the Data
'''

tracks = pd.read_csv('metadata/tracks.csv', index_col=0, header=[0, 1])
features = pd.read_csv('metadata/features.csv', index_col=0, header=[0, 1, 2])

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'
small = tracks['set', 'subset'] == 'small'

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]
y_val = tracks.loc[small & val, ('track', 'genre_top')]

X_train = features.loc[small & train, 'mfcc']
X_test = features.loc[small & test, 'mfcc']
X_val = features.loc[small & val, 'mfcc']


'''
Generate Random Matrix
'''

class HashTable:

    def __init__(self, hash_size, train_data):
        self.hash_size = hash_size
        self.train_data = train_data
        self.hash_table = dict()
        self.random_matrix = self.generate_R(self.hash_size, self.train_data.shape[1])
        self.generate_hash_table()


    def generate_R(self, rows, columns):
        """ Generates sparse zeros-matrix with 1/6-th having -sqrt(3) and
        1/6-th having sqrt(3). """
        
        R = np.zeros((rows*columns))
        
        # 1/6 of length of matrix
        j = int(np.round(rows*columns / 6))
        
        # overwrites 1/6-th of array with sqrt(3) and 1/6-th with -sqrt(3)
        R[:j] = np.sqrt(3); R[j:2*j] = -np.sqrt(3)
        np.random.shuffle(R)
        
        return np.reshape(R, (rows, columns))


    def generate_hash(self, input_data):
        projection = (np.dot(self.random_matrix, self.train_data.T) > 0).astype('int')
        hash_value = ''.join((projection).astype('str'))

        return hash_value

    
    def generate_hash_table(self):

        projection = (np.dot(self.random_matrix, self.train_data.T) > 0).astype('int')

        for i in range(projection.shape[1]):
            hash_value = ''.join((projection[:,i]).astype('str'))
            track_id = self.train_data.iloc[i].name

            if hash_value not in self.hash_table:
                self.hash_table[hash_value] = []
            self.hash_table[hash_value].append(track_id)


    def getitem(self, input_data):
        hash_value = self.generate_hash(input_data)
        return self.hash_table.get(hash_value, [])


class LSH:

    def __init__(self, num_tables, hash_size, training_data):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.training_data = training_data
        self.hash_tables = list()

        for _ in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.training_data))


    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


    def euclidean_similarity(self, vec1, vec2):
        return np.sqrt(np.sum((vec1 - vec2)**2))

    def most_common(self, List):
        return max(set(List), key = List.count)
    

    def similar(self, input_data):

        results = dict()
        for i in range(input_data.shape[0]):
            track_id = input_data.iloc[i].name
            results[track_id] = []

            for hash_table in self.hash_tables:
                results[track_id].extend(hash_table.getitem(input_data.iloc[i]))
            
            results[track_id] = set(results[track_id])

        return results


    def k_neighbors_approx(self, x_test, y_train, k=5, measure = 'Cosine'):
        """ 
        Courtesy of Christian
        Uses subset of results of find_similar to obtain the k-most similar tracks for each 
        track in test set. Returns a dictionary containing the track ids and the 
        predicted genre for the test set. 
        """
        
        similar = self.similar(x_test)
        results = dict()
        
        # Finds neighbors either based on cosine- or euclidean similarity measure
        if measure == 'Cosine':     

            # iterates trough test tracks
            for count, track_id in enumerate(similar):
                index1 = np.where(x_test.index == track_id)[0][0]
                vec1 = x_test.iloc[index1] # feature vector
                results[track_id] = []
                
                # iterates through random subset of similar training tracks for given validation track_id
                # in order to reduce complexity
                for similar_track in np.random.choice(similar[track_id], 800, replace=True):
                    index2 = np.where(self.training_data.index == similar_track)[0][0]
                    vec2 = self.training_data.iloc[index2] # feature vector
                    results[track_id].append([similar_track, self.cosine_similarity(vec1, vec2)])
                results[track_id] = sorted(results[track_id], key=lambda l:l[1], reverse=True)[:k]
            
                # Genre classification
                track_ids = [track_id[0] for track_id in results[track_id]]
                indices = [np.where(self.training_data.index == id)[0][0] for id in track_ids]
                genres = [y_train.iloc[index] for index in indices]
                
                if genres != []:
                    results[track_id] = self.most_common(genres)
                
                # Counter
                if (10*count/len(similar)) % 1 == 0 and (count/len(similar)) != 0:
                    print(f"{round(100*count/len(similar))} % Done")

        elif measure == 'Euclidean':
            
            # iterates trough test tracks
            for track_id in similar:
                index1 = np.where(x_test.index == track_id)[0][0]
                vec1 = x_test.iloc[index1]   
                results[track_id] = []
                
                # iterates through similar training tracks for given validation track_id
                for similar_track in np.random.choice(similar[track_id], 800, replace=True):
                    index2 = np.where(self.training_data.index == similar_track)[0][0]
                    vec2 = self.training_data.iloc[index2]
                    results[track_id].append([similar_track, self.euclidean_similarity(vec1, vec2)])
                results[track_id] = sorted(results[track_id], key=lambda l:l[1], reverse=False)[:k]
        
                # Genre classification            
                track_ids = [track_id[0] for track_id in results[track_id]]
                indices = [np.where(self.training_data.index == id)[0][0] for id in track_ids]
                genres = [y_train.iloc[index] for index in indices]
                
                if genres != []:
                    results[track_id] = self.most_common(genres)
        
        else:
            print("no valid similarity measure!!!")

        return results


    def __getitem__(self, input_data):
        results = list()
        for table in self.hash_tables:
            results.extend(table[input_data])

        return list(set(results))


l = 12
n = 5     
k = 5

measures = ['Cosine', 'Euclidean']
m = measures[0]

lsh = LSH(n, l, X_train)
results = lsh.k_neighbors_approx(X_val, y_train, k, m)

# Calculates the classification accuracy for validation set
u = 0
for result in results:
    id = np.where(y_val.index == result)[0][0]
    check = y_val.iloc[id]
    u += results[result] == check

print(f"Classification Accuracy of nearest neighbor search using validation set: {np.round(100 * u / len(y_val), 2)} %.")


x_train_new = pd.concat([X_train, X_val])
y_train_new = pd.concat([y_train, y_val])

lsh = LSH(x_train_new, n, l)
results = lsh.k_neighbors_approx(X_test, y_train_new)

u = 0
for result in results:
    id = np.where(y_test.index == result)[0][0]
    check = y_test.iloc[id]
    u += results[result] == check

print(f"Classification Accuracy of nearest neighbor search using test set: {np.round(100 * u / len(y_val), 2)} %.")