# -*- coding: utf-8 -*-
"""
Assignment 1: LSH (Local Sensitive Hashing) using FMA database
"""

# %% 
import numpy as np
import matplotlib.pyplot as plt # %matplotlib qt for plots in separate window
import pandas as pd
import datetime

# ---------------------- Functions -----------------------

def generate_R(rows, columns):
    """ Generates sparse zeros-matrix with 1/6-th having sqrt(3) and
    1/6-th having sqrt(3). """
    
    R = np.zeros((rows*columns))
    
    # 1/6 of length of matrix
    j = int(np.round(rows*columns / 6))
    
    # overwrites 1/6-th of array with sqrt(3) and 1/6-th with -sqrt(3)
    R[:j] = np.sqrt(3); R[j:2*j] = -np.sqrt(3)
    np.random.shuffle(R); np.random.shuffle(R)
    
    return np.reshape(R, (rows, columns))

# similarities
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def euclidean_similarity(vec1, vec2):
    return np.linalg.norm(vec1-vec2)

def most_common(List):
    return max(set(List), key = List.count)
# ------------------------------------------------------

# ---------------------- Classes -----------------------

class HashTable:
    """ class for one single hash table """
    
    def __init__(self, l, train):
        self.l = l # hash length
        self.train = train
        self.hash_table = dict()
        self.R = generate_R(self.l, self.train.shape[1])
        self.generate_hash_table()

    def generate_hash_table(self):
        """ Builds dictionary containing all unique hash values as keys and list of
        all track ids that share the same hash value, as the dictionary values. """
        
        projection = (np.dot(self.R, self.train.T) > 0).astype('int')
        
        for index in range(projection.shape[1]):
            hash_value = ''.join((projection[:,index]).astype('str'))
            track_id = self.train.iloc[index].name
    
            if hash_value not in self.hash_table:
                self.hash_table[hash_value] = []
            self.hash_table[hash_value].append(track_id)
    
    def generate_hash(self, inp_vec):
        
        projection = (np.dot(self.R, inp_vec.T) > 0).astype('int')        
        hash_value = ''.join((projection).astype('str'))
        
        return hash_value
    
    def query(self, inp_vec):
        """ Creates hash value for given input vector, searches hash table for
            the given hash value and returns the corresponding track ids. """
        
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, list()) # returns [] if hash value not in hash table
    
class LSH:
    """ Class for finding similar tracks and computing k neighbors to find 
        predicted genre. """
    
    def __init__(self, train, n=5, l=10, k=5):
        self.n = n # number of hash tables
        self.l = l # hash length
        self.k = k # number of neighbors considered
        self.train = train
        
        # Builds list containing n hash tables
        self.hash_tables = [HashTable(self.l, self.train) for _ in range(self.n)]
        
    def find_similar(self, test):
        """ Loops over every track in the test set and finds tracks of training set 
            with the same hash value in at least one hash table. """

        results = dict()
        for index in range(test.shape[0]):
            track_id = test.iloc[index].name
            results[track_id] = []

            for hash_table in self.hash_tables:
                results[track_id].extend(hash_table.query(test.iloc[index]))
            results[track_id] = set(results[track_id]) # duplicates!
        
        return results

    def k_neighbors(self, test, k, measure = 'Cosine'):
            """ Uses results of find_similar to obtain the k-most similar tracks for each 
                track in test set. Returns a dictionary containing the track ids and the 
                predicted genre for the test set. """
            
            similar = self.find_similar(test)
            results = dict()
            
            # Finds neighbors either based on cosine- or euclidean similarity measure
            if measure == 'Cosine':     

                # iterates trough test tracks
                for val_track in similar:
                    index1 = np.where(test.index == val_track)[0][0]
                    vec1 = test.iloc[index1] # feature vector
                    results[val_track] = []
                    print(val_track)
                    
                    # iterates through similar training tracks for given validation track
                    for similar_track in similar[val_track]:
                        index2 = np.where(self.train.index == similar_track)[0][0]
                        vec2 = self.train.iloc[index2] # feature vector
                        results[val_track].append([similar_track, cosine_similarity(vec1, vec2)])
                    results[val_track] = sorted(results[val_track], key=lambda l:l[1], reverse=True)[:k]
                
                    track_ids = [track[0] for track in results[val_track]]
                    indices = [np.where(self.train.index == id)[0][0] for id in track_ids]
                    genres = [y_train.iloc[index] for index in indices]
                    
                    if genres != []:
                        results[val_track] = most_common(genres)
                        print(results[val_track])
        
            elif measure == 'Euclidean':
                
                # iterates trough test tracks
                for val_track in similar:
                    index1 = np.where(test.index == val_track)[0][0]
                    vec1 = test.iloc[index1]   
                    results[val_track] = []
                    
                    # iterates through similar training tracks for given validation track
                    for similar_track in similar[val_track]:
                        index2 = np.where(self.train.index == similar_track)[0][0]
                        vec2 = self.train.iloc[index2]
                        results[val_track].append([similar_track, euclidean_similarity(vec1, vec2)])
                    results[val_track] = sorted(results[val_track], key=lambda l:l[1], reverse=False)[:k]
            
                    track_ids = [track[0] for track in results[val_track]]
                    indices = [np.where(self.train.index == id)[0][0] for id in track_ids]
                    genres = [y_train.iloc[index] for index in indices]
                    
                    if genres != []:
                        results[val_track] = most_common(genres)
            
            else:
                print("no valid similarity measure!!!")

            return results

#%%
# -------------- Feature & Track - extraction --------------------
tracks = pd.read_csv('metadata/tracks.csv', index_col=0, header=[0, 1])

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'
small = tracks['set', 'subset'] == 'small'

features = pd.read_csv('metadata/features.csv', index_col=0, header=[0, 1, 2])

x_train = features.loc[small & train, 'mfcc']
x_test = features.loc[small & test, 'mfcc']
x_val = features.loc[small & val, 'mfcc']

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]
y_val = tracks.loc[small & val, ('track', 'genre_top')] 
        
#%%%%
# ---------------- Hyperparameters & Computation --------------
l = 10
n = 3     
k = 5
measures = ['Cosine', 'Euclidean']
m = measures[0]

start = datetime.datetime.now()

lsh = LSH(x_train, n, l, k)
results = lsh.k_neighbors(x_val, k, m)

runtime = (datetime.datetime.now() - start).total_seconds()  
print(f'\nRuntime: {runtime:.2f} seconds using {m}-similarity.\n')

# Calculates the classification accuracy 
u = 0
for result in results:
    id = np.where(y_val.index == result)[0][0]
    check = y_val.iloc[id]
    u += results[result] == check

print(f"Classification Accuracy of nearest neighbor search: {np.round(100 * u / len(y_val), 2)} %.")


# %%