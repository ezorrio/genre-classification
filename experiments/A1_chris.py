# -*- coding: utf-8 -*-
"""
Assignment 1: LSH (Local Sensitive Hashing) using FMA database
"""

# %% 
import numpy as np
import matplotlib.pyplot as plt # %matplotlib qt for plots in separate window
import pandas as pd
import datetime

# ------------------ Functions -------------------------------------

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

# -----------------------------------------------------------------

#%% Feature & Track - extraction
tracks = pd.read_csv('../metadata/tracks.csv', index_col=0, header=[0, 1])

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'
small = tracks['set', 'subset'] == 'small'

features = pd.read_csv('../metadata/features.csv', index_col=0, header=[0, 1, 2])

x_train = features.loc[small & train, 'mfcc']
x_test = features.loc[small & test, 'mfcc']
x_val = features.loc[small & val, 'mfcc']

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]
y_val = tracks.loc[small & val, ('track', 'genre_top')]

#%% Classes for hash table and LHS

class HashTable:
    """ class for one single hash table """
    
    def __init__(self, l, train):
        self.l = l # hash length
        self.train = train
        self.hash_table = dict()
        R = generate_R(self.l, self.train.shape[1])
        p = (np.dot(R, self.train.T) > 0).astype('int')
        self.generate_hash_table(p)

    def generate_hash_table(self, p):
        """ Builds dictionary containing all unique hash values as keys and list of
        all track ids that share the same hash value, as the dictionary values. """
        
        for index in range(p.shape[1]):
            hash_value = ''.join((p[:,index]).astype('str'))
            track_id = self.train.iloc[index].name
    
            if hash_value not in self.hash_table:
                self.hash_table[hash_value] = []
            self.hash_table[hash_value].append(track_id)
            
    def query(self, hash):
        """ Searches the hash table for the given hash value and returns
            the corresponding track ids. """
            
        if hash in self.hash_table:
            return self.hash_table[hash]
        else:
            return []  

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
        
        # Generate hashes for test set
        R = generate_R(self.l, test.shape[1])
        p = (np.dot(R, test.T) > 0).astype('int')

        results = dict()
        for index in range(p.shape[1]):
            hash_value = ''.join((p[:,index]).astype('str'))
            track_id = test.iloc[index].name
            results[track_id] = []
            
            # Searches every hash table for given hash value and adds results to dictionary
            for hash_table in self.hash_tables:
                results[track_id].extend(hash_table.query(hash_value))
        
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
            
                # iterates through similar training tracks for given validation track
                for similar_track in similar[val_track]:
                    index2 = np.where(self.train.index == similar_track)[0][0]
                    vec2 = self.train.iloc[index2] # feature vector
                    results[val_track].append([similar_track, cosine_similarity(vec1, vec2)])
                results[val_track] = sorted(results[val_track], key=lambda l:l[1], reverse=True)[:k]
            
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
                
        # Genre classification
        track_ids = [track[0] for track in results[val_track]]
        indices = [np.where(self.train.index == id)[0][0] for id in track_ids]
        genres = [y_train.iloc[index] for index in indices]
        
        if genres != []:
            results[val_track] = most_common(genres)
    
        return results

#%%%%
# hyperparameters
l = 9
n = 7     
k = 7
measures = ['Cosine', 'Euclidean']
m = measures[0]

start = datetime.datetime.now()

lsh = LSH(x_train, n, l, k)
results = lsh.k_neighbors(x_val, k, measure=m)

u = 0
# Checks whether computed genre equals the actual genre
for result in results:
    id = np.where(y_val.index == result)[0][0]
    check = y_val.iloc[id]
    u += results[result] == check

print(f"Classification Accuracy of nearest neighbor search: {np.round(100 * u / len(y_val), 2)} %.")

runtime = (datetime.datetime.now() - start).total_seconds()  
print(f'\nRuntime: {runtime:.2f} seconds using {m}-similarity.\n')
# %%
