#%%
'''
MMD-Assignment #1: Local Sensitive Hashing (LSH) for Item Search and 
Genre Classification for FMAs "small" dataset.
    
Team: Emin Guliev, Justus Rass & Christian Wiskott.
'''

from MusicSearch import *
import datetime

# Optimal Hyperparameters obtained after testing
number_of_hashtables = 5
hash_length = 12
k_nearest_neighbors = 5
subset = 'small'
feature_fields = 'mfcc'
measure = 'Cosine'
magic_number = 5


print(" ##### Phase-1: Training the model with the training data and "
    "testing the model with the validation data #####\n")

# Initialization of class object for knn-search
search = MusicSearch("metadata/", number_of_hashtables, hash_length, subset, feature_fields, measure, 
                     k_nearest_neighbors, magic_number)

print('Begin training ...')
search.train()
print('Training completed.\n')

print('-----------------------------------------')
print('Genre classification for validation data:\n')    

search.test_with_validation()


print(" ##### Phase-2: Training the model with the combined training- and validation data "
    "and testing the model with the test data #####\n")

# Initialization of a new object for second phase
search = MusicSearch("metadata/", number_of_hashtables, hash_length, subset, feature_fields, measure, 
                     k_nearest_neighbors, magic_number)

# Trains the model with the combined training- and validation data
print('Begin training ...')
search.train_with_validation()
print('Training completed.\n')

print('-----------------------------------------')
print('Genre classification for training and validation data combined'
        ' vs. test data\n')

search.test()
# %%
