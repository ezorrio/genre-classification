from MusicSearch import *
import datetime

# Hyperparameters
number_of_hashtables = 5
hash_length = 12
k_nearest_neighbors = 5
subset = 'small'
feature_fields = 'mfcc'
measure = 'Cosine' # or 'Euclidean'
magic_number = 800


""" Phase-1: Training data vs. validation set """

search = MusicSearch("metadata/", number_of_hashtables, hash_length)

print('begin training')
search.train()
print('training done')

print('-----------------------------------------')
print('Genre classification for training and validation data:\n')    

search.test_with_validation()

""" Phase-2: Training data & validation set vs. test data """

search = MusicSearch("metadata/", number_of_hashtables, hash_length)

search.train_with_validation()

print('-----------------------------------------')
print('Genre classification for training and validation data combined'
        ' vs. test data\n')

search.test()