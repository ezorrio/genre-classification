from MusicSearch import *
import datetime

# Hyperparameters
number_of_hashtables = 5
hash_length = 12
k_nearest_neighbors = 5

search = MusicSearch("metadata/", number_of_hashtables, hash_length, measure='Euclidean')

""" Phase-1: Training data vs. validation set """

print('begin training')
search.train()
print('training done')

print('-----------------------------------------')
print('Genre classification for training and validation data:\n')    

search.test_with_validation()

""" Phase-2: Training data & validation set vs. test data """
""" NEW INSTANCE OF MusicSearch NEEDED? """

search = MusicSearch("metadata/", number_of_hashtables, hash_length)

search.train_with_validation()

print('-----------------------------------------')
print('Genre classification for training and validation data combined'
        ' vs. test data\n')

search.test()