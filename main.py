from MusicSearch import *

if __name__ == '__main__':

    # Hyperparameters
    number_of_hashtables = 5
    hash_length = 12
    k_nearest_neighbors = 5
        
    search = MusicSearch("metadata/", number_of_hashtables, hash_length)
    
    # Phase-1: Training data vs. validation set 
    training_data = search.data.get_training_data()
    validation_data = search.data.get_validation_data() 
    
    search.train(training_data)
    search.test(validation_data)

    print('-----------------------------------------')
    print('Genre classification for training and validation data:\n')
    
    # Phase-2: Training data & validation set vs. test data
    training_data = search.data.get_training_data()
    validation_data = search.data.get_validation_data()
    
    # combine those two...
    # training_data = pd.concat([training_data, validation_data])
    
    test_data = search.data.get_test_data() 
    
    search.train(training_data)
    search.test(test_data)

    print('-----------------------------------------')
    print('Genre classification for training and validation data combined'
          ' vs. test data\n')
    
