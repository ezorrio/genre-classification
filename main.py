from MusicSearch_new import *
import datetime

if __name__ == '__main__':

    # Hyperparameters
    number_of_hashtables = 5
    hash_length = 12
    k_nearest_neighbors = 5
        
    search = MusicSearch("metadata/", number_of_hashtables, hash_length)
    
    """ Phase-1: Training data vs. validation set """
    training_data = search.data.get_training_data()
    print('training data done')
    validation_data = search.data.get_validation_data() 
    print('validation data done')
    
    print('begin training')
    search.train(training_data)
    print('training done')
    search.test(validation_data)
    print('validation done')

    print('-----------------------------------------')
    print('Genre classification for training and validation data:\n')
    
    search.train(training_data)

    start = datetime.datetime.now()
    search.test(validation_data)
    
    # Prints the runtime of the given subtask
    runtime = (datetime.datetime.now() - start).total_seconds()  
    print(f'Runtime: {runtime/60:.2f} minutes.\n')
    
    
    """ Phase-2: Training data & validation set vs. test data """
    """ NEW INSTANCE OF MusicSearch NEEDED? """

    training_data = search.data.get_training_data()
    validation_data = search.data.get_validation_data()
    
    # combine those two...
    # training_data = pd.concat([training_data, validation_data])
    
    test_data = search.data.get_test_data() 
    
    print('-----------------------------------------')
    print('Genre classification for training and validation data combined'
          ' vs. test data\n')
    
    search.train(training_data)
    
    start = datetime.datetime.now()
    search.test(test_data)

    # Prints the runtime of the given subtask
    runtime = (datetime.datetime.now() - start).total_seconds()  
    print(f'\nRuntime: {runtime/60:.2f} minutes.\n')