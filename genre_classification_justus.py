import pandas as pd
import numpy as np



'''
Importing and Splitting the Data
'''

df_tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])
df_tracks = df_tracks[df_tracks['set']['subset'] == 'small']
df_features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1])


train_tracks = df_tracks.loc[(df_tracks['set']['split'] == 'training')]
test_tracks = df_tracks.loc[(df_tracks['set']['split'] == 'test')]
validation_tracks = df_tracks.loc[(df_tracks['set']['split'] == 'validation')]


'''
Generate Random Matrix
'''

def gen_rnd_matrix(rows, cols):

    matrix = np.empty((rows, cols))
    print(matrix)

    for i in range(rows):
        for j in range(cols):

            rng = np.random.rand()
            print(rng)
            
            if rng < 1/6:
                matrix[i, j] = np.sqrt(3)
            
            elif rng >= 1/6 and rng < 5/6:
                matrix[i, j] = 0
            
            else:
                matrix[i, j] = -1 * np.sqrt(3)

    return matrix

