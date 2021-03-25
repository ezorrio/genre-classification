# When drawing uniformly from this matrix, this equals the sought-for probabilities. 

import numpy as np

def generate_R(rows, columns):
    """ Generates sparse zeros-matrix with 1/6-th having -sqrt(3) and
    1/6-th having sqrt(3). """
    
    R = np.zeros((rows*columns))
    
    # 1/6 of length of matrix
    j = int(np.round(rows*columns / 6))
    
    # overwrites 1/6-th of array with sqrt(3) and 1/6-th with -sqrt(3)
    R[:j] = np.sqrt(3); R[j:2*j] = -np.sqrt(3)
    np.random.shuffle(R)
    
    return np.reshape(R, (rows, columns))
