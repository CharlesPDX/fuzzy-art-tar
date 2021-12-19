import numpy as np

def complement_encode(original_vector: np.array) -> np.array:
    complement = 1-original_vector
    complement_encoded_value = np.concatenate((original_vector,complement), axis=1)
    return complement_encoded_value
