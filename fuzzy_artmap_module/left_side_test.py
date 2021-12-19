# Simple linearly seperable groups test

import random

import numpy as np

from fuzzy_artmap import FuzzyArtMap
import complement_encode

valid_vector = np.array([[1, 0]])
invalid_vector = np.array([[0, 1]])


def get_test_input_and_output():
    first = random.random()
    second = random.random()
    if first <= 0.5:
        output_value = valid_vector
    else:
        output_value = invalid_vector
    test_input = np.array([[first, second]])
    complement_encoded_input = complement_encode.complement_encode(test_input)

    return complement_encoded_input, output_value


if __name__ == "__main__":
    fuzzy_artmap = FuzzyArtMap(4, 1)

    random.seed(52)
    for i in range(100):
        comp, class_vector = get_test_input_and_output()
        fuzzy_artmap.train(comp, class_vector)

    comp, class_vector = get_test_input_and_output()
    prediction, membership_degree = fuzzy_artmap.predict(comp)
    print(f"predicted: {prediction}\nactual:{class_vector}\nmembership:{membership_degree}")
