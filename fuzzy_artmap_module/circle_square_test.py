# Replication of section VI. Simulation Circle-in-the-square from Carpenter et al., 1992
from math import sqrt
from collections import Counter
from datetime import datetime

import numpy as np

from fuzzy_artmap import FuzzyArtMap
import complement_encode

num_pats = 1000
test_pats = 100
sq = 1;                         # Size of square
r = sq/sqrt(2*np.pi);           # Radius of circle so it's half area of square
xcent = 0.5
ycent = 0.5                     # Center of circle
xs = xcent*np.ones((num_pats, 1))
ys = ycent*np.ones((num_pats, 1))
train_rng = np.random.Generator(np.random.PCG64(12345))
rng = np.random.Generator(np.random.PCG64())
a = np.concatenate((xs,ys), axis=1) + 0.5-train_rng.random((num_pats, 2))
bmat = ((a[:, 0, np.newaxis]-xcent)**2 + (a[:, 1, np.newaxis]-ycent)**2) > r**2
bmat = np.concatenate((bmat, 1-bmat), axis=1)

xs = xcent*np.ones((test_pats, 1))
ys = ycent*np.ones((test_pats, 1))
test_set = np.concatenate((xs,ys), axis=1) + 0.5-rng.random((test_pats, 2))
test_truth = ((test_set[:, 0, np.newaxis]-xcent)**2 + (test_set[:, 1, np.newaxis]-ycent)**2) > r**2
test_truth =np.concatenate((test_truth, 1-test_truth), axis=1)


if __name__ == "__main__":
    fuzzy_artmap = FuzzyArtMap(4,1)
    start_time = datetime.now()
    print(start_time)
    for i in range(num_pats):
        test_input = a[np.newaxis, i, :]
        ground_truth = bmat[np.newaxis, i, :]
        complement_encoded_input = complement_encode.complement_encode(test_input)        
        fuzzy_artmap.train(complement_encoded_input, ground_truth)

    out_test_point = np.array([[0.115, 0.948]])
    encoded_test_point = complement_encode.complement_encode(out_test_point)
    prediction, membership_degree = fuzzy_artmap.predict(encoded_test_point)
    print(prediction, membership_degree)

    in_test_point = np.array([[0.262, 0.782]])
    encoded_test_point = complement_encode.complement_encode(in_test_point)
    prediction, membership_degree = fuzzy_artmap.predict(encoded_test_point)
    print(prediction, membership_degree)

    # run predictions on training and test data, calculating error rate
    # training_predictions = Counter()
    # for i in range(num_pats):
    #     test_input = a[:, i, np.newaxis]
    #     ground_truth = bmat[:, i, np.newaxis]
    #     complement_encoded_input = complement_encode.complement_encode(test_input)        
    #     prediction = fuzzy_artmap.predict(complement_encoded_input)
    #     correct = np.all(prediction == ground_truth)
    #     training_predictions.update([correct])
    # print(training_predictions)

    # test_predictions = Counter()
    # for i in range(test_pats):
    #     test_input = test_set[:, i, np.newaxis]
    #     ground_truth = test_truth[:, i, np.newaxis]
    #     complement_encoded_input = complement_encode.complement_encode(test_input)        
    #     prediction = fuzzy_artmap.predict(complement_encoded_input)
    #     correct = np.all(prediction == ground_truth)
    #     test_predictions.update([correct])
    # stop_time = datetime.now()
    # print(f"elapsed: {stop_time-start_time}- {stop_time}")
    # print(test_predictions)
    # print(fuzzy_artmap.weight_a.shape)
    # print(fuzzy_artmap.weight_ab.shape)
    # print(np.count_nonzero(fuzzy_artmap.weight_a[0, :] < 1, 0))
