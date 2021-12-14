# Based on fuzzyartmap_demo.m by Rajeev Raizada, wih permission, and 
# Carpenter, G. A., Grossberg, S., Markuzon, N., Reynolds, J. H. and Rosen, D. B. (1992)
# "Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps"
# IEEE Transactions on Neural Networks, Vol. 3, No. 5, pp. 698-713.

import numpy as np

class FuzzyArtMap:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa
        self.weight_a = np.ones((f1_size, f2_size)) # Initial weights in ARTa. All set to 1 Row-i, col-j entry = weight from input node i to F2 coding node j
        self.weight_ab = np.ones((number_of_categories, f2_size))  # Row-k, col-j entry = weight from ARTa F2  node j to Map Field node k
        # self.committed_nodes = [] # probably originally intended as an optimization for Fa mismatch to find first uncommited node
        
    def _resonance_search(self, input_vector, already_reset_nodes, rho_a, allow_category_growth = True, predict = False):
        resonant_a = False
        subset_by_node = {}
        while not resonant_a:
            N = self.weight_a.shape[1]  # Count how many F2a nodes we have

            A_for_each_F2_node = input_vector * np.ones((1, N))
            # Matrix containing a copy of A for each F2 node. 
            # was optimization for Matlab, might be different in Python

            A_AND_w = np.minimum(A_for_each_F2_node, self.weight_a)
            # Fuzzy AND = min

            S = sum(A_AND_w) # fsum might be a better operator
            # Row vector of signals to F2 nodes

            T = S / (self.alpha + sum(self.weight_a))
            # Choice function vector for F2

            # Set all the reset nodes to zero
            T[already_reset_nodes] = np.zeros((1, len(already_reset_nodes)))

            # Finding the winning node, J
            J = np.argmax(T)
            # NumPy argmax function works such that J is the lowest index of max T elements, as desired. J is the winning F2 category node

            # y = np.zeros((N, 1))
            # y[J]=1
            # # Activities of F2. All zero, except J; unused

            w_J = self.weight_a[:, J, np.newaxis]
            # Weight vector into winning F2 node, J

            x = np.minimum(input_vector, w_J)
            # Fuzzy version of 2/3 rule. x is F1 activity
            # NB: We could also use J-th element of S, since the top line of the match fraction
            # |I and w|/|I| is sum(x), which is
            # S = sum(A_AND_w) from above

            # Testing if the winning node resonates in ARTa
            membership_degree = sum(x)/sum(input_vector)
            if predict:
                subset_by_node[J] = membership_degree[0]
            
            if membership_degree >= rho_a:
                resonant_a = True
                # returning from this method will return winning ARTMAPa node index (J) and weighted input vector
            else:
                # If mismatch then we reset
                resonant_a = False
                already_reset_nodes.append(J)
                # Record that node J has been reset already.

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) == N:
                if allow_category_growth:
                    self.weight_a = np.concatenate((self.weight_a, np.ones((self.weight_a.shape[0], 1))), axis=1)
                    self.weight_ab = np.concatenate((self.weight_ab, np.ones((self.weight_ab.shape[0], 1))), axis=1)
                    # Give the new F2a node a w_ab entry, this new node should win
                else:
                    return -1, None, subset_by_node
            # End of the while loop searching for ARTa resonance
            # If not resonant_a, we pick the next highest Tj and see if *that* node resonates, i.e. goto "while"
            # If resonant_a, we have found an ARTa resonance, namely node J
            # Return from method to see if we get Fab match with node J

        return J, x, subset_by_node

    def train(self, input_vector: np.array, class_vector: np.array):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = []
        # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa

        while not resonant_ab:
            J, x, _ = self._resonance_search(input_vector, already_reset_nodes, rho_a)

            # Desired output for input number i
            z = np.minimum(class_vector, self.weight_ab[:, J, np.newaxis])   # Fab activation vector, z
            # (Called x_ab in Fuzzy ARTMAP paper)
            
            #Test for Fab resonance
            if sum(z)/sum(class_vector) >= self.rho_ab:
                resonant_ab = True
            # This will cause us to leave the while 'not resonant_ab' loop and go on to do learning.

            else: # We have an Fab mismatch
                # Increase rho_a vigilance.
                # This will cause F2a node J to get reset when we go back through the ARTa search loop again.
                # Also, *for this input*, the above-baseline vigilance will cause a finer ARTa category to win
                rho_a = sum(x)/sum(input_vector) + self.epsilon

        #### End of the while 'not resonant_ab' loop.
        #### Now we have a resonating ARTa output which gives a match at the Fab layer.
        #### So, we go on to have learning in the w_a and w_ab weights

        #### Let the winning, matching node J learn
        self.weight_a[:, J, np.newaxis] = self.beta * x + (1-self.beta) * self.weight_a[:, J, np.newaxis]
        # NB: x = min(A,w_J) = I and w
        
        #### Learning on F1a <--> F2a weights
        self.weight_ab[:, J, np.newaxis] = self.beta * z + (1-self.beta) * self.weight_ab[:, J, np.newaxis]
        # NB: z=min(b,w_ab(J))=b and w

    def predict(self, input_vector: np.array):
        rho_a = 0 # set ARTa vigilance to first match
        J, _, membership_by_node = self._resonance_search(input_vector, [], rho_a, False, predict = True)
        
        # prediction transliterated from fuzzyartmap_demo.m, does not appear to be any different from z
        # prediction_transliteration = self.weight_ab[:,J]/sum(self.weight_ab[:,J]) 
        # print(prediction_transliteration)
        # (Called x_ab in Fuzzy ARTMAP paper)
        return self.weight_ab[:, J, np.newaxis], membership_by_node[J] # Fab activation vector & fuzzy membership value
