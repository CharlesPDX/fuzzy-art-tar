# Test using Tf-Idf vectorization over toy corpus

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from fuzzy_artmap import FuzzyArtMap
import complement_encode


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    "And this is about the first document.",
    "What about the third document?",
    "The first document is the important one."
]
relevance = [
    1,
    0,
    0,
    1,
    1,
    0,
    1
]

valid_vector = np.array([[0], [1]])
invalid_vector = np.array([[1], [0]])


def get_test_input_and_output(doc_index, vector):
    if relevance[doc_index] == 1:
        output_value = valid_vector
    else:
        output_value = invalid_vector
    
    complement_encoded_input = complement_encode.complement_encode(vector.toarray().transpose())
    return complement_encoded_input, output_value


if __name__ == "__main__":
    vectorizer = TfidfVectorizer()
    vectorized_corpus = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names_out())

    # print(vectorized_corpus)
    # print(vectorized_corpus.shape)

    # for doc_index, vector in enumerate(X):
    #     print(f"{doc_index} - {vector.todense()}")
    #     print("------")

    fuzzy_artmap = FuzzyArtMap(vectorized_corpus.shape[1]*2, 1, rho_a_bar=0.95)

    last_element = len(relevance) - 1
    for doc_index, vector in enumerate(vectorized_corpus):
        if doc_index == last_element:
            break
        comp, class_vector = get_test_input_and_output(doc_index, vector)
        fuzzy_artmap.train(comp, class_vector)
    
    comp, class_vector = get_test_input_and_output(last_element, vectorized_corpus[last_element])
    prediction = fuzzy_artmap.predict(comp)
    print(f"predicted: {prediction}\nactual:{class_vector}")
