import random
from collections import Counter
import datetime

import numpy as np

from preprocessing import get_tf_idf_twenty_newsgroup_corpus

from fuzzy_artmap_module import FuzzyArtMap
from fuzzy_artmap_module import complement_encode



valid_vector = np.array([[1], [0]])
invalid_vector = np.array([[0], [1]])


def get_test_input_and_output(doc_index, vector, categories, relevant_category):
    if categories[doc_index] == relevant_category:
        output_value = valid_vector
    else:
        output_value = invalid_vector
    
    complement_encoded_input = complement_encode(vector.toarray().transpose())
    return complement_encoded_input, output_value


def test_predictions(fuzzy_artmap, document_indexes, corpus, categories, relevant_category):
    accuracy_counter = Counter({"TP": 0, "TN": 0, "FP": 0, "FN": 0})
    # predictions = []
    for document_index in document_indexes[100:]:
        input_vector, class_vector = get_test_input_and_output(document_index, corpus[document_index], categories, relevant_category)
        prediction, membership_degree = fuzzy_artmap.predict(input_vector)
        # if prediction[0][0]:
        #     predictions.append((membership_degree, document_index, class_vector[0][0]))
        # if prediction[0][0]:
        #     print(f"predicted: {prediction[0][0]} actual: {class_vector[0][0]} membership: {membership_degree} doc: {document_index}")
        # print(f"predicted: {prediction[0][0]} actual: {class_vector[0][0]} membership: {membership_degree}")
        if class_vector[0][0]:
            if prediction[0][0]:
                update = {"TP": 1}
            else:
                update = {"FN": 1}
        else:
            if prediction[0][0]:
                update = {"FP": 1}
            else:
                update = {"TN": 1}
        accuracy_counter.update(update)
    print(accuracy_counter)
    return accuracy_counter

def setup_corpus():
    corpus, categories, _ = get_tf_idf_twenty_newsgroup_corpus()
    seed_indexes = [4000, 4001]
    test_topics = [ "alt.atheism", "comp.graphics", "sci.crypt", "misc.forsale", "sci.med"]
    documents = {index: category for index, category in categories.items() if category in test_topics and index not in seed_indexes }
    shuffled_document_indexes = seed_indexes + random.sample(list(documents.keys()), len(documents))
    return corpus, categories, shuffled_document_indexes

def train_model(corpus, shuffled_document_indexes, categories, relevant_category):
    fuzzy_artmap = FuzzyArtMap(corpus.shape[1]*2, 1, rho_a_bar=0.95)
    for iteration_count, document_index in enumerate(shuffled_document_indexes[:100]):
        print(iteration_count)
        input_vector, class_vector = get_test_input_and_output(document_index, corpus[document_index], categories, relevant_category)
        fuzzy_artmap.train(input_vector, class_vector)
    return fuzzy_artmap

def calculate_metrics(accuracy_data: Counter, duration: datetime.timedelta, number_of_relevant_documents):
    total_documents_tested = sum(accuracy_data.values())
    accuracy = (accuracy_data["TP"] + accuracy_data["TN"]) / total_documents_tested
    precision = accuracy_data["TP"] / (accuracy_data["TP"] + accuracy_data["FP"])
    recall = accuracy_data["TP"] / (accuracy_data["TP"] + accuracy_data["FN"])
    recall_set = accuracy_data["TP"] / number_of_relevant_documents
    rate = total_documents_tested / duration.seconds
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nrecall (set): {recall_set}\ntotal relevant docs: {number_of_relevant_documents}\ntotal docs:{total_documents_tested}\nprediction rate:{rate}")

def run_test():
    relevant_category = "alt.atheism"

    print(f"start: {datetime.datetime.now()}")
    corpus, categories, shuffled_document_indexes = setup_corpus()
    
    print(f"start training: {datetime.datetime.now()}")    
    fuzzy_artmap = train_model(corpus, shuffled_document_indexes, categories, relevant_category)
    
    start_predictions = datetime.datetime.now()
    print(f"start predictions: {start_predictions}")
    accuracy_data = test_predictions(fuzzy_artmap, shuffled_document_indexes, corpus, categories, relevant_category)
    
    end_predictions = datetime.datetime.now()
    prediction_duration = end_predictions-start_predictions
    print(f"end predictions: {end_predictions} - elapsed: {prediction_duration}")
    
    number_of_relevant_documents = len(list([i for i in shuffled_document_indexes[100:] if categories[i] == relevant_category]))
    calculate_metrics(accuracy_data, prediction_duration, number_of_relevant_documents)

if __name__ == "__main__":
   run_test()