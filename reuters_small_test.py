import os 
import random
from collections import Counter
import datetime

import numpy as np
from unsync import unsync

from preprocessing import get_tf_idf_reuters_small_corpus

from fuzzy_artmap_module import FuzzyArtMap
from fuzzy_artmap_module import complement_encode



valid_vector = np.array([[1, 0]])
invalid_vector = np.array([[0, 1]])


def get_test_input_and_output(doc_index, vector, categories, relevant_category):
    if relevant_category in categories[doc_index]:
        output_value = valid_vector
    else:
        output_value = invalid_vector
    
    complement_encoded_input = complement_encode(vector.toarray())
    return complement_encoded_input, output_value


def test_predictions(fuzzy_artmap, document_indexes, corpus, categories, relevant_category, document_corpus_index_map):
    accuracy_counter = Counter({"TP": 0, "TN": 0, "FP": 0, "FN": 0})
    for corpus_index in document_indexes[100:]:
        document_index = document_corpus_index_map[corpus_index]
        input_vector, class_vector = get_test_input_and_output(document_index, corpus[corpus_index], categories, relevant_category)
        prediction, _ = fuzzy_artmap.predict(input_vector)
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
    corpus, document_corpus_index_map, categories = get_tf_idf_reuters_small_corpus()
    corpus_seed_indexes = set()
    for corpus_index, document_index in document_corpus_index_map.items():
        if document_index in seed_indexes:
            corpus_seed_indexes.add(corpus_index)
    # documents = {index: category for index, category in categories.items() if category in test_topics and index not in seed_indexes }
    # categories = {index: category for index, category in categories.items() if index not in seed_indexes }
    random_indexes = set(document_corpus_index_map.keys())
    random_indexes.difference_update(corpus_seed_indexes)
    random_indexes = random.sample(random_indexes, len(random_indexes))
    
    shuffled_document_indexes = list(corpus_seed_indexes) + random_indexes
    return corpus, categories, shuffled_document_indexes, document_corpus_index_map


def train_model(corpus, shuffled_document_indexes, categories, relevant_category, document_corpus_index_map):
    fuzzy_artmap = FuzzyArtMap(corpus.shape[1]*2, 36, rho_a_bar=0.95)
    training_split = Counter()
    for iteration_count, corpus_index in enumerate(shuffled_document_indexes[:100]):
        document_index = document_corpus_index_map[corpus_index]
        print(f"{iteration_count} - {categories[document_index]}")
        training_split.update({''.join(categories[document_index]):1})
        input_vector, class_vector = get_test_input_and_output(document_index, corpus[corpus_index], categories, relevant_category)
        fuzzy_artmap.train(input_vector, class_vector)
    print(training_split)
    processed_document_indexes.update(shuffled_document_indexes[:100])
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
    print(f"start: {datetime.datetime.now()}")
    corpus, categories, shuffled_document_indexes, document_corpus_index_map = setup_corpus()
    
    print(f"start training: {datetime.datetime.now()}")    
    fuzzy_artmap = train_model(corpus, shuffled_document_indexes, categories, relevant_category, document_corpus_index_map)
    
    start_predictions = datetime.datetime.now()
    print(f"start predictions: {start_predictions}")
    accuracy_data = test_predictions(fuzzy_artmap, shuffled_document_indexes, corpus, categories, relevant_category, document_corpus_index_map)
    
    end_predictions = datetime.datetime.now()
    prediction_duration = end_predictions-start_predictions
    print(f"end predictions: {end_predictions} - elapsed: {prediction_duration}")
    
    number_of_relevant_documents = len(list([i for i in shuffled_document_indexes[100:] if relevant_category in categories[document_corpus_index_map[i]]]))
    calculate_metrics(accuracy_data, prediction_duration, number_of_relevant_documents)

# @profile
def run_active_learning_test():
    print(f"start: {datetime.datetime.now()}")
    corpus, categories, shuffled_document_indexes, document_corpus_index_map = setup_corpus()
    available_document_indexes = set(shuffled_document_indexes[100:])
    number_of_relevant_documents = len(list([i for i in shuffled_document_indexes[100:] if relevant_category in categories[document_corpus_index_map[i]]]))

    print(f"start training: {datetime.datetime.now()}")    
    fuzzy_artmap = train_model(corpus, shuffled_document_indexes, categories, relevant_category, document_corpus_index_map)
    
    found_relevant_documents = 0
    active_learning_iteration = 0
    has_candidates = True
    start_predictions = datetime.datetime.now()
    print(f"start active learning: {start_predictions}")
    batch_size = 100
    while found_relevant_documents != number_of_relevant_documents and has_candidates: 
        relevent_documents_in_batch = 0
        candidates = query(fuzzy_artmap, corpus, categories, available_document_indexes, document_corpus_index_map)
        for candidate in candidates[:batch_size]:
            # print(f"{datetime.datetime.now()} - training")
            fuzzy_artmap.train(candidate[3], candidate[2])
            available_document_indexes.remove(candidate[1]) 
            if candidate[2][0,][0]:
                found_relevant_documents += 1
                relevent_documents_in_batch += 1

        if len(candidates) == 0:
            has_candidates = False
        active_learning_iteration += 1
        print(f"{datetime.datetime.now()} - {active_learning_iteration} - {found_relevant_documents}/{number_of_relevant_documents} | batch recall: {relevent_documents_in_batch/batch_size} | recall - {found_relevant_documents/number_of_relevant_documents} precision - {found_relevant_documents/(batch_size * active_learning_iteration)} | {len(available_document_indexes)}")
    
    end_predictions = datetime.datetime.now()
    prediction_duration = end_predictions-start_predictions
    print(f"end active learning: {end_predictions} - elapsed: {prediction_duration}")
    print(f"number of Fuzzy ARTMAP Categories: {fuzzy_artmap.weight_a.shape[0]}")


def query(fuzzy_artmap, corpus, categories, available_document_indexes, document_corpus_index_map):    
    working_indexes = list(available_document_indexes)
    chunk_size = 250
    tasks = []
    for i in range(0, len(working_indexes), chunk_size):
        tasks.append(get_predictions(fuzzy_artmap, corpus, categories, working_indexes[i:i+chunk_size], document_corpus_index_map))
    
    predictions = []
    for task in tasks:
        predictions.extend(task.result())
        
    predictions.sort(key=lambda p: p[0], reverse=True)    
    return predictions


@unsync(cpu_bound=True)
def get_predictions(fuzzy_artmap, corpus, categories, document_index_chunk, document_corpus_index_map):
    # print(f"{datetime.datetime.now()} - {os.getpid()} - {len(document_index_chunk)}")
    predictions = []
    for corpus_index in document_index_chunk:
        document_index = document_corpus_index_map[corpus_index]
        input_vector, class_vector = get_test_input_and_output(document_index, corpus[corpus_index], categories, relevant_category)
        prediction, membership_degree = fuzzy_artmap.predict(input_vector)
        if prediction[0][0]:
            predictions.append((membership_degree, corpus_index, class_vector, input_vector))
    return predictions


if __name__ == "__main__":
    processed_document_indexes = set()
    relevant_category = "grain"
    # seed_indexes = [13067,13070,13458,6267,124,136,4524,4637,1299,1406,1623,1631,1731,1845,15303,15580,15914,8374,8613,8656,8686,8895,8943,14212,14313,14340,14389,14828,21123,19964,3390,5363,5408,5611,5800,5826,5972,11065,11768,11862,12338,12830,9907,16359,7154]
    seed_indexes = [13067,13070,13458,6267]    
    # run_test()
    run_active_learning_test()
    