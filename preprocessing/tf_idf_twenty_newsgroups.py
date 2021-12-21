#tf-idf vectorization from the 20 Newsgroups data set from https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

import os
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing.processed_corpus import ProcessedCorpus

corpora_root_path = os.path.abspath("../corpora")
twenty_newsgroup_corpus = "20_newsgroups"
full_twenty_newsgroup_path = os.path.join(corpora_root_path, twenty_newsgroup_corpus)

def get_corpus_files(corpus, full_path):
    for root, _, file in os.walk(full_path):
        if os.path.split(root)[1] == corpus:
            continue

        for file in file:
            # print(os.path.join(root,file))
            yield os.path.join(root,file)

def get_newsgroup_categories():
    return {index: os.path.split(os.path.split(file)[0])[1] for index, file in enumerate(get_corpus_files(twenty_newsgroup_corpus, full_twenty_newsgroup_path))}

def get_tf_idf_twenty_newsgroup_corpus() -> ProcessedCorpus:
    twenty_newsgroup_vectorizer = TfidfVectorizer(input='filename', encoding="latin1", stop_words='english', min_df=0.001, max_df=0.9)
    twenty_newsgroup_vectorized_corpus = twenty_newsgroup_vectorizer.fit_transform(get_corpus_files(twenty_newsgroup_corpus, full_twenty_newsgroup_path))
    twenty_newsgroup_categories = get_newsgroup_categories()
    return ProcessedCorpus(vectorized_corpus = twenty_newsgroup_vectorized_corpus,
                           document_corpus_map = {index: index for index in range(twenty_newsgroup_vectorized_corpus.shape[0])},
                           categories = twenty_newsgroup_categories)