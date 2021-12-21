# tf-idf vectorization from the Reuters-21578 data set from https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection
import os

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing.processed_corpus import ProcessedCorpus
from .reuters_small_parser import ReutersParser

corpora_root_path = os.path.abspath("../corpora")
reuters_small_corpus = "reuters21578"
full_reuters_small_path = os.path.join(corpora_root_path, reuters_small_corpus)

def get_corpus_files(full_path):
    for root, _, file in os.walk(full_path):
        for file in file:
            if ".sgm" in file:
                # print(os.path.join(root,file))
                yield os.path.join(root,file)

def get_text_corpus(full_path):
    parser = ReutersParser()
    documents = {}
    for file in get_corpus_files(full_path):
        with open(file, 'rb') as corpus_file:
            documents.update(parser.parse(corpus_file))
    return documents

def get_tf_idf_reuters_small_corpus() -> ProcessedCorpus:
    documents = get_text_corpus(full_reuters_small_path)
    documents_with_topics = {document_id: document for document_id, document in documents.items() if len(document["topics"]) > 0}
    documents_with_topics_and_bodies = {document_id: document for document_id, document in documents_with_topics.items() if len(document["body"]) > 0}
    reuters_small_vectorizer = TfidfVectorizer(input='content', encoding="latin1", stop_words='english', min_df=0.001, max_df=0.9)
    reuters_small_vectorized_corpus = reuters_small_vectorizer.fit_transform([document["body"] for document in documents_with_topics_and_bodies.values()])
    document_corpus_index_map = {index: document_id for index, document_id in enumerate(documents_with_topics_and_bodies.keys())}
    return ProcessedCorpus(vectorized_corpus = reuters_small_vectorized_corpus, 
                           document_corpus_map = document_corpus_index_map, 
                           categories = {document_id: document["topics"] for document_id, document in documents_with_topics_and_bodies.items()})
