# format the 20Newsgroup dataset in TREC format for evaluation with the auto-stop-tar framework https://github.com/dli1/auto-stop-tar 

import os
from pathlib import Path
import json


def get_corpus_files(corpus, full_path):
    for root, _, file in os.walk(full_path):
        if os.path.split(root)[1] == corpus:
            continue

        for file in file:
            # print(os.path.join(root,file))
            yield os.path.join(root,file)

def get_categories():
    for root, _, file in os.walk(full_twenty_newsgroup_path):
        if os.path.split(root)[1] == twenty_newsgroup_corpus:
            continue
        yield os.path.split(root)[1]

def get_newsgroup_categories():
    return {index: os.path.split(os.path.split(file)[0])[1] for index, file in enumerate(get_corpus_files(twenty_newsgroup_corpus, full_twenty_newsgroup_path))}

# topics - query file
def make_topics(target_location):
    topics_path = os.path.join(target_location, "topics")
    if not Path(topics_path).exists():
        os.mkdir(topics_path)
    for category in get_categories():
        topic = {"id": category, "query": "", "title": category}
        with open(os.path.join(topics_path, category), "w") as topic_file:
            topic_file.write(json.dumps(topic))
        
        # {"id": "alt.atheism", "query": "", "title": "alt.atheism" }
# qrels
def make_qrels(target_location, topic):
    qrels_path = os.path.join(target_location, "qrels")
    if not Path(qrels_path).exists():
        os.mkdir(qrels_path)
    qrels = {}
    for file in get_corpus_files(twenty_newsgroup_corpus, full_twenty_newsgroup_path):
        rel_path, document_id = os.path.split(file)
        document_topic = os.path.split(rel_path)[1]
        relevant = document_topic == topic
        qrel = f"{topic}     0  {document_id:5}     {int(relevant)}\n"
        if document_id not in qrels:
            qrels[document_id] = qrel
        elif relevant:
            qrels[document_id] = qrel
    
    with open(os.path.join(qrels_path, topic), "w") as qrel_file:
        qrel_file.writelines(qrels.values())

    #alt.atheism \s+ 0 \s+ doc_id \s+ [0|1] (rel/not rel)

# docids
def make_doc_ids(target_location, topic, doc_ids):
    docids_path = os.path.join(target_location, "docids")
    if not Path(docids_path).exists():
        os.mkdir(docids_path)
    
    with open(os.path.join(docids_path, topic), "w") as docids_file:
        docids_file.writelines(doc_ids)
    #docid\n

def get_all_doc_ids():
    return set([os.path.split(file)[1] + "\n" for file in get_corpus_files(twenty_newsgroup_corpus, full_twenty_newsgroup_path)])


# doctext
def make_doctext(target_location, topic, docs):
    doctext_path = os.path.join(target_location, "doctexts")
    if not Path(doctext_path).exists():
        os.mkdir(doctext_path)
    
    with open(os.path.join(doctext_path, topic), "w") as doctext_file:
        for doc in docs:
            doctext_file.write(json.dumps(doc) + "\n")
    #{"id": "doc_id", "title": "", "content": "body"}\n

def get_all_doctext():
    docs = []
    for file in get_corpus_files(twenty_newsgroup_corpus, full_twenty_newsgroup_path):
        doc = {"title": ""}
        doc["id"] = os.path.split(file)[1]
        with open(file, "r", encoding="latin1") as document_file:
            doc["content"] = document_file.read()
        docs.append(doc)
    return docs

if __name__ == "__main__":
    corpora_root_path = os.path.abspath("../corpora")
    twenty_newsgroup_corpus = "20_newsgroups"
    full_twenty_newsgroup_path = os.path.join(corpora_root_path, twenty_newsgroup_corpus)

    target_location = "/home/ccourc/auto-stop-tar/data/20newsgroups"
    make_topics(target_location)

    for category in get_categories():
        make_qrels(target_location, category)

    all_doc_ids = get_all_doc_ids()
    for category in get_categories():
        make_doc_ids(target_location, category, all_doc_ids)

    docs = get_all_doctext()
    for category in get_categories():
        make_doctext(target_location, category, docs)
