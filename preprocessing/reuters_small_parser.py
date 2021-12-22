import os
import html
import pprint
import re
from html.parser import HTMLParser


class ReutersParser(HTMLParser):
    """
    ReutersParser subclasses HTMLParser and is used to open the SGML
    files associated with the Reuters-21578 categorised test collection.

    The parser is a generator and will yield a single document at a time.
    Since the data will be chunked on parsing, it is necessary to keep 
    some internal state of when tags have been "entered" and "exited".
    Hence the in_body, in_topics and in_topic_d boolean members.
    Parser based on https://www.quantstart.com/articles/supervised-learning-for-document-classification-with-scikit-learn/
    """
    def __init__(self, encoding='latin-1'):
        """
        Initialise the superclass (HTMLParser) and reset the parser.
        Sets the encoding of the SGML files by default to latin-1.
        """
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        """
        This is called only on initialisation of the parser class
        and when a new topic-body tuple has been generated. It
        resets all off the state so that a new tuple can be subsequently
        generated.
        """
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""
        self.id = ""
        self.lewis_split = ""
        self.cgi_split = ""

    def parse(self, fd):
        """
        parse accepts a file descriptor and loads the data in chunks
        in order to minimise memory usage. It then yields new documents
        as they are parsed.
        """
        self.docs = {} # convert docs to dictionary to return richer information, including document id
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs.items():
                yield doc
            self.docs = {}
        self.close()

    def handle_starttag(self, tag, attributes):
        """
        This method is used to determine what to do when the parser
        comes across a particular tag of type "tag". In this instance
        we simply set the internal state booleans to True if that particular
        tag has been found.
        """
        if tag == "reuters":
            reuters_tag_attributes = {attribute[0]: attribute[1] for attribute in attributes}
            self.id = int(reuters_tag_attributes["newid"])
            self.lewis_split = reuters_tag_attributes["lewissplit"]
            self.cgi_split = reuters_tag_attributes["cgisplit"]
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = self.in_topics and True 

    def handle_endtag(self, tag):
        """
        This method is used to determine what to do when the parser
        finishes with a particular tag of type "tag". 

        If the tag is a <REUTERS> tag, then we remove all 
        white-space with a regular expression and then append the 
        topic-body tuple.

        If the tag is a <BODY> or <TOPICS> tag then we simply set
        the internal state to False for these booleans, respectively.

        If the tag is a <D> tag (found within a <TOPICS> tag), then we
        append the particular topic to the "topics" list and 
        finally reset it.
        """
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs[self.id] = {"topics": self.topics, "body": self.body, "lewis_split": self.lewis_split, "cgi_split": self.cgi_split}
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            if self.in_topics: # fixed from original to only extend topics while in the topics tag
                self.topics.append(self.topic_d)
            self.topic_d = ""  

    def handle_data(self, data):
        """
        The data is simply appended to the appropriate member state
        for that particular tag, up until the end closing tag appears.
        """
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data

        
if __name__ == "__main__":
    corpora_root_path = os.path.abspath("../corpora")
    reuters_small_corpus = "reuters21578"
    full_reuters_small_path = os.path.join(corpora_root_path, reuters_small_corpus)
    # Open the first Reuters data set and create the parser
    filename = os.path.join(full_reuters_small_path, "reut2-000.sgm")
    parser = ReutersParser()

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    doc = parser.parse(open(filename, 'rb'))
    pprint.pprint(list(doc))
