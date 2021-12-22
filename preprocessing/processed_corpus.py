from dataclasses import dataclass

from scipy.sparse.csr import csr_matrix

@dataclass
class ProcessedCorpus:
    vectorized_corpus: csr_matrix
    document_corpus_map: dict[int: int]
    categories: dict[int: list[str]]
