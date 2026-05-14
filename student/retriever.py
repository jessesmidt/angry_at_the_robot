# Retrieving: Since the model is not trained on your specific data, it needs to
# search the database to retrieve the most useful snippets. First, the model needs to
# understand your question. Once that’s done, it matches the query with the indexed
# database to choose the best results and finally pulls out the most relevant pieces of
# information. This involves query encoding, similarity search, and ranking

# load saved BM25 index + chunks
#         ↓
# take a query string
#         ↓
# tokenize + search BM25
#         ↓
# get back indices [42, 107, 3, 891, 204]
#         ↓
# look up those indices in chunks list
#         ↓
# return list[Chunk]

import bm25s
from pydantic import BaseModel
from .models import Chunk
from .indexer import load_chunks

class Retriever:
    def __init__(self) -> None:
        self.bm25 = bm25s.BM25.load("data/processed/bm_25_index")
        self.chunks = load_chunks("data/processed/chunks")

    def retrieve(self, query: str, k: int) -> list[list[Chunk]]:
        query_tokens = bm25s.tokenize(query)
        results, _ = self.bm25.retrieve(query_tokens, k=k)

        return [self.chunks[i] for i in results[0]]
