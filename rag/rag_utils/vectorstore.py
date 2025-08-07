from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb.api.models.Collection import Collection


@dataclass
class SearchResults:
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]


def get_or_create_collection(persist_path: str, collection_name: str, embedding_function) -> Collection:
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)


def add_documents(collection: Collection, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
    # Chroma accepts batches
    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def similarity_search(collection: Collection, query: str, k: int = 6) -> SearchResults:
    res = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    # Chroma returns lists of lists (per query)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return SearchResults(ids=ids, documents=docs, metadatas=metas)