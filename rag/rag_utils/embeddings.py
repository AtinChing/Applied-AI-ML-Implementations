from typing import List
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


class LocalHuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        vectors = self.model.encode(list(input), convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]