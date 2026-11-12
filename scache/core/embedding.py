from numpy import ndarray
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def encoded(query: str|list[str]) -> ndarray:
    """
        output L2-normalized float32 ndarrays(shape is 2d : num_inputs, output_dimension)
        so that FAISS similarity directly computes consine similarity
    """
    if isinstance(query, str):
        query = [query]

    return model.encode(query, precision='float32', normalize_embeddings=True)

