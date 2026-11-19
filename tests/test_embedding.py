import numpy as np
import pytest

from scache.core.embedding import encoded
from scache.core.similarity import cosine_similarity, batch_cosine_similarity


def test_embedding_single_string_shape():
    """Verify that a single string input produces a 2D array of shape (1, 384)."""
    vec = encoded("Hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 2
    assert vec.shape == (1, 384)
    assert vec.dtype == np.float32


def test_embedding_batch_strings_shape():
    """Verify that a list of strings produces shape (N, 384)."""
    prompts = ["Hello", "How are you?", "Testing embeddings."]
    vecs = encoded(prompts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2
    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32


def test_embedding_l2_normalization():
    """Verify that output vectors are L2-normalized (length == 1.0)."""
    vecs = encoded(["Alpha", "Beta", "Gamma"])
    norms = np.linalg.norm(vecs, axis=1)
    for norm in norms:
        assert pytest.approx(norm, rel=1e-5) == 1.0


def test_cosine_similarity_identical():
    """Identical sentences should have cosine similarity approx 1.0."""
    v1 = encoded("Reverse a linked list in Python")
    v2 = encoded("Reverse a linked list in Python")
    sim = cosine_similarity(v1, v2)
    assert pytest.approx(sim, abs=1e-4) == 1.0


def test_semantic_ranking():
    """Semantically related phrases should score higher than unrelated phrases."""
    v_base = encoded("How do I invert a binary tree?")
    v_similar = encoded("Algorithm to reverse a binary tree")
    v_unrelated = encoded("Recipe for chocolate chip cookies")

    sim_similar = cosine_similarity(v_base, v_similar)
    sim_unrelated = cosine_similarity(v_base, v_unrelated)

    assert sim_similar > 0.80
    assert sim_unrelated < 0.35
    assert sim_similar > sim_unrelated


def test_batch_cosine_similarity():
    """Verify batch pairwise similarity computation."""
    list_a = ["Apple Inc.", "Python language"]
    list_b = ["Apple company", "Python programming"]

    v_a = encoded(list_a)
    v_b = encoded(list_b)

    sims = batch_cosine_similarity(v_a, v_b)
    assert len(sims) == 2
    assert sims[0] > 0.85
    assert sims[1] > 0.85
