import math
from typing import List


class SimpleEmbeddingFunction:
    """A fallback, offline embedding function that does not need API keys or internet connection.

    Generates a deterministic unit-length vector based on word counts mapped to 128 dimensions.
    """

    def _compute_word_hash(self, word: str) -> int:
        """Compute a deterministic hash mapped to 128 dimensions."""
        return hash(word) % 128

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        results = []
        for doc in input_texts:
            vector = [0.0] * 128
            # Basic tokenisation
            words = doc.lower().replace("\n", " ").split()
            for word in words:
                vector[self._compute_word_hash(word)] += 1.0
            # Normalise vector to unit length
            norm = sum(x * x for x in vector) ** 0.5
            if norm > 0:
                vector = [x / norm for x in vector]
            results.append(vector)
        return results


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute the cosine similarity between two vectors.

    Assumes vectors are unit-length (as produced by SimpleEmbeddingFunction),
    in which case the dot product equals cosine similarity.
    Falls back to full cosine formula for non-normalised inputs.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}"
        )

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
