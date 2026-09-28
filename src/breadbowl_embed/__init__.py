"""BreadBowl: encode once, retrieve and rerank."""
from .config import ScoringConfig
from .embeddings import Embeddings
from .model import BreadBowl
from .retrieval import Index, SearchResult
from .scoring import Scores, score

__all__ = ["BreadBowl", "Embeddings", "Index", "ScoringConfig", "Scores", "SearchResult", "score"]
__version__ = "0.1.0a1"
