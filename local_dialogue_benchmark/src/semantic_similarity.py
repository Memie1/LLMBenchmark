from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover - exercised only when the package is missing
    SentenceTransformer = None
    util = None


DEFAULT_EMBEDDING_MODEL = os.getenv("BENCHMARK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
ALLOW_FALLBACK = os.getenv("BENCHMARK_ALLOW_FALLBACK", "").lower() in {"1", "true", "yes"}
# small token normalization keeps the explicit dev fallback less noisy in tests
WORD_NUMBERS = {
    "a": "1",
    "an": "1",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def singularize(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_tokens(text: str) -> list[str]:
    # normalize numbers and plurals before the fallback matcher compares strings
    tokens = re.findall(r"[a-z0-9$']+", normalize_text(text))
    normalized: list[str] = []

    for token in tokens:
        if token.startswith("$") and token[1:].isdigit():
            normalized.append(token[1:])
            continue
        if token in WORD_NUMBERS:
            normalized.append(WORD_NUMBERS[token])
            continue
        normalized.append(singularize(token))

    return normalized


def char_ngrams(text: str, size: int = 3) -> set[str]:
    # character ngrams make the fallback a bit more forgiving to wording changes
    normalized = " ".join(normalize_tokens(text))
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index:index + size] for index in range(len(normalized) - size + 1)}


def fallback_similarity(a: str, b: str) -> float:
    # this only exists for explicit dev mode and should not back official benchmark runs
    normalized_a = " ".join(normalize_tokens(a))
    normalized_b = " ".join(normalize_tokens(b))

    if not normalized_a or not normalized_b:
        return 0.0

    seq_score = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    grams_a = char_ngrams(normalized_a)
    grams_b = char_ngrams(normalized_b)
    overlap = len(grams_a & grams_b) / len(grams_a | grams_b) if grams_a and grams_b else 0.0
    tokens_a = set(normalize_tokens(a))
    tokens_b = set(normalize_tokens(b))
    token_overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a and tokens_b else 0.0
    return (0.35 * seq_score) + (0.25 * overlap) + (0.4 * token_overlap)


@dataclass
class SemanticSimilarityBackend:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    allow_fallback: bool = ALLOW_FALLBACK
    backend_name: str = field(init=False, default="uninitialized")
    _model: SentenceTransformer | None = field(init=False, default=None)
    _load_attempted: bool = field(init=False, default=False)

    # load lazily so importing the scorer stays cheap
    def _ensure_model(self) -> SentenceTransformer | None:
        if self._load_attempted:
            return self._model

        self._load_attempted = True
        if SentenceTransformer is None:
            if self.allow_fallback:
                self.backend_name = "fallback"
                return None
            raise RuntimeError(
                "sentence-transformers is required for benchmark scoring but is not installed. "
                "Install it or set BENCHMARK_ALLOW_FALLBACK=1 for explicit dev-only fallback mode."
            )

        try:
            self._model = SentenceTransformer(self.model_name)
            self.backend_name = f"sentence-transformers:{self.model_name}"
        except Exception as exc:
            if self.allow_fallback:
                self._model = None
                self.backend_name = "fallback"
                return None
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}'."
            ) from exc
        return self._model

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0

        model = self._ensure_model()
        if model is None or util is None:
            return fallback_similarity(a, b)

        # map cosine similarity into a clean 0 to 1 range for the scorer
        embeddings = model.encode([a, b], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, min(1.0, (cosine_score + 1.0) / 2.0))


_BACKEND: SemanticSimilarityBackend | None = None


def get_similarity_backend() -> SemanticSimilarityBackend:
    # reuse one backend instance so repeated scoring calls do not reload the embedding model
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = SemanticSimilarityBackend()
    return _BACKEND


def reset_similarity_backend() -> None:
    # tests use this when they need a clean backend state
    global _BACKEND
    _BACKEND = None
