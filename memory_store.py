import json
import logging
import math
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorEmbedding:
    """Neural-style vector embedding for semantic memory search."""

    vector: List[float]  # Dense embedding vector
    embedding_type: str  # 'experiment', 'pattern', 'strategy', 'failure'
    dimensions: int
    created_at: str

    def to_dict(self) -> Dict:
        return {
            "vector": self.vector,
            "embedding_type": self.embedding_type,
            "dimensions": self.dimensions,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VectorEmbedding":
        return cls(
            vector=data["vector"],
            embedding_type=data["embedding_type"],
            dimensions=data["dimensions"],
            created_at=data["created_at"],
        )


@dataclass
class MemoryEntry:
    """Enhanced memory entry with vector embeddings for XPR* Engine."""

    timestamp: str
    experiment_name: str
    pattern_type: str  # 'success_pattern', 'failure_mode', or 'strategy'
    content: str
    context: Dict[str, str]  # Additional metadata
    embedding: Optional[VectorEmbedding] = None  # Semantic vector representation
    memory_id: str = ""  # Unique identifier

    def __post_init__(self):
        if not self.memory_id:
            self.memory_id = hashlib.md5(
                f"{self.timestamp}:{self.experiment_name}:{self.content}".encode()
            ).hexdigest()[:12]


class MemoryStore:
    """
    Indexed knowledge continuous learning storage.
    Acts as a primitive 'Vector DB' / indexed JSON system for textual success and failure patterns.
    Supports both keyword-exact and TF-IDF semantic search for retrieval.
    """

    def __init__(self, storage_path: str = "xpr_memory.json"):
        self.storage_path = Path(storage_path)
        self.memory: List[MemoryEntry] = self._load_memory()
        # TF-IDF caches (rebuilt on first semantic query or after add)
        self._tfidf_dirty = True
        self._idf: Dict[str, float] = {}
        self._doc_vectors: List[Dict[str, float]] = []
        # Check for advanced embedding availability
        self.has_semantic_embeddings = self._check_embedding_availability()

    def _check_embedding_availability(self) -> bool:
        """Check if advanced semantic embedding models are available."""
        try:
            from sentence_transformers import SentenceTransformer

            # Test if we can load a model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            _ = model.encode("test")  # Test encoding
            logger.info(
                "Advanced semantic embeddings (sentence-transformers) available"
            )
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not available, using hash-based embeddings only"
            )
            return False

    def _load_memory(self) -> List[MemoryEntry]:
        """Load the memory file if it exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    return [MemoryEntry(**entry) for entry in data]
            except Exception as e:
                logger.error(f"Failed to load memory store: {e}")
        return []

    def _save_memory(self):
        """Save the memory list to the file."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump([asdict(entry) for entry in self.memory], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory store: {e}")

    def add_memory(
        self,
        experiment_name: str,
        pattern_type: str,
        content: str,
        context: Optional[Dict[str, str]] = None,
    ):
        """Add a new insight to the memory store."""
        new_entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            experiment_name=experiment_name,
            pattern_type=pattern_type,
            content=content,
            context=context or {},
        )
        self.memory.append(new_entry)
        self._tfidf_dirty = True
        self._save_memory()
        logger.info(f"Added new memory [{pattern_type}] for {experiment_name}.")

    def _generate_embedding(
        self, text: str, embedding_type: str = "experiment", dimensions: int = 128
    ) -> VectorEmbedding:
        """
        Generate a neural-style vector embedding for semantic search.
        Uses advanced semantic embedding techniques for better representation.

        Args:
            text: The text content to embed
            embedding_type: Type of embedding ('experiment', 'pattern', 'strategy', 'failure')
            dimensions: Vector dimensionality

        Returns:
            VectorEmbedding object with dense vector representation
        """
        try:
            # Try to use sentence-transformers for proper semantic embeddings
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )  # Lightweight, efficient model

            # Generate proper semantic embedding
            embedding_vector = model.encode(text, convert_to_numpy=True)

            # Ensure correct dimensions
            if len(embedding_vector) != dimensions:
                # Resize or truncate to match dimensions
                if len(embedding_vector) > dimensions:
                    embedding_vector = embedding_vector[:dimensions]
                else:
                    # Pad with zeros if needed
                    padding = np.zeros(dimensions - len(embedding_vector))
                    embedding_vector = np.concatenate([embedding_vector, padding])

            # Normalize to unit vector
            norm = np.linalg.norm(embedding_vector)
            if norm > 0:
                embedding_vector = embedding_vector / norm

            return VectorEmbedding(
                vector=embedding_vector.tolist(),
                embedding_type=embedding_type,
                dimensions=dimensions,
                created_at=datetime.now().isoformat(),
            )

        except ImportError:
            # Fallback to enhanced hash-based embedding if sentence-transformers not available
            logger.warning(
                "sentence-transformers not available, using enhanced hash-based embedding"
            )
            return self._generate_hash_embedding(text, embedding_type, dimensions)

    def _generate_hash_embedding(
        self, text: str, embedding_type: str, dimensions: int
    ) -> VectorEmbedding:
        """
        Enhanced hash-based embedding using multiple hash functions for variety.
        Fallback method when sentence-transformers is not available.
        """
        # Create deterministic embedding using multiple hash functions
        vector = []
        text_hash = hashlib.md5(text.encode()).hexdigest()

        for i in range(dimensions):
            # Use different hash algorithms for variety
            if i % 3 == 0:
                hash_val = hashlib.sha256(f"{text_hash}:{i}".encode()).hexdigest()
            elif i % 3 == 1:
                hash_val = hashlib.blake2b(f"{text_hash}:{i}".encode()).hexdigest()
            else:
                hash_val = hashlib.sha3_256(f"{text_hash}:{i}".encode()).hexdigest()

            # Convert hash to normalized float [-1, 1]
            val = int(hash_val[:8], 16) / 0xFFFFFFFF
            val = (val * 2) - 1  # Normalize to [-1, 1]
            vector.append(val)

        # L2 normalize to unit vector
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return VectorEmbedding(
            vector=vector,
            embedding_type=embedding_type,
            dimensions=dimensions,
            created_at=datetime.now().isoformat(),
        )

    def _cosine_similarity_dense(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two dense vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def add_memory_with_embedding(
        self,
        experiment_name: str,
        pattern_type: str,
        content: str,
        context: Optional[Dict[str, str]] = None,
        generate_embedding: bool = True,
    ) -> MemoryEntry:
        """
        Add a new memory with optional vector embedding for semantic search.

        Args:
            experiment_name: Name of the experiment
            pattern_type: Type of pattern ('success_pattern', 'failure_mode', 'strategy')
            content: The memory content
            context: Additional metadata
            generate_embedding: Whether to generate a vector embedding

        Returns:
            The created MemoryEntry
        """
        embedding = None
        if generate_embedding:
            text_to_embed = f"{content} {experiment_name} {pattern_type}"
            embedding = self._generate_embedding(
                text_to_embed, embedding_type=pattern_type
            )

        new_entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            experiment_name=experiment_name,
            pattern_type=pattern_type,
            content=content,
            context=context or {},
            embedding=embedding,
        )

        self.memory.append(new_entry)
        self._tfidf_dirty = True
        self._save_memory()
        logger.info(
            f"Added new memory [{pattern_type}] for {experiment_name} with embedding."
        )
        return new_entry

    def vector_semantic_search(
        self, query: str, top_k: int = 10, min_similarity: float = 0.3
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Advanced semantic search using dense vector embeddings with enhanced similarity metrics.
        Supports both proper semantic embeddings (when available) and hash-based fallbacks.

        Args:
            query: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (default: 0.3 for better precision)

        Returns:
            List of (MemoryEntry, similarity_score) tuples sorted by relevance
        """
        if not self.memory:
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        query_vector = query_embedding.vector

        # Score all memories with embeddings using multiple similarity metrics
        scored_memories = []
        for entry in self.memory:
            if entry.embedding and entry.embedding.vector:
                # Primary cosine similarity
                cosine_sim = self._cosine_similarity_dense(
                    query_vector, entry.embedding.vector
                )

                # Enhanced similarity metrics
                euclidean_dist = self._euclidean_distance(
                    query_vector, entry.embedding.vector
                )
                euclidean_sim = 1 / (1 + euclidean_dist)  # Convert to similarity

                # Combined score (weighted average)
                combined_score = 0.7 * cosine_sim + 0.3 * euclidean_sim

                if combined_score >= min_similarity:
                    scored_memories.append((entry, combined_score))

        # Sort by combined score (highest first) and return top_k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories[:top_k]

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        if len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        tfidf_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Hybrid search combining TF-IDF and vector embedding similarities.

        Args:
            query: Search query
            top_k: Number of results
            tfidf_weight: Weight for TF-IDF scores (0-1)
            vector_weight: Weight for vector embedding scores (0-1)

        Returns:
            Combined ranked results with composite scores
        """
        # Get TF-IDF results
        tfidf_results = self.semantic_search(query, top_k=top_k * 2)
        tfidf_scores = (
            {id(r): 1.0 - (i / len(tfidf_results)) for i, r in enumerate(tfidf_results)}
            if tfidf_results
            else {}
        )

        # Get vector results
        vector_results = self.vector_semantic_search(query, top_k=top_k * 2)
        vector_scores = {id(entry): score for entry, score in vector_results}

        # Combine scores
        all_entries = set(tfidf_results) | set(e for e, _ in vector_results)
        combined_scores = []

        for entry in all_entries:
            tfidf_score = tfidf_scores.get(id(entry), 0.0)
            vector_score = vector_scores.get(id(entry), 0.0)
            combined = (tfidf_weight * tfidf_score) + (vector_weight * vector_score)
            combined_scores.append((entry, combined))

        # Sort and return top_k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]

    def get_related_memories(
        self, memory_entry: MemoryEntry, top_k: int = 5
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Find memories related to a given entry using vector similarity.

        Args:
            memory_entry: The entry to find related memories for
            top_k: Number of related memories to return

        Returns:
            List of (MemoryEntry, similarity) tuples
        """
        if not memory_entry.embedding or not memory_entry.embedding.vector:
            # Generate embedding if missing
            text = f"{memory_entry.content} {memory_entry.experiment_name} {memory_entry.pattern_type}"
            memory_entry.embedding = self._generate_embedding(
                text, memory_entry.pattern_type
            )

        query_vector = memory_entry.embedding.vector

        scored = []
        for entry in self.memory:
            if entry.memory_id == memory_entry.memory_id:
                continue  # Skip the same entry

            if entry.embedding and entry.embedding.vector:
                sim = self._cosine_similarity_dense(
                    query_vector, entry.embedding.vector
                )
                scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def update_embeddings_for_all(self):
        """Generate embeddings for all memories that don't have them."""
        updated_count = 0
        for entry in self.memory:
            if not entry.embedding:
                text = f"{entry.content} {entry.experiment_name} {entry.pattern_type}"
                entry.embedding = self._generate_embedding(text, entry.pattern_type)
                updated_count += 1

        if updated_count > 0:
            self._save_memory()
            logger.info(f"Generated embeddings for {updated_count} memories.")

    # ------------------------------------------------------------------
    # TF-IDF Semantic Search
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _build_tfidf_index(self):
        """Build IDF values and per-document TF vectors."""
        if not self.memory:
            self._idf = {}
            self._doc_vectors = []
            self._tfidf_dirty = False
            return

        n_docs = len(self.memory)
        doc_freq: Counter[str] = Counter()
        tf_vectors: List[Dict[str, float]] = []

        for entry in self.memory:
            doc_text = f"{entry.content} {entry.experiment_name} {entry.pattern_type}"
            tokens = self._tokenize(doc_text)
            tf: Dict[str, float] = {}
            total = max(len(tokens), 1)
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            # Normalize TF
            for tok in tf:
                tf[tok] /= total
            tf_vectors.append(tf)
            # Unique terms in this doc
            for tok in set(tokens):
                doc_freq[tok] += 1

        # Compute IDF: log(N / (1 + df)) + 1  (smoothed)
        self._idf = {
            tok: math.log(n_docs / (1 + df)) + 1.0 for tok, df in doc_freq.items()
        }
        # Compute TF-IDF vectors
        self._doc_vectors = []
        for tf in tf_vectors:
            tfidf_vec = {
                tok: freq * self._idf.get(tok, 1.0) for tok, freq in tf.items()
            }
            self._doc_vectors.append(tfidf_vec)

        self._tfidf_dirty = False

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        common_keys = set(a.keys()) & set(b.keys())
        if not common_keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in common_keys)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def retrieve_memories(
        self, experiment_name: str, limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Retrieve the top-K most semantically relevant memories using TF-IDF similarity.

        Args:
            experiment_name: Experiment name
            limit: Number of results

        Returns:
            List of MemoryEntry objects sorted by relevance (most relevant first).
        """
        if self._tfidf_dirty:
            self._build_tfidf_index()

        if not self._doc_vectors:
            return []

        # Build query vector
        query_tokens = self._tokenize(experiment_name)
        if not query_tokens:
            return self.memory[:limit] if limit else self.memory[:5]

        query_tf: Dict[str, float] = {}
        total = len(query_tokens)
        for tok in query_tokens:
            query_tf[tok] = query_tf.get(tok, 0) + 1
        for tok in query_tf:
            query_tf[tok] /= total
        query_vec = {
            tok: freq * self._idf.get(tok, 1.0) for tok, freq in query_tf.items()
        }

        # Score all documents
        scored: List[tuple] = []
        for idx, doc_vec in enumerate(self._doc_vectors):
            sim = self._cosine_similarity(query_vec, doc_vec)
            scored.append((sim, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self.memory[idx] for _, idx in scored[:limit]]

    def retrieve_memories_by_filter(
        self, experiment_name: Optional[str] = None, pattern_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve memories by experiment name and/or pattern type.

        Args:
            experiment_name: Filter by experiment name
            pattern_type: Filter by pattern type (e.g., "success_pattern", "failure_mode", "strategy")

        Returns:
            List of matching MemoryEntry objects
        """
        filtered = self.memory

        if experiment_name is not None:
            filtered = [
                entry for entry in filtered if entry.experiment_name == experiment_name
            ]

        if pattern_type is not None:
            filtered = [
                entry for entry in filtered if entry.pattern_type == pattern_type
            ]

        return filtered

    def update_embeddings_for_all(self):
        """Generate embeddings for all memories that don't have them."""
        updated_count = 0
        for entry in self.memory:
            if not entry.embedding:
                text = f"{entry.content} {entry.experiment_name} {entry.pattern_type}"
                entry.embedding = self._generate_embedding(text, entry.pattern_type)
                updated_count += 1

        if updated_count > 0:
            self._save_memory()
            logger.info(f"Generated embeddings for {updated_count} memories.")


def update_memory_from_report(
    report_data: dict,
    memory_store: MemoryStore,
    llm_call_fn=None,
):
    """
    Extract lessons learned from an ExecutionReport and store them.

    If an LLM callable is provided (llm_call_fn), it will be used for
    nuanced pattern extraction. Otherwise, falls back to rule-based logic.

    Args:
        report_data: Dict from asdict(ExecutionReport).
        memory_store: MemoryStore instance.
        llm_call_fn: Optional callable(str) -> str for LLM-based extraction.
    """
    experiment_name = report_data.get("experiment_name", "unknown")

    # ---- LLM-based extraction (Phase 3 upgrade) ----
    if llm_call_fn is not None:
        try:
            extraction_prompt = (
                f"Analyze this experiment report and extract structured lessons.\n"
                f"Report: {json.dumps(report_data, indent=2)}\n\n"
                f"Return a JSON array where each element has:\n"
                f'  "pattern_type": one of "success_pattern", "failure_mode", "strategy"\n'
                f'  "content": a concise lesson (1-2 sentences)\n'
                f"If there are no meaningful lessons, return an empty array []."
            )
            llm_output = llm_call_fn(extraction_prompt)
            lessons = json.loads(llm_output)
            if isinstance(lessons, list):
                for lesson in lessons:
                    if (
                        isinstance(lesson, dict)
                        and "pattern_type" in lesson
                        and "content" in lesson
                    ):
                        memory_store.add_memory(
                            experiment_name=experiment_name,
                            pattern_type=lesson["pattern_type"],
                            content=lesson["content"],
                            context={"source": "llm_extraction"},
                        )
                return  # LLM extraction succeeded — skip rule-based fallback
        except Exception as e:
            logger.warning(
                f"LLM-based memory extraction failed, falling back to rules: {e}"
            )

    # ---- Rule-based fallback ----
    # Store root causes as failure modes
    root_causes = report_data.get("root_causes", [])
    for cause in root_causes:
        memory_store.add_memory(
            experiment_name=experiment_name,
            pattern_type="failure_mode",
            content=cause,
            context={"source": "auto_update_from_report"},
        )

    # Store suggested fixes as strategies
    suggested_fixes = report_data.get("suggested_fixes", [])
    for fix in suggested_fixes:
        memory_store.add_memory(
            experiment_name=experiment_name,
            pattern_type="strategy",
            content=fix,
            context={"source": "auto_update_from_report"},
        )

    # Analyze if it was a success based on positive metric deltas (simplified)
    metric_deltas = report_data.get("metric_deltas", {})
    is_success = any(
        val > 0 for val in metric_deltas.values() if isinstance(val, (int, float))
    )

    if is_success:
        memory_store.add_memory(
            experiment_name=experiment_name,
            pattern_type="success_pattern",
            content=f"Successful execution: {report_data.get('summary', '')}",
            context={"metric_deltas": str(metric_deltas)},
        )

    # Save to disk after adding memories
    memory_store._save_memory()
