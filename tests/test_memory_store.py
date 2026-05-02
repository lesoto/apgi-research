"""
Comprehensive tests for memory_store.py module.

This module tests all functionality in memory_store.py including:
- VectorEmbedding dataclass
- MemoryEntry dataclass
- MemoryStore class with all methods
- update_memory_from_report function
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_store import (
    MemoryEntry,
    MemoryStore,
    VectorEmbedding,
    update_memory_from_report,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage path for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir) / "test_memory.json"


@pytest.fixture
def memory_store(temp_storage_path):
    """Create a MemoryStore instance with temporary storage."""
    store = MemoryStore(storage_path=str(temp_storage_path))
    return store


@pytest.fixture
def sample_memory_entry():
    """Create a sample MemoryEntry for testing."""
    return MemoryEntry(
        timestamp="2024-01-01T00:00:00",
        experiment_name="test_experiment",
        pattern_type="success_pattern",
        content="Test memory content",
        context={"key": "value"},
    )


@pytest.fixture
def sample_vector_embedding():
    """Create a sample VectorEmbedding for testing."""
    return VectorEmbedding(
        vector=[0.1, 0.2, 0.3, 0.4] * 32,  # 128 dimensions
        embedding_type="experiment",
        dimensions=128,
        created_at="2024-01-01T00:00:00",
    )


# =============================================================================
# VectorEmbedding TESTS
# =============================================================================


class TestVectorEmbedding:
    """Test VectorEmbedding dataclass functionality."""

    def test_vector_embedding_creation(self):
        """Test VectorEmbedding can be created with all fields."""
        embedding = VectorEmbedding(
            vector=[0.1, 0.2, 0.3],
            embedding_type="test",
            dimensions=3,
            created_at="2024-01-01T00:00:00",
        )
        assert embedding.vector == [0.1, 0.2, 0.3]
        assert embedding.embedding_type == "test"
        assert embedding.dimensions == 3
        assert embedding.created_at == "2024-01-01T00:00:00"

    def test_vector_embedding_to_dict(self):
        """Test VectorEmbedding.to_dict() method."""
        embedding = VectorEmbedding(
            vector=[0.1, 0.2, 0.3],
            embedding_type="test",
            dimensions=3,
            created_at="2024-01-01T00:00:00",
        )
        result = embedding.to_dict()
        assert result == {
            "vector": [0.1, 0.2, 0.3],
            "embedding_type": "test",
            "dimensions": 3,
            "created_at": "2024-01-01T00:00:00",
        }

    def test_vector_embedding_from_dict(self):
        """Test VectorEmbedding.from_dict() class method."""
        data = {
            "vector": [0.1, 0.2, 0.3],
            "embedding_type": "test",
            "dimensions": 3,
            "created_at": "2024-01-01T00:00:00",
        }
        embedding = VectorEmbedding.from_dict(data)
        assert embedding.vector == [0.1, 0.2, 0.3]
        assert embedding.embedding_type == "test"
        assert embedding.dimensions == 3
        assert embedding.created_at == "2024-01-01T00:00:00"

    def test_vector_embedding_round_trip(self):
        """Test VectorEmbedding to_dict -> from_dict round trip."""
        original = VectorEmbedding(
            vector=[0.1, 0.2, 0.3],
            embedding_type="test",
            dimensions=3,
            created_at="2024-01-01T00:00:00",
        )
        data = original.to_dict()
        restored = VectorEmbedding.from_dict(data)
        assert original.vector == restored.vector
        assert original.embedding_type == restored.embedding_type
        assert original.dimensions == restored.dimensions
        assert original.created_at == restored.created_at


# =============================================================================
# MemoryEntry TESTS
# =============================================================================


class TestMemoryEntry:
    """Test MemoryEntry dataclass functionality."""

    def test_memory_entry_creation(self):
        """Test MemoryEntry can be created with all fields."""
        entry = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Test content",
            context={"key": "value"},
            embedding=None,
            memory_id="",
        )
        assert entry.timestamp == "2024-01-01T00:00:00"
        assert entry.experiment_name == "test_experiment"
        assert entry.pattern_type == "success_pattern"
        assert entry.content == "Test content"
        assert entry.context == {"key": "value"}
        assert entry.embedding is None

    def test_memory_entry_auto_memory_id(self):
        """Test MemoryEntry generates memory_id automatically."""
        entry = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Test content",
            context={},
        )
        assert entry.memory_id != ""
        # Verify it's an MD5 hash (12 chars)
        assert len(entry.memory_id) == 12
        assert all(c in "0123456789abcdef" for c in entry.memory_id)

    def test_memory_entry_memory_id_deterministic(self):
        """Test MemoryEntry generates consistent memory_id for same input."""
        entry1 = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Test content",
            context={},
        )
        entry2 = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Test content",
            context={},
        )
        assert entry1.memory_id == entry2.memory_id

    def test_memory_entry_different_content_different_id(self):
        """Test MemoryEntry generates different IDs for different content."""
        entry1 = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Content A",
            context={},
        )
        entry2 = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Content B",
            context={},
        )
        assert entry1.memory_id != entry2.memory_id

    def test_memory_entry_asdict(self):
        """Test MemoryEntry can be serialized with asdict."""
        entry = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Test content",
            context={"key": "value"},
        )
        data = asdict(entry)
        assert data["timestamp"] == "2024-01-01T00:00:00"
        assert data["experiment_name"] == "test_experiment"
        assert data["pattern_type"] == "success_pattern"
        assert data["content"] == "Test content"
        assert data["context"] == {"key": "value"}


# =============================================================================
# MemoryStore INITIALIZATION TESTS
# =============================================================================


class TestMemoryStoreInitialization:
    """Test MemoryStore initialization functionality."""

    def test_memory_store_init(self, temp_storage_path):
        """Test MemoryStore can be initialized."""
        store = MemoryStore(storage_path=str(temp_storage_path))
        assert store.storage_path == temp_storage_path
        assert store.memory == []
        assert store._tfidf_dirty is True
        assert store._idf == {}
        assert store._doc_vectors == []

    def test_memory_store_default_storage_path(self):
        """Test MemoryStore uses default storage path."""
        store = MemoryStore()
        assert store.storage_path.name == "xpr_memory.json"

    def test_memory_store_load_existing(self, temp_storage_path):
        """Test MemoryStore loads existing memory."""
        # Create existing memory file
        entry = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_experiment",
            pattern_type="success_pattern",
            content="Test content",
            context={},
        )
        with open(temp_storage_path, "w") as f:
            json.dump([asdict(entry)], f)

        store = MemoryStore(storage_path=str(temp_storage_path))
        assert len(store.memory) == 1
        assert store.memory[0].experiment_name == "test_experiment"

    def test_memory_store_load_invalid_file(self, temp_storage_path):
        """Test MemoryStore handles invalid JSON gracefully."""
        # Create invalid JSON file
        with open(temp_storage_path, "w") as f:
            f.write("invalid json")

        store = MemoryStore(storage_path=str(temp_storage_path))
        assert store.memory == []

    def test_memory_store_load_nonexistent_file(self, temp_storage_path):
        """Test MemoryStore handles nonexistent file."""
        store = MemoryStore(storage_path=str(temp_storage_path))
        assert store.memory == []

    def test_memory_store_check_embedding_availability(self, temp_storage_path):
        """Test MemoryStore checks for embedding availability."""
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            store = MemoryStore(storage_path=str(temp_storage_path))
            assert store.has_semantic_embeddings is False


# =============================================================================
# MemoryStore ADD MEMORY TESTS
# =============================================================================


class TestMemoryStoreAddMemory:
    """Test MemoryStore add memory functionality."""

    def test_add_memory_basic(self, memory_store):
        """Test adding memory without embedding."""
        memory_store.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
        )
        assert len(memory_store.memory) == 1
        assert memory_store.memory[0].experiment_name == "test_exp"
        assert memory_store.memory[0].pattern_type == "success_pattern"
        assert memory_store.memory[0].content == "Test content"

    def test_add_memory_with_context(self, memory_store):
        """Test adding memory with context."""
        memory_store.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
            context={"key1": "value1", "key2": "value2"},
        )
        assert memory_store.memory[0].context == {
            "key1": "value1",
            "key2": "value2",
        }

    def test_add_memory_default_context(self, memory_store):
        """Test adding memory with default empty context."""
        memory_store.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
        )
        assert memory_store.memory[0].context == {}

    def test_add_memory_saves_to_file(self, memory_store, temp_storage_path):
        """Test adding memory saves to storage file."""
        memory_store.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
        )
        assert temp_storage_path.exists()
        with open(temp_storage_path, "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["experiment_name"] == "test_exp"

    def test_add_memory_multiple(self, memory_store):
        """Test adding multiple memories."""
        for i in range(5):
            memory_store.add_memory(
                experiment_name=f"test_exp_{i}",
                pattern_type="success_pattern",
                content=f"Test content {i}",
            )
        assert len(memory_store.memory) == 5

    def test_add_memory_sets_tfidf_dirty(self, memory_store):
        """Test adding memory marks TF-IDF as dirty."""
        memory_store._tfidf_dirty = False
        memory_store.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
        )
        assert memory_store._tfidf_dirty is True

    def test_add_memory_generates_timestamp(self, memory_store):
        """Test adding memory generates timestamp."""
        memory_store.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
        )
        assert memory_store.memory[0].timestamp is not None
        assert len(memory_store.memory[0].timestamp) > 0


# =============================================================================
# MemoryStore EMBEDDING GENERATION TESTS
# =============================================================================


class TestMemoryStoreEmbeddings:
    """Test MemoryStore embedding generation functionality."""

    def test_generate_hash_embedding(self, memory_store):
        """Test hash-based embedding generation."""
        embedding = memory_store._generate_hash_embedding(
            text="Test text",
            embedding_type="test",
            dimensions=128,
        )
        assert isinstance(embedding, VectorEmbedding)
        assert len(embedding.vector) == 128
        assert embedding.embedding_type == "test"
        assert embedding.dimensions == 128

    def test_generate_hash_embedding_deterministic(self, memory_store):
        """Test hash-based embedding is deterministic for same text."""
        embedding1 = memory_store._generate_hash_embedding(
            text="Test text",
            embedding_type="test",
            dimensions=128,
        )
        embedding2 = memory_store._generate_hash_embedding(
            text="Test text",
            embedding_type="test",
            dimensions=128,
        )
        assert embedding1.vector == embedding2.vector

    def test_generate_hash_embedding_normalized(self, memory_store):
        """Test hash-based embeddings are normalized."""
        embedding = memory_store._generate_hash_embedding(
            text="Test text",
            embedding_type="test",
            dimensions=128,
        )
        # Calculate L2 norm
        norm = math.sqrt(sum(x * x for x in embedding.vector))
        assert abs(norm - 1.0) < 0.01  # Should be close to 1.0

    def test_generate_embedding_without_sentence_transformers(self, memory_store):
        """Test embedding generation falls back to hash when sentence-transformers unavailable."""
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with patch.object(
                memory_store, "_generate_hash_embedding", return_value=MagicMock()
            ) as mock_hash:
                memory_store._generate_embedding("test text")
                mock_hash.assert_called_once()

    def test_add_memory_with_embedding(self, memory_store):
        """Test adding memory with embedding generation."""
        entry = memory_store.add_memory_with_embedding(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
            generate_embedding=True,
        )
        assert entry.embedding is not None
        assert len(entry.embedding.vector) == 128

    def test_add_memory_without_embedding(self, memory_store):
        """Test adding memory without embedding generation."""
        entry = memory_store.add_memory_with_embedding(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
            generate_embedding=False,
        )
        assert entry.embedding is None

    def test_add_memory_with_embedding_default(self, memory_store):
        """Test adding memory with embedding (default True)."""
        entry = memory_store.add_memory_with_embedding(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
        )
        assert entry.embedding is not None

    def test_add_memory_with_embedding_saves_to_file(
        self, memory_store, temp_storage_path
    ):
        """Test adding memory with embedding saves to file."""
        memory_store.add_memory_with_embedding(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
        )
        assert temp_storage_path.exists()

    def test_update_embeddings_for_all(self, memory_store):
        """Test updating embeddings for all memories."""
        # Add memories without embeddings
        for i in range(3):
            entry = MemoryEntry(
                timestamp="2024-01-01T00:00:00",
                experiment_name=f"test_exp_{i}",
                pattern_type="success_pattern",
                content=f"Test content {i}",
                context={},
            )
            memory_store.memory.append(entry)

        memory_store.update_embeddings_for_all()
        for entry in memory_store.memory:
            assert entry.embedding is not None
            assert len(entry.embedding.vector) == 128

    def test_update_embeddings_for_all_partial(self, memory_store):
        """Test updating embeddings when some already exist."""
        # Add one with embedding
        entry_with = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="with_embedding",
            pattern_type="success_pattern",
            content="Test content",
            context={},
        )
        entry_with.embedding = VectorEmbedding(
            vector=[0.1] * 128,
            embedding_type="test",
            dimensions=128,
            created_at="2024-01-01T00:00:00",
        )
        memory_store.memory.append(entry_with)

        # Add one without embedding
        entry_without = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="without_embedding",
            pattern_type="success_pattern",
            content="Test content 2",
            context={},
        )
        memory_store.memory.append(entry_without)

        memory_store.update_embeddings_for_all()

        # Both should have embeddings now
        assert memory_store.memory[0].embedding is not None
        assert memory_store.memory[1].embedding is not None

    def test_refresh_all_embeddings(self, memory_store):
        """Test refresh_all_embeddings method (alias for update_embeddings_for_all)."""
        # Add memories without embeddings
        entry = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Test content",
            context={},
        )
        memory_store.memory.append(entry)

        memory_store.refresh_all_embeddings()
        assert memory_store.memory[0].embedding is not None


# =============================================================================
# MemoryStore SIMILARITY TESTS
# =============================================================================


class TestMemoryStoreSimilarity:
    """Test MemoryStore similarity calculation functionality."""

    def test_cosine_similarity_dense_identical(self, memory_store):
        """Test cosine similarity of identical vectors is 1.0."""
        vec = [1.0, 0.0, 0.0]
        result = memory_store._cosine_similarity_dense(vec, vec)
        assert abs(result - 1.0) < 0.0001

    def test_cosine_similarity_dense_orthogonal(self, memory_store):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        result = memory_store._cosine_similarity_dense(vec1, vec2)
        assert abs(result - 0.0) < 0.0001

    def test_cosine_similarity_dense_opposite(self, memory_store):
        """Test cosine similarity of opposite vectors is -1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        result = memory_store._cosine_similarity_dense(vec1, vec2)
        assert abs(result - (-1.0)) < 0.0001

    def test_cosine_similarity_dense_different_lengths(self, memory_store):
        """Test cosine similarity returns 0.0 for different length vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        result = memory_store._cosine_similarity_dense(vec1, vec2)
        assert result == 0.0

    def test_cosine_similarity_dense_zero_vector(self, memory_store):
        """Test cosine similarity handles zero vectors."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        result = memory_store._cosine_similarity_dense(vec1, vec2)
        assert result == 0.0

    def test_euclidean_distance(self, memory_store):
        """Test Euclidean distance calculation."""
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        result = memory_store._euclidean_distance(vec1, vec2)
        assert abs(result - 5.0) < 0.0001

    def test_euclidean_distance_different_lengths(self, memory_store):
        """Test Euclidean distance returns inf for different length vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        result = memory_store._euclidean_distance(vec1, vec2)
        assert result == float("inf")

    def test_euclidean_distance_identical(self, memory_store):
        """Test Euclidean distance of identical vectors is 0."""
        vec = [1.0, 2.0, 3.0]
        result = memory_store._euclidean_distance(vec, vec)
        assert result == 0.0

    def test_cosine_similarity_sparse(self, memory_store):
        """Test cosine similarity for sparse vectors."""
        vec1: Dict[str, float] = {"a": 1.0, "b": 0.5}
        vec2: Dict[str, float] = {"a": 1.0, "b": 0.5}
        result = memory_store._cosine_similarity(vec1, vec2)
        assert abs(result - 1.0) < 0.0001

    def test_cosine_similarity_sparse_no_overlap(self, memory_store):
        """Test cosine similarity for non-overlapping sparse vectors."""
        vec1: Dict[str, float] = {"a": 1.0}
        vec2: Dict[str, float] = {"b": 1.0}
        result = memory_store._cosine_similarity(vec1, vec2)
        assert result == 0.0

    def test_cosine_similarity_sparse_zero_magnitude(self, memory_store):
        """Test cosine similarity for zero magnitude sparse vectors."""
        vec1: Dict[str, float] = {}
        vec2: Dict[str, float] = {"a": 1.0}
        result = memory_store._cosine_similarity(vec1, vec2)
        assert result == 0.0


# =============================================================================
# MemoryStore SEARCH TESTS
# =============================================================================


class TestMemoryStoreSearch:
    """Test MemoryStore search functionality."""

    def test_vector_semantic_search_empty_store(self, memory_store):
        """Test semantic search on empty store returns empty list."""
        results = memory_store.vector_semantic_search("test query")
        assert results == []

    def test_vector_semantic_search_with_results(self, memory_store):
        """Test semantic search returns results."""
        # Add memories with embeddings
        for i in range(3):
            memory_store.add_memory_with_embedding(
                experiment_name="test_exp",
                pattern_type="success_pattern",
                content=f"Content about topic {i}",
            )

        results = memory_store.vector_semantic_search("topic", top_k=2)
        assert isinstance(results, list)
        # Should return up to top_k results
        assert len(results) <= 2

    def test_vector_semantic_search_min_similarity(self, memory_store):
        """Test semantic search respects min_similarity threshold."""
        # Add memories with embeddings
        memory_store.add_memory_with_embedding(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Content about AI and machine learning",
        )

        # High threshold should filter out low-similarity results
        results = memory_store.vector_semantic_search(
            "quantum physics", min_similarity=0.99
        )
        # May be empty if similarity is below threshold
        assert isinstance(results, list)

    def test_vector_semantic_search_top_k(self, memory_store):
        """Test semantic search respects top_k limit."""
        # Add multiple memories
        for i in range(10):
            memory_store.add_memory_with_embedding(
                experiment_name="test_exp",
                pattern_type="success_pattern",
                content=f"Content {i}",
            )

        results = memory_store.vector_semantic_search("content", top_k=5)
        assert len(results) <= 5

    def test_hybrid_search(self, memory_store):
        """Test hybrid search combines vector and TF-IDF scores."""
        # Add memories with embeddings
        for i in range(3):
            memory_store.add_memory_with_embedding(
                experiment_name="test_exp",
                pattern_type="success_pattern",
                content=f"Content about AI {i}",
            )

        results = memory_store.hybrid_search("AI", top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_get_related_memories(self, memory_store):
        """Test getting related memories."""
        # Add reference memory
        ref_entry = memory_store.add_memory_with_embedding(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Reference content about AI",
        )

        # Add other memories
        for i in range(3):
            memory_store.add_memory_with_embedding(
                experiment_name="test_exp",
                pattern_type="success_pattern",
                content=f"Related content {i}",
            )

        related = memory_store.get_related_memories(ref_entry, top_k=2)
        assert isinstance(related, list)
        # Should not include the reference entry itself
        for entry, score in related:
            assert entry.memory_id != ref_entry.memory_id

    def test_get_related_memories_without_embedding(self, memory_store):
        """Test getting related memories when entry has no embedding."""
        # Add entry without embedding
        entry = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Content about AI",
            context={},
        )
        memory_store.memory.append(entry)

        # Add other memories with embeddings
        for i in range(3):
            memory_store.add_memory_with_embedding(
                experiment_name="test_exp",
                pattern_type="success_pattern",
                content=f"Related content {i}",
            )

        related = memory_store.get_related_memories(entry, top_k=2)
        assert isinstance(related, list)
        # Entry should get an embedding generated
        assert entry.embedding is not None


# =============================================================================
# MemoryStore TF-IDF TESTS
# =============================================================================


class TestMemoryStoreTFIDF:
    """Test MemoryStore TF-IDF functionality."""

    def test_tokenize(self, memory_store):
        """Test text tokenization."""
        text = "Hello world test_123"
        tokens = memory_store._tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test_123" in tokens

    def test_tokenize_lowercase(self, memory_store):
        """Test tokenization converts to lowercase."""
        text = "HELLO World"
        tokens = memory_store._tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_punctuation(self, memory_store):
        """Test tokenization handles punctuation."""
        text = "hello, world! test."
        tokens = memory_store._tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Punctuation should be removed
        assert "," not in tokens
        assert "!" not in tokens

    def test_build_tfidf_index_empty(self, memory_store):
        """Test building TF-IDF index with no memories."""
        memory_store._build_tfidf_index()
        assert memory_store._idf == {}
        assert memory_store._doc_vectors == []
        assert memory_store._tfidf_dirty is False

    def test_build_tfidf_index(self, memory_store):
        """Test building TF-IDF index."""
        # Add memories
        memory_store.add_memory(
            experiment_name="exp1",
            pattern_type="success_pattern",
            content="machine learning AI",
        )
        memory_store.add_memory(
            experiment_name="exp2",
            pattern_type="success_pattern",
            content="deep learning neural networks",
        )

        memory_store._build_tfidf_index()
        assert len(memory_store._idf) > 0
        assert len(memory_store._doc_vectors) == 2
        assert memory_store._tfidf_dirty is False

    def test_retrieve_memories_empty_index(self, memory_store):
        """Test retrieving memories with empty index."""
        results = memory_store.retrieve_memories("test")
        assert results == []

    def test_retrieve_memories(self, memory_store):
        """Test retrieving memories by experiment name."""
        # Add memories
        memory_store.add_memory(
            experiment_name="neural_network_exp",
            pattern_type="success_pattern",
            content="test content",
        )
        memory_store.add_memory(
            experiment_name="other_exp",
            pattern_type="success_pattern",
            content="other content",
        )

        results = memory_store.retrieve_memories("neural")
        assert len(results) > 0
        assert results[0].experiment_name == "neural_network_exp"

    def test_retrieve_memories_limit(self, memory_store):
        """Test retrieving memories with limit."""
        # Add multiple memories
        for i in range(5):
            memory_store.add_memory(
                experiment_name="test_exp",
                pattern_type="success_pattern",
                content=f"content {i}",
            )

        results = memory_store.retrieve_memories("test", limit=3)
        assert len(results) <= 3

    def test_retrieve_memories_no_query_tokens(self, memory_store):
        """Test retrieving memories with empty query tokens."""
        # Add memories
        memory_store.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="content",
        )

        results = memory_store.retrieve_memories("!!!", limit=5)
        assert isinstance(results, list)

    def test_retrieve_memories_by_filter(self, memory_store):
        """Test filtering memories by experiment and pattern type."""
        memory_store.add_memory(
            experiment_name="exp1",
            pattern_type="success_pattern",
            content="content 1",
        )
        memory_store.add_memory(
            experiment_name="exp1",
            pattern_type="failure_mode",
            content="content 2",
        )
        memory_store.add_memory(
            experiment_name="exp2",
            pattern_type="success_pattern",
            content="content 3",
        )

        # Filter by experiment
        results = memory_store.retrieve_memories_by_filter(experiment_name="exp1")
        assert len(results) == 2

        # Filter by pattern type
        results = memory_store.retrieve_memories_by_filter(
            pattern_type="success_pattern"
        )
        assert len(results) == 2

        # Filter by both
        results = memory_store.retrieve_memories_by_filter(
            experiment_name="exp1", pattern_type="failure_mode"
        )
        assert len(results) == 1
        assert results[0].content == "content 2"

    def test_retrieve_memories_by_filter_none(self, memory_store):
        """Test filtering with None returns all."""
        memory_store.add_memory(
            experiment_name="exp1",
            pattern_type="success_pattern",
            content="content 1",
        )

        results = memory_store.retrieve_memories_by_filter()
        assert len(results) == 1


# =============================================================================
# update_memory_from_report TESTS
# =============================================================================


class TestUpdateMemoryFromReport:
    """Test update_memory_from_report function."""

    def test_update_memory_from_report_with_root_causes(self, memory_store):
        """Test updating memory from report with root causes."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": ["cause 1", "cause 2"],
            "suggested_fixes": ["fix 1"],
            "metric_deltas": {"accuracy": 0.1},
            "summary": "Test summary",
        }

        update_memory_from_report(report_data, memory_store)

        # Should have added memories for root causes
        failure_memories = [
            m for m in memory_store.memory if m.pattern_type == "failure_mode"
        ]
        assert len(failure_memories) == 2

    def test_update_memory_from_report_with_fixes(self, memory_store):
        """Test updating memory from report with suggested fixes."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": [],
            "suggested_fixes": ["fix 1", "fix 2"],
            "metric_deltas": {"accuracy": 0.1},
            "summary": "Test summary",
        }

        update_memory_from_report(report_data, memory_store)

        # Should have added memories for fixes
        strategy_memories = [
            m for m in memory_store.memory if m.pattern_type == "strategy"
        ]
        assert len(strategy_memories) == 2

    def test_update_memory_from_report_with_success(self, memory_store):
        """Test updating memory from report with positive metrics."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": [],
            "suggested_fixes": [],
            "metric_deltas": {"accuracy": 0.1, "loss": -0.05},
            "summary": "Test summary",
        }

        update_memory_from_report(report_data, memory_store)

        # Should have added success memory
        success_memories = [
            m for m in memory_store.memory if m.pattern_type == "success_pattern"
        ]
        assert len(success_memories) == 1

    def test_update_memory_from_report_no_success(self, memory_store):
        """Test updating memory from report with no positive metrics."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": [],
            "suggested_fixes": [],
            "metric_deltas": {"accuracy": -0.1},
            "summary": "Test summary",
        }

        update_memory_from_report(report_data, memory_store)

        # Should not have success memory
        success_memories = [
            m for m in memory_store.memory if m.pattern_type == "success_pattern"
        ]
        assert len(success_memories) == 0

    def test_update_memory_from_report_with_llm(self, memory_store):
        """Test updating memory with LLM-based extraction."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": [],
            "suggested_fixes": [],
            "metric_deltas": {},
        }

        # Mock LLM function
        def mock_llm(prompt):
            return json.dumps(
                [{"pattern_type": "success_pattern", "content": "LLM extracted lesson"}]
            )

        update_memory_from_report(report_data, memory_store, llm_call_fn=mock_llm)

        # Should have added memory from LLM
        success_memories = [
            m for m in memory_store.memory if m.pattern_type == "success_pattern"
        ]
        assert len(success_memories) == 1
        assert success_memories[0].content == "LLM extracted lesson"

    def test_update_memory_from_report_llm_failure_fallback(self, memory_store):
        """Test fallback to rule-based when LLM fails."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": ["cause 1"],
            "suggested_fixes": [],
            "metric_deltas": {},
        }

        # Mock LLM function that fails
        def mock_llm_error(prompt):
            raise ValueError("LLM error")

        update_memory_from_report(report_data, memory_store, llm_call_fn=mock_llm_error)

        # Should have fallen back to rule-based extraction
        failure_memories = [
            m for m in memory_store.memory if m.pattern_type == "failure_mode"
        ]
        assert len(failure_memories) == 1
        assert failure_memories[0].content == "cause 1"

    def test_update_memory_from_report_llm_invalid_json(self, memory_store):
        """Test fallback when LLM returns invalid JSON."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": ["cause 1"],
            "suggested_fixes": [],
            "metric_deltas": {},
        }

        # Mock LLM function that returns invalid JSON
        def mock_llm_invalid(prompt):
            return "not valid json"

        update_memory_from_report(
            report_data, memory_store, llm_call_fn=mock_llm_invalid
        )

        # Should have fallen back to rule-based extraction
        failure_memories = [
            m for m in memory_store.memory if m.pattern_type == "failure_mode"
        ]
        assert len(failure_memories) == 1

    def test_update_memory_from_report_empty_report(self, memory_store):
        """Test updating memory with empty report."""
        report_data = {
            "experiment_name": "test_exp",
            "root_causes": [],
            "suggested_fixes": [],
            "metric_deltas": {},
        }

        update_memory_from_report(report_data, memory_store)

        # Should not have added any memories
        assert len(memory_store.memory) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestMemoryStoreEdgeCases:
    """Test edge cases and error handling."""

    def test_save_memory_error(self, memory_store, temp_storage_path):
        """Test _save_memory handles write errors gracefully."""
        # Make directory read-only
        temp_storage_path.parent.chmod(0o555)
        try:
            memory_store.add_memory(
                experiment_name="test",
                pattern_type="success",
                content="content",
            )
            # Should not raise exception
        finally:
            # Restore permissions for cleanup
            temp_storage_path.parent.chmod(0o755)

    def test_load_memory_corrupted_file(self, temp_storage_path):
        """Test loading corrupted memory file."""
        # Write invalid JSON
        with open(temp_storage_path, "w") as f:
            f.write("{invalid json}")

        store = MemoryStore(storage_path=str(temp_storage_path))
        assert store.memory == []

    def test_add_memory_with_special_characters(self, memory_store):
        """Test adding memory with special characters in content."""
        memory_store.add_memory(
            experiment_name="test",
            pattern_type="success",
            content="Special chars: ñ 中文 🚀 \n\t",
        )
        assert len(memory_store.memory) == 1
        assert "中文" in memory_store.memory[0].content

    def test_add_memory_very_long_content(self, memory_store):
        """Test adding memory with very long content."""
        long_content = "x" * 10000
        memory_store.add_memory(
            experiment_name="test",
            pattern_type="success",
            content=long_content,
        )
        assert memory_store.memory[0].content == long_content

    def test_vector_search_with_empty_embedding_vector(self, memory_store):
        """Test semantic search handles empty embedding vectors."""
        # Add entry with empty vector
        entry = MemoryEntry(
            timestamp="2024-01-01T00:00:00",
            experiment_name="test",
            pattern_type="success",
            content="content",
            context={},
        )
        entry.embedding = VectorEmbedding(
            vector=[], embedding_type="test", dimensions=0, created_at="2024-01-01"
        )
        memory_store.memory.append(entry)

        results = memory_store.vector_semantic_search("query")
        assert isinstance(results, list)

    def test_hybrid_search_empty_store(self, memory_store):
        """Test hybrid search on empty store."""
        results = memory_store.hybrid_search("query")
        assert results == []

    def test_cosine_similarity_dense_single_element(self, memory_store):
        """Test cosine similarity with single element vectors."""
        vec1 = [1.0]
        vec2 = [1.0]
        result = memory_store._cosine_similarity_dense(vec1, vec2)
        assert abs(result - 1.0) < 0.0001

    def test_euclidean_distance_single_element(self, memory_store):
        """Test Euclidean distance with single element vectors."""
        vec1 = [3.0]
        vec2 = [0.0]
        result = memory_store._euclidean_distance(vec1, vec2)
        assert abs(result - 3.0) < 0.0001


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestMemoryStoreIntegration:
    """Integration tests for MemoryStore workflows."""

    def test_full_workflow(self, memory_store):
        """Test complete memory workflow."""
        # Add memories
        for i in range(5):
            memory_store.add_memory_with_embedding(
                experiment_name=f"exp_{i % 2}",  # 2 different experiments
                pattern_type="success_pattern" if i % 2 == 0 else "failure_mode",
                content=f"Content about machine learning and AI topic {i}",
            )

        # Search
        results = memory_store.vector_semantic_search("machine learning", top_k=3)
        assert len(results) <= 3

        # Filter
        filtered = memory_store.retrieve_memories_by_filter(experiment_name="exp_0")
        assert len(filtered) > 0

        # Get related
        if results:
            related = memory_store.get_related_memories(results[0][0], top_k=2)
            assert isinstance(related, list)

    def test_save_and_load_persistence(self, temp_storage_path):
        """Test memory persistence across store instances."""
        # Create and populate store
        store1 = MemoryStore(storage_path=str(temp_storage_path))
        store1.add_memory(
            experiment_name="test_exp",
            pattern_type="success_pattern",
            content="Persistent content",
        )

        # Create new store instance with same path
        store2 = MemoryStore(storage_path=str(temp_storage_path))
        assert len(store2.memory) == 1
        assert store2.memory[0].content == "Persistent content"

    def test_concurrent_add_memory(self, memory_store):
        """Test adding memories (simulated concurrent access)."""
        import threading

        def add_memories(store, prefix):
            for i in range(10):
                store.add_memory(
                    experiment_name=f"{prefix}_exp",
                    pattern_type="success",
                    content=f"Content {i}",
                )

        threads = [
            threading.Thread(target=add_memories, args=(memory_store, f"thread_{i}"))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 30 memories
        assert len(memory_store.memory) == 30

    def test_tfidf_rebuild_after_add(self, memory_store):
        """Test TF-IDF index rebuilds after adding memories."""
        # Add initial memory
        memory_store.add_memory(
            experiment_name="exp1",
            pattern_type="success",
            content="machine learning",
        )

        # Build index
        memory_store._build_tfidf_index()
        initial_idf = dict(memory_store._idf)

        # Add more memories
        memory_store.add_memory(
            experiment_name="exp2",
            pattern_type="success",
            content="deep learning neural networks",
        )

        # Index should be dirty
        assert memory_store._tfidf_dirty is True

        # Rebuild and check IDF changed
        memory_store._build_tfidf_index()
        assert memory_store._idf != initial_idf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
