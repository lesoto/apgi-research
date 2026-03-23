"""
Test suite for prepare.py module.

Tests data preparation functionality including downloading and tokenizer training.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path to import the module
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the external dependencies before importing
sys.modules["requests"] = MagicMock()
sys.modules["pyarrow.parquet"] = MagicMock()
sys.modules["rustbpe"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["torch"] = MagicMock()

import prepare as prep


class TestConstants:
    """Test module constants."""

    def test_max_seq_len(self):
        """Test MAX_SEQ_LEN constant."""
        assert prep.MAX_SEQ_LEN == 2048

    def test_time_budget(self):
        """Test TIME_BUDGET constant."""
        assert prep.TIME_BUDGET == 600

    def test_eval_tokens(self):
        """Test EVAL_TOKENS constant."""
        assert prep.EVAL_TOKENS == 40 * 524288

    def test_vocab_size(self):
        """Test VOCAB_SIZE constant."""
        assert prep.VOCAB_SIZE == 8192

    def test_cache_dir(self):
        """Test CACHE_DIR constant."""
        expected = Path.home() / ".cache" / "autoresearch"
        assert prep.CACHE_DIR == expected

    def test_data_dir(self):
        """Test DATA_DIR constant."""
        expected = prep.CACHE_DIR / "data"
        assert prep.DATA_DIR == expected

    def test_tokenizer_dir(self):
        """Test TOKENIZER_DIR constant."""
        expected = prep.CACHE_DIR / "tokenizer"
        assert prep.TOKENIZER_DIR == expected

    def test_base_url(self):
        """Test BASE_URL constant."""
        expected = (
            "https://huggingface.co/datasets/karpathy/climbix-400b-shuffle/resolve/main"
        )
        assert prep.BASE_URL == expected

    def test_max_shard(self):
        """Test MAX_SHARD constant."""
        assert prep.MAX_SHARD == 6542

    def test_val_shard(self):
        """Test VAL_SHARD constant."""
        assert prep.VAL_SHARD == 6542

    def test_val_filename(self):
        """Test VAL_FILENAME constant."""
        assert prep.VAL_FILENAME == "shard_06542.parquet"


class TestDownloadFunction:
    """Test download functionality."""

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_file_success(self, mock_mkdir, mock_exists, mock_get):
        """Test successful file download."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-length": "1000"}
        mock_response.iter_content.return_value = [b"data chunk"]
        mock_get.return_value = mock_response

        # Mock file operations
        with patch("builtins.open", mock_open()):
            with patch("tqdm.tqdm") as mock_tqdm:
                mock_tqdm.return_value.__enter__.return_value = mock_tqdm
                mock_tqdm.return_value.__iter__.return_value = iter([b"data chunk"])

                result = prep.download_file(
                    "http://example.com/file", Path("/tmp/file")
                )
                assert result is True

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    def test_download_file_already_exists(self, mock_exists, mock_get):
        """Test download when file already exists."""
        mock_exists.return_value = True

        result = prep.download_file("http://example.com/file", Path("/tmp/file"))
        assert result is True
        mock_get.assert_not_called()

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_file_http_error(self, mock_mkdir, mock_exists, mock_get):
        """Test download with HTTP error."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        result = prep.download_file("http://example.com/file", Path("/tmp/file"))
        assert result is False

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_shard(self, mock_mkdir, mock_exists):
        """Test downloading a shard."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_file", return_value=True) as mock_download:
            result = prep.download_shard(1, prep.DATA_DIR)
            assert result is True
            mock_download.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_shard_failure(self, mock_mkdir, mock_exists):
        """Test shard download failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_file", return_value=False) as mock_download:
            result = prep.download_shard(1, prep.DATA_DIR)
            assert result is False
            mock_download.assert_called_once()


class TestTokenizerTraining:
    """Test tokenizer training functionality."""

    @patch("prepare.rustbpe.Trainer")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_train_tokenizer_success(self, mock_mkdir, mock_exists, mock_trainer):
        """Test successful tokenizer training."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.load.return_value = None
        mock_trainer_instance.train.return_value = None

        # Mock file operations
        with patch("builtins.open", mock_open()):
            result = prep.train_tokenizer(prep.DATA_DIR, prep.TOKENIZER_DIR)
            assert result is True

    @patch("prepare.Path.exists")
    def test_train_tokenizer_already_exists(self, mock_exists):
        """Test training when tokenizer already exists."""
        # Mock tokenizer file exists
        mock_exists.return_value = True

        result = prep.train_tokenizer(prep.DATA_DIR, prep.TOKENIZER_DIR)
        assert result is True

    @patch("prepare.rustbpe.Trainer")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_train_tokenizer_failure(self, mock_mkdir, mock_exists, mock_trainer):
        """Test tokenizer training failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        mock_trainer.side_effect = Exception("Training failed")

        result = prep.train_tokenizer(prep.DATA_DIR, prep.TOKENIZER_DIR)
        assert result is False


class TestDataProcessing:
    """Test data processing functionality."""

    @patch("prepare.pyarrow.parquet.read_table")
    @patch("prepare.Path.exists")
    def test_read_shard_success(self, mock_exists, mock_read):
        """Test successful shard reading."""
        mock_exists.return_value = True

        # Mock parquet table
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = MagicMock()
        mock_read.return_value = mock_table

        result = prep.read_shard(1, prep.DATA_DIR)
        assert result is not None

    @patch("prepare.Path.exists")
    def test_read_shard_not_exists(self, mock_exists):
        """Test reading non-existent shard."""
        mock_exists.return_value = False

        result = prep.read_shard(1, prep.DATA_DIR)
        assert result is None

    @patch("prepare.pyarrow.parquet.read_table")
    @patch("prepare.Path.exists")
    def test_read_shard_error(self, mock_exists, mock_read):
        """Test shard reading with error."""
        mock_exists.return_value = True
        mock_read.side_effect = Exception("Read error")

        result = prep.read_shard(1, prep.DATA_DIR)
        assert result is None

    def test_tokenize_text(self):
        """Test text tokenization."""
        with patch("prepare.rustbpe.Encoder") as mock_encoder:
            mock_encoder_instance = MagicMock()
            mock_encoder_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_encoder.return_value = mock_encoder_instance

            result = prep.tokenize_text("test text", mock_encoder_instance)
            assert result == [1, 2, 3, 4, 5]

    def test_tokenize_batch(self):
        """Test batch tokenization."""
        with patch("prepare.rustbpe.Encoder") as mock_encoder:
            mock_encoder_instance = MagicMock()
            mock_encoder_instance.encode.return_value = [1, 2, 3]
            mock_encoder.return_value = mock_encoder_instance

            texts = ["text1", "text2", "text3"]
            result = prep.tokenize_batch(texts, mock_encoder_instance)
            assert len(result) == 3
            assert all(isinstance(tokens, list) for tokens in result)


class TestValidation:
    """Test validation functionality."""

    @patch("prepare.tiktoken.get_encoding")
    def test_validate_tokenizer_success(self, mock_get_encoding):
        """Test successful tokenizer validation."""
        mock_encoding = MagicMock()
        mock_encoding.decode.return_value = "test text"
        mock_get_encoding.return_value = mock_encoding

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "test text"

        result = prep.validate_tokenizer(mock_tokenizer)
        assert result is True

    @patch("prepare.tiktoken.get_encoding")
    def test_validate_tokenizer_failure(self, mock_get_encoding):
        """Test tokenizer validation failure."""
        mock_get_encoding.side_effect = Exception("Validation failed")

        mock_tokenizer = MagicMock()

        result = prep.validate_tokenizer(mock_tokenizer)
        assert result is False

    def test_validate_data_format(self):
        """Test data format validation."""
        # Mock valid data
        valid_data = [{"text": "Sample text 1"}, {"text": "Sample text 2"}]

        result = prep.validate_data_format(valid_data)
        assert result is True

    def test_validate_data_format_invalid(self):
        """Test invalid data format validation."""
        # Mock invalid data
        invalid_data = [
            {"content": "Sample text 1"},  # Wrong key
            {"text": "Sample text 2"},
        ]

        result = prep.validate_data_format(invalid_data)
        assert result is False


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_parse_args_defaults(self):
        """Test default argument parsing."""
        args = prep.parse_args([])
        assert args.num_shards is None
        assert args.skip_download is False
        assert args.skip_tokenizer is False
        assert args.force is False

    def test_parse_args_custom_values(self):
        """Test custom argument parsing."""
        args = prep.parse_args(
            ["--num-shards", "10", "--skip-download", "--skip-tokenizer", "--force"]
        )
        assert args.num_shards == 10
        assert args.skip_download is True
        assert args.skip_tokenizer is True
        assert args.force is True

    def test_parse_args_num_shards_validation(self):
        """Test num_shards validation."""
        # Valid num_shards
        args = prep.parse_args(["--num-shards", "100"])
        assert args.num_shards == 100

        # Invalid num_shards (should still parse, validation happens later)
        args = prep.parse_args(["--num-shards", "10000"])
        assert args.num_shards == 10000


class TestMainFunction:
    """Test main function execution."""

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_full_prep_success(self, mock_mkdir, mock_exists):
        """Test successful full preparation."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_shards", return_value=True) as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True) as mock_train:
                with patch.object(
                    prep, "validate_tokenizer", return_value=True
                ) as mock_validate:
                    result = prep.main([])
                    assert result == 0
                    mock_download.assert_called_once()
                    mock_train.assert_called_once()
                    mock_validate.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_download_only(self, mock_mkdir, mock_exists):
        """Test download only mode."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_shards", return_value=True) as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True) as mock_train:
                result = prep.main(["--skip-tokenizer"])
                assert result == 0
                mock_download.assert_called_once()
                mock_train.assert_not_called()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_tokenizer_only(self, mock_mkdir, mock_exists):
        """Test tokenizer only mode."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_shards", return_value=True) as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True) as mock_train:
                result = prep.main(["--skip-download"])
                assert result == 0
                mock_download.assert_not_called()
                mock_train.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_download_failure(self, mock_mkdir, mock_exists):
        """Test main function with download failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_shards", return_value=False) as mock_download:
            result = prep.main([])
            assert result == 1
            mock_download.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_tokenizer_failure(self, mock_mkdir, mock_exists):
        """Test main function with tokenizer failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_shards", return_value=True) as mock_download:
            with patch.object(
                prep, "train_tokenizer", return_value=False
            ) as mock_train:
                result = prep.main([])
                assert result == 1
                mock_download.assert_called_once()
                mock_train.assert_called_once()

    def test_main_custom_shards(self):
        """Test main function with custom number of shards."""
        with patch.object(prep, "download_shards", return_value=True) as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True):
                with patch.object(prep, "validate_tokenizer", return_value=True):
                    result = prep.main(["--num-shards", "5"])
                    assert result == 0
                    mock_download.assert_called_once_with(5, prep.DATA_DIR)


class TestMultiprocessing:
    """Test multiprocessing functionality."""

    def test_download_worker(self):
        """Test download worker function."""
        with patch.object(prep, "download_shard", return_value=True) as mock_download:
            result = prep.download_worker((1, prep.DATA_DIR))
            assert result == (1, True)
            mock_download.assert_called_once_with(1, prep.DATA_DIR)

    def test_download_worker_failure(self):
        """Test download worker with failure."""
        with patch.object(prep, "download_shard", return_value=False) as mock_download:
            result = prep.download_worker((1, prep.DATA_DIR))
            assert result == (1, False)
            mock_download.assert_called_once_with(1, prep.DATA_DIR)

    @patch("prepare.Pool")
    def test_download_shards_parallel(self, mock_pool):
        """Test parallel shard downloading."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, True), (3, True)]

        result = prep.download_shards_parallel([1, 2, 3], prep.DATA_DIR)
        assert result is True
        mock_pool_instance.map.assert_called_once()

    @patch("prepare.Pool")
    def test_download_shards_parallel_failure(self, mock_pool):
        """Test parallel shard downloading with failure."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, False), (3, True)]

        result = prep.download_shards_parallel([1, 2, 3], prep.DATA_DIR)
        assert result is False


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        with patch("prepare.Path.exists") as mock_exists:
            with patch("prepare.Path.stat") as mock_stat:
                mock_exists.return_value = True
                mock_stat.return_value.st_size = 1000000
                mock_stat.return_value.st_mtime = 1234567890

                info = prep.get_cache_info()
                assert isinstance(info, dict)
                assert "size_bytes" in info
                assert "last_modified" in info

    def test_get_cache_info_not_exists(self):
        """Test cache info when cache doesn't exist."""
        with patch("prepare.Path.exists") as mock_exists:
            mock_exists.return_value = False

            info = prep.get_cache_info()
            assert isinstance(info, dict)
            assert info["size_bytes"] == 0

    def test_cleanup_cache(self):
        """Test cache cleanup."""
        with patch("prepare.shutil.rmtree") as mock_rmtree:
            with patch("prepare.Path.exists") as mock_exists:
                mock_exists.return_value = True

                result = prep.cleanup_cache()
                assert result is True
                mock_rmtree.assert_called()

    def test_cleanup_cache_not_exists(self):
        """Test cache cleanup when cache doesn't exist."""
        with patch("prepare.Path.exists") as mock_exists:
            mock_exists.return_value = False

            result = prep.cleanup_cache()
            assert result is True


if __name__ == "__main__":
    pytest.main([__file__])
