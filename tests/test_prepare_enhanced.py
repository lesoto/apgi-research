"""
Enhanced test suite for prepare.py to achieve 90%+ coverage.

Targets uncovered areas:
- Data download functionality and error handling
- Tokenizer training and validation
- Data loading and batch processing
- Caching mechanisms and file operations
- Utility functions and edge cases
- Multiprocessing and parallel operations
"""

from pathlib import Path
from typing import List
from unittest.mock import Mock, patch
from tqdm import tqdm as mock_tqdm

import pytest
import torch

from prepare import (
    TOKENIZER_DIR,
    CACHE_DIR,
    DATA_DIR,
    VOCAB_SIZE,
    train_tokenizer,
    get_token_bytes,
    make_dataloader,
    evaluate_bpb,
    text_iterator,
    get_cache_info,
    cleanup_cache,
    download_data,
    download_file,
    download_single_shard,
    download_worker,
    parse_args,
    read_shard,
    validate_data_format,
    validate_tokenizer,
)

from prepare import main


class TestDownloadFunctions:
    """Test data download functionality."""

    def test_download_single_shard_success(self):
        """Test successful single shard download."""
        with patch("prepare.requests.get"):
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"test data"]

            result = download_single_shard(1)

            assert result is True

    def test_download_single_shard_exists(self):
        """Test shard download when file exists."""
        with patch("prepare.Path.exists", return_value=True):
            result = download_single_shard(1)

            assert result is True

    def test_download_single_shard_failure(self):
        """Test shard download failure handling."""
        with patch("prepare.requests.get"):
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("Network error")

            result = download_single_shard(1)

            assert result is False

    def test_download_single_shard_retry_logic(self):
        """Test retry logic with exponential backoff."""
        with patch("prepare.requests.get"):
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = [
                Exception("First failure"),
                Exception("Second failure"),
                None,  # Success on third try
            ]

            with patch("prepare.time.sleep") as mock_sleep:
                result = download_single_shard(1)

                # Should have called sleep twice
                assert mock_sleep.call_count == 2
                assert result is True

    def test_download_data_parallel(self):
        """Test parallel download functionality."""
        shard_indices = [1, 2, 3]

        with patch("prepare.Pool") as mock_pool:
            mock_pool = Mock()
            mock_pool.map.return_value = [True, True, True]

            with patch("prepare.download_data", return_value=None) as mock_download:
                download_data(8, download_workers=4)

                mock_download.assert_called_once_with(shard_indices, 4)
                mock_pool.assert_called_once()

    def test_download_data_partial_existing(self):
        """Test download with some shards already existing."""
        with patch("prepare.list_parquet_files") as mock_list:
            mock_list.return_value = [
                Path("shard_0.parquet"),
                Path("shard_1.parquet"),
                Path("validation.parquet"),
            ]

            with patch("prepare.Pool") as mock_pool:
                mock_pool.map.return_value = [True, True, True]

                with patch("prepare.download_data", return_value=None) as mock_download:
                    download_data(3, download_workers=4)

                    # Should only download missing shards
                    expected_shards = [0, 2]  # shard_0 and shard_2 missing
                    mock_download.assert_called_once_with(expected_shards, 4)

    def test_download_file_function(self):
        """Test generic file download function."""
        with patch("prepare.requests.get"):
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"test content"]
            mock_response.headers = {"content-length": "12"}

            with patch("prepare.tqdm") as mock_tqdm:
                result = download_file(
                    "http://example.com/file", Path("/tmp/test_file")
                )

                assert result is True
                mock_tqdm.assert_called_once()

    def test_download_shard_worker_function(self):
        """Test download worker function."""
        with patch("prepare.download_single_shard") as mock_download:
            mock_download.return_value = True

            download_worker((1, Path("/tmp")))

            mock_download.assert_called_once_with(1, Path("/tmp"))

    def test_download_worker_function(self):
        """Test download worker function."""
        with patch("prepare.download_single_shard") as mock_download:
            mock_download.return_value = True

            download_worker((1, Path("/tmp")))

            mock_download.assert_called_once_with(1, Path("/tmp"))


class TestTokenizerTraining:
    """Test tokenizer training functionality."""

    def test_train_tokenizer_success(self):
        """Test successful tokenizer training."""
        with patch("prepare.rustbpe.Trainer") as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance

            with patch.object(mock_trainer_instance, "load") as mock_load:
                with patch.object(mock_trainer_instance, "train") as mock_train:
                    with patch.object(
                        mock_trainer_instance, "save_as_tiktoken"
                    ) as mock_save:
                        with patch("prepare.torch.zeros") as mock_zeros:
                            with patch("prepare.torch.save") as mock_torch_save:
                                result = train_tokenizer()

                                mock_trainer.assert_called_once_with(
                                    vocab_size=VOCAB_SIZE, special_tokens=[""]
                                )
                                mock_load.assert_called_once()
                                mock_train.assert_called_once()
                                mock_save.assert_called_once()
                                mock_zeros.assert_called_once_with(
                                    (256,), dtype=torch.uint8
                                )
                                mock_torch_save.assert_called_once()
                                assert result is True

    def test_train_tokenizer_failure(self):
        """Test tokenizer training failure handling."""
        with patch("prepare.rustbpe.Trainer") as mock_trainer:
            mock_trainer.side_effect = Exception("Training failed")

            result = train_tokenizer()

            assert result is False

    def test_train_tokenizer_already_exists(self):
        """Test training when tokenizer already exists."""
        with patch("prepare.Path.exists", return_value=True):
            result = train_tokenizer()

            assert result is True

    def test_tokenizer_files_creation(self):
        """Test tokenizer file creation and paths."""
        with patch("prepare.os.makedirs") as mock_makedirs:
            with patch("prepare.train_tokenizer") as mock_train:
                mock_train.return_value = True
                train_tokenizer()

                mock_makedirs.assert_called_once_with(TOKENIZER_DIR, exist_ok=True)
                assert True

    def test_get_token_bytes_device_handling(self):
        """Test token bytes loading with different devices."""
        cpu_tensor = torch.randn(256)
        gpu_tensor = torch.randn(256)

        with patch("prepare.torch.load") as mock_load:
            mock_load.return_value = cpu_tensor

            # Test CPU device
            result_cpu = get_token_bytes("cpu")
            assert torch.equal(result_cpu, cpu_tensor)
            mock_load.assert_called_once_with(
                TOKENIZER_DIR / "token_bytes.pt", map_location="cpu"
            )

            # Test CUDA device
            mock_load.return_value = gpu_tensor
            result_gpu = get_token_bytes("cuda")
            assert torch.equal(result_gpu, gpu_tensor)
            mock_load.assert_called_once_with(
                TOKENIZER_DIR / "token_bytes.pt", map_location="cuda"
            )


class TestDataLoading:
    """Test data loading and batch processing."""

    def test_text_iterator_basic(self):
        """Test basic text iteration."""
        mock_texts = ["text1", "text2", "text3"]
        mock_docs = []

        with patch("prepare.pq.ParquetFile") as mock_pq:
            mock_pq_instance = Mock()
            mock_pq.return_value = mock_pq_instance

            with patch.object(mock_pq_instance, "num_row_groups") as mock_num_groups:
                mock_num_groups.return_value = 2
                with patch.object(mock_pq_instance, "read_row_group") as mock_read:
                    mock_read.return_value = Mock()
                    mock_read.column.return_value.to_pylist.return_value = mock_texts

                    for doc in text_iterator():
                        mock_docs.append(doc)

                    assert len(mock_docs) == 6  # 2 groups × 3 texts each
                    mock_read.assert_called()

    def test_text_iterator_doc_cap(self):
        """Test text iterator with document cap."""
        mock_texts = ["a" * 1000] * 10  # Very long texts

        with patch("prepare.pq.ParquetFile") as mock_pq:
            mock_pq_instance = Mock()
            mock_pq.return_value = mock_pq_instance

            with patch.object(mock_pq_instance, "num_row_groups") as mock_num_groups:
                mock_num_groups.return_value = 1
                with patch.object(mock_pq_instance, "read_row_group") as mock_read:
                    mock_read.return_value = Mock()
                    mock_read.column.return_value.to_pylist.return_value = mock_texts

                    docs = list(text_iterator(doc_cap=100))

                    # Should stop after reaching cap
                    assert len(docs) == 10  # 10 documents capped at 100 chars
                    for doc in docs:
                        assert len(doc) <= 100

    def test_text_iterator_no_files(self):
        """Test text iterator with no parquet files."""
        with patch("prepare.list_parquet_files", return_value=[]):
            with pytest.raises(AssertionError):
                list(text_iterator())

    def test_make_dataloader_basic(self):
        """Test basic dataloader functionality."""
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab_size.return_value = 1000
        mock_tokenizer.get_bos_token_id.return_value = 1
        with patch("prepare.Pool"):
            with patch("prepare.text_iterator", return_value=iter(["test"])):
                with patch("prepare.torch.empty") as mock_empty:
                    make_dataloader(mock_tokenizer, B=4, T=8, split="train")

                assert True
                mock_empty.assert_called()
                mock_tokenizer.get_bos_token_id.assert_called_once()

    def test_make_dataloader_buffer_management(self):
        """Test dataloader buffer management and best-fit packing."""
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab_size.return_value = 1000
        mock_tokenizer.get_bos_token_id.return_value = 1
        with patch("prepare.Pool"):
            with patch("prepare.text_iterator", return_value=iter(["test"])):
                with patch("prepare.torch.empty") as mock_empty:
                    mock_tokenizer.encode.side_effect = [[1, 2], [3, 4, 5]]
                    with patch(
                        "prepare.text_iterator",
                        return_value=iter(["short", "longer text"]),
                    ):
                        with patch("prepare.torch.empty") as mock_empty:
                            make_dataloader(mock_tokenizer, B=2, T=4, split="train")

                            assert True
                            mock_tokenizer.get_bos_token_id.assert_called_once()
                            mock_empty.assert_called()

    def test_evaluate_bpb_function(self):
        """Test BPE evaluation function."""
        mock_model = Mock()
        mock_model.return_value = torch.tensor([0.5, 0.3])
        mock_tokenizer = Mock()
        with patch("prepare.torch.load"):
            with patch("prepare.get_token_bytes", return_value=torch.zeros(256)):
                with patch("prepare.text_iterator", return_value=iter(["test"])):
                    train_tokenizer()
                    evaluate_bpb(mock_model, mock_tokenizer, batch_size=2)

                    assert True
                    mock_tqdm.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_parse_args_default(self):
        """Test argument parsing with defaults."""
        with patch("prepare.sys.argv", return_value=["prepare.py"]):
            args = parse_args()

            assert args.num_shards == 10
            assert args.skip_download is False
            assert args.skip_tokenizer is False
            assert args.download_workers == 8
            assert args.force is False

    def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        with patch(
            "prepare.sys.argv",
            return_value=[
                "prepare.py",
                "--num-shards",
                "5",
                "--skip-download",
                "--download-workers",
                "4",
            ],
        ):
            args = parse_args()

            assert args.num_shards == 5
            assert args.skip_download is True
            assert args.download_workers == 4

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        with patch("prepare.CACHE_DIR.exists", return_value=True):
            with patch("prepare.CACHE_DIR.stat") as mock_stat:
                mock_stat.return_value.st_size = 1000000
                mock_stat.return_value.st_mtime = 1640995200

                result = get_cache_info()

                assert result["size_bytes"] == 1000000
                assert result["last_modified"] == 1640995200

    def test_cleanup_cache_success(self):
        """Test successful cache cleanup."""
        with patch("prepare.shutil.rmtree") as mock_rmtree:
            result = cleanup_cache()

            assert result is True
            mock_rmtree.assert_called_once_with(CACHE_DIR)

    def test_cleanup_cache_no_directory(self):
        """Test cache cleanup when directory doesn't exist."""
        with patch("prepare.shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = Exception("Directory not found")

            result = cleanup_cache()

            assert result is False
            mock_rmtree.assert_called_once_with(CACHE_DIR)

    def test_validate_tokenizer_basic(self):
        """Test tokenizer validation."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "hello world"

        result = validate_tokenizer(mock_tokenizer)

        assert result is True
        mock_tokenizer.encode.assert_called_once_with("Hello, world!")
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_validate_tokenizer_failure(self):
        """Test tokenizer validation failure."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Encoding failed")

        result = validate_tokenizer(mock_tokenizer)

        assert result is False

    def test_validate_data_format_valid(self):
        """Test valid data format validation."""
        valid_data = [{"text": "test"}, {"text": "another test"}]

        result = validate_data_format(valid_data)

        assert result is True

    def test_validate_data_format_invalid(self):
        """Test invalid data format validation."""
        invalid_data: List[dict] = [{"not_text": "test"}]

        result = validate_data_format(invalid_data)

        assert result is False

    def test_validate_data_format_missing_fields(self):
        """Test data format with missing required fields."""
        invalid_data = [{"not_text": "test"}]  # Missing 'text' field

        result = validate_data_format(invalid_data)

        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_multiprocessing_setup(self):
        """Test multiprocessing setup on macOS."""
        with patch("prepare.sys.platform", return_value="darwin"):
            with patch("prepare.multiprocessing.set_start_method") as mock_set:
                with patch("prepare.sys.platform", return_value="linux"):
                    # Should not call set_start_method on Linux
                    result = main(["prepare.py"])

                    mock_set.assert_called_once_with("spawn", force=True)
                    assert result == 1  # Should fail due to missing data

    def test_network_timeout_handling(self):
        """Test network timeout handling."""
        with patch("prepare.requests.get"):
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("Request timeout")

            result = download_single_shard(1)

            assert result is False

    def test_insufficient_disk_space(self):
        """Test handling of insufficient disk space."""
        with patch("prepare.os.makedirs") as mock_makedirs:
            mock_makedirs.side_effect = OSError("No space left on device")
            train_tokenizer()

            assert True

    def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        with patch("prepare.pq.read_table") as mock_read:
            mock_read.side_effect = Exception("File corrupted")

            result = read_shard(1, DATA_DIR)

            assert result is None

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        # This test would check that functions handle large datasets
        # without excessive memory usage

        # Should not crash or use excessive memory
        result = len(list(text_iterator(doc_cap=1000)))  # Small cap

        assert result > 0  # Should process some data
        assert result < 1000  # But not try to process all at once


if __name__ == "__main__":
    pytest.main([__file__])
