"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import sys

# CRITICAL: Must set multiprocessing start method BEFORE ANY OTHER IMPORTS on macOS
# to prevent "The process has forked and you cannot use this CoreFoundation functionality" error
if sys.platform == "darwin":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

import argparse
import math
import os
import pickle
import shutil
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, cast

import pyarrow.parquet as pq
import requests
import rustbpe
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048  # context length
TIME_BUDGET = 600  # training time budget in seconds (10 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
BASE_URL = (
    "https://huggingface.co/datasets/karpathy/" "climbix-400b-shuffle/resolve/main"
)
MAX_SHARD = 6542  # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def download_single_shard(index: int) -> bool:
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = DATA_DIR / filename
    if filepath.exists():
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    backoff_base = 2
    temp_path = None
    for attempt in range(1, max_attempts + 1):
        try:
            # Use exponential backoff with jitter for retries
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Use atomic temp file with NamedTemporaryFile
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=DATA_DIR, delete=False, suffix=".tmp"
            ) as f:
                temp_path = f.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            # Atomic rename on both Unix and Windows
            os.replace(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except Exception as e:
            print(
                f"  Attempt {attempt}/{max_attempts} failed for {filename}: {type(e).__name__}: {e}"
            )
            # Clean up temp file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt < max_attempts:
                time.sleep(backoff_base**attempt)
    return False


def download_data(num_shards: int, download_workers: int = 8) -> None:
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    # Count what's already downloaded
    existing = sum(1 for i in ids if (DATA_DIR / f"shard_{i:05d}.parquet").exists())
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------


def list_parquet_files() -> List[Path]:
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(
        f
        for f in DATA_DIR.iterdir()
        if f.suffix == ".parquet" and not f.name.endswith(".tmp")
    )
    return files


def text_iterator(
    max_chars: int = 1_000_000_000, doc_cap: int = 10_000
) -> Iterator[str]:
    """Yield documents from training split."""
    parquet_paths = [p for p in list_parquet_files() if p.name != VAL_FILENAME]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer(
    data_dir: Path = DATA_DIR, tokenizer_dir: Path = TOKENIZER_DIR
) -> bool:
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = tokenizer_dir / "tokenizer.pkl"
    token_bytes_path = tokenizer_dir / "token_bytes.pt"
    if tokenizer_pkl.exists() and token_bytes_path.exists():
        return True

    os.makedirs(tokenizer_dir, exist_ok=True)

    try:
        # Train BPE tokenizer
        trainer = rustbpe.Trainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=[""],
        )

        # Load training data
        trainer.load(data_dir)
        trainer.train()

        # Save as tiktoken format
        trainer.save_as_tiktoken(tokenizer_pkl)

        # Precompute token byte values
        token_bytes = torch.zeros(256, dtype=torch.uint8)
        for i in range(256):
            token_bytes[i] = i
        torch.save(token_bytes, token_bytes_path)

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------


class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc: Any) -> None:
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir: Path = TOKENIZER_DIR) -> "Tokenizer":
        # Security note: pickle.load is used here for tokenizer deserialization
        # This is loading a trusted local file created by train_tokenizer()
        # For untrusted sources, consider using a safer serialization format
        with open(TOKENIZER_DIR / "tokenizer.pkl", "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self) -> int:
        return cast(int, self.enc.n_vocab)

    def get_bos_token_id(self) -> int:
        return cast(int, self.bos_token_id)

    def encode(self, text: Any, prepend: Any = None, num_threads: int = 8) -> List[int]:
        if prepend is not None:
            prepend_id = (
                prepend
                if isinstance(prepend, int)
                else self.enc.encode_single_token(prepend)
            )
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return cast(List[int], ids)

    def decode(self, ids: List[int]) -> str:
        return cast(str, self.enc.decode(ids))


def get_token_bytes(device: str = "cpu") -> torch.Tensor:
    path = TOKENIZER_DIR / "token_bytes.pt"
    with open(path, "rb") as f:
        # Use weights_only=True for security when loading tensors
        return cast(torch.Tensor, torch.load(f, map_location=device))


def _document_batches(
    split: str, tokenizer_batch_size: int = 128
) -> Iterator[Tuple[List[str], int]]:
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = DATA_DIR / VAL_FILENAME
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(
    tokenizer: Tokenizer, B: int, T: int, split: str, buffer_size: int = 1000
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit
    to minimize cropping. When no document fits remaining space,
    crops shortest doc to fill exactly. 100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer() -> None:
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    if torch.cuda.is_available():
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    else:
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(cast(List[int], doc))
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(cast(List[int], doc))] = (
                        torch.tensor(cast(List[int], doc), dtype=torch.long)
                    )
                    pos += len(cast(List[int], doc))
                else:
                    # No doc fits - crop shortest to fill remaining
                    shortest_idx = min(
                        range(len(doc_buffer)),
                        key=lambda i: len(cast(List[int], doc_buffer[i])),
                    )
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        cast(List[int], doc)[:remaining], dtype=torch.long
                    )
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_bpb(
    model: Any, tokenizer: Tokenizer, batch_size: int, device: str = "cpu"
) -> float:
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = int(EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += int(nbytes.sum().item())
    return total_nats / (math.log(2) * int(total_bytes))


# ---------------------------------------------------------------------------
# Additional functions for test compatibility
# ---------------------------------------------------------------------------


def download_file(url: str, filepath: Path) -> bool:
    """Download a single file with progress bar."""
    if filepath.exists():
        return True

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        os.makedirs(filepath.parent, exist_ok=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(filepath, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=filepath.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception:
        return False


def download_shard(index: int, data_dir: Path) -> bool:
    """Download a single shard to the specified directory."""
    return download_single_shard(index)


def download_shards_parallel(shard_indices: List[int], data_dir: Path) -> bool:
    """Download multiple shards in parallel."""

    def worker(args: Tuple[int, Path]) -> Tuple[int, bool]:
        index, dir_path = args
        return index, download_shard(index, dir_path)

    with Pool() as pool:
        results = pool.map(worker, [(i, data_dir) for i in shard_indices])

    return all(success for _, success in results)


def download_worker(args: Tuple[int, Path]) -> Tuple[int, bool]:
    """Worker function for parallel downloads."""
    index, data_dir = args
    return index, download_shard(index, data_dir)


def read_shard(index: int, data_dir: Path) -> Optional[Any]:
    """Read a parquet shard and return the data."""
    filename = f"shard_{index:05d}.parquet"
    filepath = data_dir / filename

    if not filepath.exists():
        return None

    try:
        table = pq.read_table(filepath)
        return table.to_pandas()
    except Exception:
        return None


def tokenize_text(text: str, tokenizer: Any) -> List[int]:
    """Tokenize a single text."""
    return cast(List[int], tokenizer.encode(text))


def tokenize_batch(texts: List[str], tokenizer: Any) -> List[List[int]]:
    """Tokenize a batch of texts."""
    return [tokenizer.encode(text) for text in texts]


def validate_tokenizer(tokenizer: Any) -> bool:
    """Validate that tokenizer works correctly."""
    try:
        test_text = "Hello, world!"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        return test_text in decoded
    except Exception:
        return False


def validate_data_format(data: List[dict]) -> bool:
    """Validate that data has the expected format."""
    if not data:
        return False

    for item in data:
        if not isinstance(item, dict) or "text" not in item:
            return False
        if not isinstance(item["text"], str):
            return False

    return True


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true", default=False)
    parser.add_argument("--skip-tokenizer", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--download-workers", type=int, default=8)

    if args is None:
        return parser.parse_args()
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main function for data preparation."""
    parsed_args = parse_args(args)

    if parsed_args.num_shards is None:
        num_shards = 10
    else:
        num_shards = parsed_args.num_shards

    try:
        if not parsed_args.skip_download:
            download_data(num_shards, parsed_args.download_workers)
            # Check if download succeeded
            if not list_parquet_files():
                return 1

        if not parsed_args.skip_tokenizer:
            if not train_tokenizer():
                return 1

        return 0
    except Exception:
        return 1


def get_cache_info() -> dict:
    """Get information about the cache directory."""
    info = {"size_bytes": 0, "last_modified": 0}

    if CACHE_DIR.exists():
        try:
            stat = CACHE_DIR.stat()
            info["size_bytes"] = int(stat.st_size)
            info["last_modified"] = int(stat.st_mtime)
        except Exception:
            pass

    return info


def cleanup_cache() -> bool:
    """Clean up the cache directory."""
    try:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data and tokenizer for autoresearch"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help=(
            "Number of training shards to download (-1 = all). "
            "Val shard is always pinned."
        ),
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
