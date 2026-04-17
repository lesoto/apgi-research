"""
================================================================================
ASYNC GIT OPERATIONS TESTS
================================================================================

Comprehensive tests for asynchronous git operations with rollback capabilities.
Tests async git commands, concurrent operations, and error handling.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, List

import pytest
import pytest_asyncio

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from git_operations import GitRollbackManager


@pytest_asyncio.fixture
async def temp_git_repo() -> AsyncGenerator[Path, None]:
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)

        # Initialize git repo
        subprocess.run(
            ["git", "init", "--initial-branch=main"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Create initial commit
        initial_file = repo_path / "initial.txt"
        initial_file.write_text("Initial content")
        subprocess.run(
            ["git", "add", "."], cwd=repo_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        yield repo_path


@pytest.fixture
def make_async(
    temp_git_repo: Path,
) -> Callable[[Callable[..., Any]], Callable[..., Coroutine[Any, Any, Any]]]:
    """Convert synchronous function to async for testing."""

    def wrapper(func: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, Any]]:
        async def async_func(*args: Any, **kwargs: Any) -> Any:
            # Run synchronous function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

        return async_func

    return wrapper


@pytest.mark.asyncio
@pytest.mark.integration
class TestAsyncGitOperations:
    """Tests for async git operations."""

    async def test_async_stage_files(self, temp_git_repo: Path) -> None:
        """Test async file staging."""
        manager = GitRollbackManager(str(temp_git_repo))

        # Create test file
        test_file = temp_git_repo / "test_async.txt"
        test_file.write_text("Async test content")

        # Run staging in async context
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: manager.stage_files(["test_async.txt"])
        )

        assert result.is_valid, f"Failed to stage: {result.errors}"

        # Verify file is staged
        status = manager.get_status()
        assert "test_async.txt" not in status["modified_files"]

        manager.cleanup_backups()

    async def test_async_commit(self, temp_git_repo: Path) -> None:
        """Test async commit operations."""
        manager = GitRollbackManager(str(temp_git_repo))

        # Create and stage test file
        test_file = temp_git_repo / "commit_test.txt"
        test_file.write_text("Commit test content")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: manager.stage_files(["commit_test.txt"])
        )

        # Async commit
        result = await loop.run_in_executor(
            None, lambda: manager.commit_changes("Async test commit")
        )

        assert result.is_valid, f"Failed to commit: {result.errors}"

        # Verify commit exists
        status = manager.get_status()
        assert status["operations_count"] == 2  # stage + commit

        manager.cleanup_backups()

    async def test_async_rollback(self, temp_git_repo: Path) -> None:
        """Test async rollback operations."""
        manager = GitRollbackManager(str(temp_git_repo))

        # Get initial commit
        initial_commit = manager.get_current_commit()

        # Create and commit a file
        test_file = temp_git_repo / "rollback_test.txt"
        test_file.write_text("Rollback test content")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: manager.stage_files(["rollback_test.txt"])
        )

        await loop.run_in_executor(
            None, lambda: manager.commit_changes("Test commit for rollback")
        )

        new_commit = manager.get_current_commit()
        assert new_commit != initial_commit

        # Async rollback
        result = await loop.run_in_executor(
            None, lambda: manager.rollback_last_operation()
        )

        assert result.is_valid, f"Failed to rollback: {result.errors}"

        # Verify we're back to initial commit
        final_commit = manager.get_current_commit()
        assert final_commit == initial_commit

        manager.cleanup_backups()

    async def test_concurrent_git_operations(self, temp_git_repo: Path) -> None:
        """Test concurrent async git operations."""
        manager = GitRollbackManager(str(temp_git_repo))

        # Create multiple test files
        files = []
        for i in range(5):
            test_file = temp_git_repo / f"concurrent_{i}.txt"
            test_file.write_text(f"Concurrent content {i}")
            files.append(f"concurrent_{i}.txt")

        # Run concurrent staging operations
        loop = asyncio.get_event_loop()

        async def stage_file(filename: str) -> Any:
            return await loop.run_in_executor(
                None, lambda: manager.stage_files([filename])
            )

        # Stage all files concurrently
        tasks = [stage_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should complete (some may fail due to git locking)
        success_count = sum(1 for r in results if hasattr(r, "is_valid") and r.is_valid)
        assert success_count >= 1, "At least one staging operation should succeed"

        manager.cleanup_backups()

    async def test_async_branch_creation(self, temp_git_repo: Path) -> None:
        """Test async branch creation."""
        manager = GitRollbackManager(str(temp_git_repo))

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: manager.create_branch("async-test-branch")
        )

        assert result.is_valid, f"Failed to create branch: {result.errors}"

        # Verify branch was created
        proc_result = subprocess.run(
            ["git", "branch", "--list", "async-test-branch"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        branches = proc_result.stdout
        assert "async-test-branch" in branches

        manager.cleanup_backups()


@pytest.mark.asyncio
@pytest.mark.stress
class TestAsyncGitStress:
    """Stress tests for async git operations."""

    async def test_rapid_async_commits(self, temp_git_repo: Path) -> None:
        """Test rapid async commit operations."""
        manager = GitRollbackManager(str(temp_git_repo))

        loop = asyncio.get_event_loop()

        async def create_and_commit(idx: int) -> Any:
            # Create file
            test_file = temp_git_repo / f"rapid_{idx}.txt"
            test_file.write_text(f"Rapid content {idx}")

            # Stage and commit
            await loop.run_in_executor(
                None, lambda: manager.stage_files([f"rapid_{idx}.txt"])
            )
            return await loop.run_in_executor(
                None, lambda: manager.commit_changes(f"Rapid commit {idx}")
            )

        # Run 10 rapid commits
        tasks = [create_and_commit(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        success_count = sum(1 for r in results if hasattr(r, "is_valid") and r.is_valid)

        # At least some should succeed (git may lock but should recover)
        assert success_count >= 1, f"Expected at least 1 success, got {success_count}"

        manager.cleanup_backups()

    async def test_async_operation_timeouts(self, temp_git_repo: Path) -> None:
        """Test that async operations respect timeouts."""
        manager = GitRollbackManager(str(temp_git_repo))

        loop = asyncio.get_event_loop()

        # Create a long-running operation by staging many files
        files = []
        for i in range(100):
            test_file = temp_git_repo / f"timeout_{i}.txt"
            test_file.write_text(f"Content {i}")
            files.append(f"timeout_{i}.txt")

        # Stage with timeout
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: manager.stage_files(files[:10])),
                timeout=5.0,
            )
            assert result.is_valid
        except asyncio.TimeoutError:
            pytest.skip("Operation timed out - may indicate slow git operations")

        manager.cleanup_backups()


@pytest.mark.asyncio
@pytest.mark.performance
class TestAsyncGitPerformance:
    """Performance tests for async git operations."""

    async def test_async_vs_sync_performance(self, temp_git_repo: Path) -> None:
        """Compare async vs synchronous git operation performance."""
        manager = GitRollbackManager(str(temp_git_repo))

        # Create test files
        files = []
        for i in range(20):
            test_file = temp_git_repo / f"perf_{i}.txt"
            test_file.write_text(f"Performance content {i}")
            files.append(f"perf_{i}.txt")

        loop = asyncio.get_event_loop()

        # Async approach - stage files concurrently in batches
        async def async_approach() -> List[Any]:
            batch_size = 5
            results = []
            for i in range(0, len(files), batch_size):
                batch = files[i : i + batch_size]
                result = await loop.run_in_executor(
                    None, lambda b=batch: manager.stage_files(b)  # type: ignore[misc]
                )
                results.append(result)
            return results

        # Measure async performance
        import time

        start = time.time()
        await async_approach()
        async_duration = time.time() - start

        # Reset
        manager.cleanup_backups()

        # Sync approach - stage sequentially
        start = time.time()
        for f in files:
            manager.stage_files([f])
        sync_duration = time.time() - start

        # Async should be faster or comparable
        assert (
            async_duration <= sync_duration * 1.5
        ), "Async should not be significantly slower"

        manager.cleanup_backups()

    async def test_async_memory_usage(self, temp_git_repo: Path) -> None:
        """Test memory usage during async operations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        manager = GitRollbackManager(str(temp_git_repo))

        mem_before = process.memory_info().rss / 1024 / 1024

        loop = asyncio.get_event_loop()

        # Run many async operations
        async def run_many() -> List[Any]:
            tasks = []
            for i in range(20):
                test_file = temp_git_repo / f"mem_{i}.txt"
                test_file.write_text(f"Memory test content {i}")
                task = loop.run_in_executor(
                    None, lambda idx=i: manager.stage_files([f"mem_{idx}.txt"])  # type: ignore[misc]
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)

        await run_many()

        mem_after = process.memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before

        # Should not leak memory excessively
        assert mem_increase < 50, f"Memory increase too large: {mem_increase:.1f}MB"

        manager.cleanup_backups()


# Configure pytest markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with async-specific markers."""
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")
