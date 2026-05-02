"""
Comprehensive tests for git_operations.py - Git operations module.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from git_operations import (
    GitError,
    GitOperation,
    GitOperations,
    GitStatus,
    RepositoryInfo,
    commit_changes,
    create_branch,
    get_current_branch,
    get_repository_info,
    is_git_repository,
    push_changes,
    stage_files,
)


class TestGitOperation:
    """Tests for GitOperation dataclass."""

    def test_default_values(self):
        """Test default GitOperation values."""
        op = GitOperation(operation_type="stage", files=["test.py"])
        assert op.operation_type == "stage"
        assert op.files == ["test.py"]
        assert op.commit_hash is None
        assert op.branch_name is None
        assert op.timestamp == 0.0

    def test_custom_values(self):
        """Test custom GitOperation values."""
        op = GitOperation(
            operation_type="commit",
            files=["file1.py", "file2.py"],
            commit_hash="abc123",
            branch_name="main",
            timestamp=1234567890.0,
        )
        assert op.operation_type == "commit"
        assert op.files == ["file1.py", "file2.py"]
        assert op.commit_hash == "abc123"
        assert op.branch_name == "main"
        assert op.timestamp == 1234567890.0


class TestGitStatus:
    """Tests for GitStatus dataclass."""

    def test_default_values(self):
        """Test default GitStatus values."""
        status = GitStatus()
        assert status.branch == ""
        assert status.modified_files == []
        assert status.untracked_files == []
        assert status.staged_files == []
        assert status.is_clean is True

    def test_custom_values(self):
        """Test custom GitStatus values."""
        status = GitStatus(
            branch="main",
            modified_files=["file1.py"],
            untracked_files=["file2.py"],
            staged_files=["file3.py"],
            is_clean=False,
        )
        assert status.branch == "main"
        assert status.modified_files == ["file1.py"]
        assert status.untracked_files == ["file2.py"]
        assert status.staged_files == ["file3.py"]
        assert status.is_clean is False


class TestRepositoryInfo:
    """Tests for RepositoryInfo dataclass."""

    def test_default_values(self):
        """Test default RepositoryInfo values."""
        info = RepositoryInfo()
        assert info.root_path == Path()
        assert info.current_branch == ""
        assert info.remote_url is None
        assert info.last_commit_hash is None

    def test_custom_values(self):
        """Test custom RepositoryInfo values."""
        info = RepositoryInfo(
            root_path=Path("/repo"),
            current_branch="main",
            remote_url="https://github.com/user/repo.git",
            last_commit_hash="abc123",
        )
        assert info.root_path == Path("/repo")
        assert info.current_branch == "main"
        assert info.remote_url == "https://github.com/user/repo.git"
        assert info.last_commit_hash == "abc123"


class TestGitError:
    """Tests for GitError exception."""

    def test_error_creation(self):
        """Test creating GitError."""
        error = GitError("Test error")
        assert str(error) == "Test error"

    def test_error_with_command(self):
        """Test GitError with command info."""
        error = GitError("Command failed", command="git status")
        assert "Command failed" in str(error)


class TestIsGitRepository:
    """Tests for is_git_repository function."""

    def test_valid_repository(self, tmp_path):
        """Test with valid git repository."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        assert is_git_repository(tmp_path) is True

    def test_invalid_repository(self, tmp_path):
        """Test with non-git directory."""
        assert is_git_repository(tmp_path) is False

    def test_nonexistent_path(self):
        """Test with nonexistent path."""
        assert is_git_repository(Path("/nonexistent/path")) is False


class TestGetRepositoryInfo:
    """Tests for get_repository_info function."""

    @patch("git_operations.subprocess.run")
    def test_get_info_success(self, mock_run, tmp_path):
        """Test getting repository info successfully."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456",
            stderr="",
        )
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        with patch("git_operations.is_git_repository", return_value=True):
            info = get_repository_info(tmp_path)
            assert isinstance(info, RepositoryInfo)

    def test_get_info_not_repo(self, tmp_path):
        """Test getting info from non-repository."""
        with pytest.raises(GitError):
            get_repository_info(tmp_path)


class TestGetCurrentBranch:
    """Tests for get_current_branch function."""

    @patch("git_operations.subprocess.run")
    def test_get_branch_success(self, mock_run):
        """Test getting current branch successfully."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="main\n",
            stderr="",
        )
        branch = get_current_branch()
        assert branch == "main"

    @patch("git_operations.subprocess.run")
    def test_get_branch_failure(self, mock_run):
        """Test getting branch when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        branch = get_current_branch()
        assert branch == ""


class TestCreateBranch:
    """Tests for create_branch function."""

    @patch("git_operations.subprocess.run")
    def test_create_branch_success(self, mock_run):
        """Test creating branch successfully."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = create_branch("feature/new-feature")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_create_branch_failure(self, mock_run):
        """Test creating branch when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = create_branch("feature/new-feature")
        assert result is False


class TestStageFiles:
    """Tests for stage_files function."""

    @patch("git_operations.subprocess.run")
    def test_stage_single_file(self, mock_run):
        """Test staging single file."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = stage_files("file.py")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_stage_multiple_files(self, mock_run):
        """Test staging multiple files."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = stage_files(["file1.py", "file2.py"])
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_stage_failure(self, mock_run):
        """Test staging when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = stage_files("file.py")
        assert result is False


class TestCommitChanges:
    """Tests for commit_changes function."""

    @patch("git_operations.subprocess.run")
    def test_commit_success(self, mock_run):
        """Test committing changes successfully."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = commit_changes("Test commit message")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_commit_failure(self, mock_run):
        """Test committing when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = commit_changes("Test commit message")
        assert result is False


class TestPushChanges:
    """Tests for push_changes function."""

    @patch("git_operations.subprocess.run")
    def test_push_success(self, mock_run):
        """Test pushing changes successfully."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = push_changes()
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_push_with_remote(self, mock_run):
        """Test pushing to specific remote."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = push_changes(remote="origin", branch="main")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_push_failure(self, mock_run):
        """Test pushing when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = push_changes()
        assert result is False


class TestGitOperations:
    """Tests for GitOperations class."""

    def test_init_default(self):
        """Test initialization with default path."""
        ops = GitOperations()
        # Path is resolved to absolute, so check it ends with current dir
        assert ops.repo_path.is_absolute()
        assert ops.repo_path.name == "apgi-research"

    def test_init_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        ops = GitOperations(tmp_path)
        assert ops.repo_path == tmp_path

    @patch("git_operations.subprocess.run")
    def test_status(self, mock_run, tmp_path):
        """Test getting git status."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="## main\n M file1.py\n?? file2.py\nA  file3.py",
            stderr="",
        )
        ops = GitOperations(tmp_path)
        status = ops.status()
        assert isinstance(status, GitStatus)
        assert status.branch == "main"

    @patch("git_operations.subprocess.run")
    def test_add(self, mock_run, tmp_path):
        """Test adding files."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.add("file.py")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_commit(self, mock_run, tmp_path):
        """Test committing."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.commit("Test message")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_push(self, mock_run, tmp_path):
        """Test pushing."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.push()
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_pull(self, mock_run, tmp_path):
        """Test pulling."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.pull()
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_fetch(self, mock_run, tmp_path):
        """Test fetching."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.fetch()
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_checkout(self, mock_run, tmp_path):
        """Test checkout."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.checkout("feature-branch")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_create_branch_method(self, mock_run, tmp_path):
        """Test create branch method."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.create_branch("new-branch")
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_get_log(self, mock_run, tmp_path):
        """Test getting commit log."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123 - Test commit\ndef456 - Another commit",
            stderr="",
        )
        ops = GitOperations(tmp_path)
        log = ops.get_log(max_entries=2)
        assert len(log) == 2

    @patch("git_operations.subprocess.run")
    def test_get_remotes(self, mock_run, tmp_path):
        """Test getting remotes."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="origin\thttps://github.com/user/repo.git (fetch)\n",
            stderr="",
        )
        ops = GitOperations(tmp_path)
        remotes = ops.get_remotes()
        assert "origin" in remotes

    @patch("git_operations.subprocess.run")
    def test_is_clean(self, mock_run, tmp_path):
        """Test checking if working tree is clean."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ops = GitOperations(tmp_path)
        result = ops.is_clean()
        assert result is True

    @patch("git_operations.subprocess.run")
    def test_get_current_branch_method(self, mock_run, tmp_path):
        """Test getting current branch."""
        mock_run.return_value = MagicMock(returncode=0, stdout="main\n", stderr="")
        ops = GitOperations(tmp_path)
        branch = ops.get_current_branch()
        assert branch == "main"
