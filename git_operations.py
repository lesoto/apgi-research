"""
Git operations with rollback capabilities for APGI experiments.

Provides safe git operations with automatic rollback on failures.
"""

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from validation import ValidationResult, validate_git_operations


class GitError(Exception):
    """Exception for git operation errors."""

    def __init__(self, message: str, command: Optional[str] = None):
        super().__init__(message)
        self.command = command


@dataclass
class GitConfig:
    """Git configuration settings."""

    user_name: str = ""
    user_email: str = ""
    default_branch: str = "main"
    auto_commit: bool = False


@dataclass
class GitStatus:
    """Git repository status."""

    branch: str = ""
    modified_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)
    staged_files: List[str] = field(default_factory=list)
    is_clean: bool = True


@dataclass
class RepositoryInfo:
    """Information about a git repository."""

    root_path: Path = field(default_factory=Path)
    current_branch: str = ""
    remote_url: Optional[str] = None
    last_commit_hash: Optional[str] = None


@dataclass
class GitOperation:
    """Represents a git operation that can be rolled back."""

    operation_type: str
    files: List[str]
    commit_hash: Optional[str] = None
    branch_name: Optional[str] = None
    timestamp: float = 0.0


class GitRollbackManager:
    """Manages git operations with rollback capabilities."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.operations_history: List[GitOperation] = []
        self.temp_backup_dir: Optional[Path] = None

    def _run_git_command(
        self, cmd: List[str], check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        try:
            result = subprocess.run(
                ["git"] + cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check,
            )
            return result
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\nError: {e.stderr}"
            )

    def _create_backup(self, files: List[str]) -> Path:
        """Create backup of files before modification."""
        if self.temp_backup_dir is None:
            self.temp_backup_dir = Path(tempfile.mkdtemp(prefix="apgi_git_backup_"))

        backup_dir = self.temp_backup_dir / f"backup_{int(time.time())}"
        backup_dir.mkdir(exist_ok=True)

        for file_path in files:
            path_obj = Path(file_path)
            # If path is absolute, use it directly; otherwise resolve relative to repo
            if path_obj.is_absolute():
                src = path_obj
                # Use just the filename for backup path to avoid nesting issues
                dst = backup_dir / path_obj.name
            else:
                src = self.repo_path / file_path
                dst = backup_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)

            if src.exists():
                shutil.copy2(str(src), str(dst))

        return backup_dir

    def _restore_backup(self, backup_dir: Path, files: List[str]) -> None:
        """Restore files from backup."""
        for file_path in files:
            src = backup_dir / file_path
            dst = self.repo_path / file_path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def get_current_commit(self) -> str:
        """Get the current commit hash."""
        try:
            result = self._run_git_command(["rev-parse", "HEAD"])
            return str(result.stdout.strip())
        except RuntimeError:
            # No commits yet, try to get the initial commit or return empty string
            try:
                result = self._run_git_command(
                    ["rev-list", "--max-parents=0", "HEAD"],
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return str(result.stdout.strip())
                return ""
            except Exception:
                return ""

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        try:
            result = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
            return str(result.stdout.strip())
        except RuntimeError:
            # No commits yet or no HEAD, try to get the default branch
            try:
                result = self._run_git_command(
                    ["branch", "--show-current"], check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    return str(result.stdout.strip())
                return "main"  # Default fallback
            except Exception:
                return "main"  # Default fallback

    def stage_files(self, files: List[str]) -> ValidationResult:
        """Stage files with rollback capability."""
        errors: List[str] = []
        warnings: List[str] = []

        # Convert relative paths to absolute paths based on repo_path
        absolute_files = []
        for file_path in files:
            path_obj = Path(file_path)
            if not path_obj.is_absolute():
                path_obj = self.repo_path / file_path
            absolute_files.append(str(path_obj))

        # Validate files first
        validation = validate_git_operations(absolute_files, "add")
        if not validation.is_valid:
            return validation

        # Get current state for rollback
        current_commit = self.get_current_commit()
        current_branch = self.get_current_branch()

        # If no commits exist, create an initial commit first
        if not current_commit:
            try:
                # Create a dummy file to ensure there's something to commit
                dummy_file = self.repo_path / ".gitkeep"
                dummy_file.write_text("Initial commit")
                self._run_git_command(["add", ".gitkeep"])
                self._run_git_command(["commit", "-m", "Initial commit"])
                current_commit = self.get_current_commit()
                warnings.append("Created initial commit")
            except Exception as e:
                errors.append(f"Failed to create initial commit: {str(e)}")
                return ValidationResult(
                    is_valid=False, errors=errors, warnings=warnings
                )

        try:
            # Create backup before staging (needs absolute paths)
            backup_dir = self._create_backup(absolute_files)

            # Stage files using relative paths from repo root
            for file_path in files:
                self._run_git_command(["add", file_path])

            # Record operation for rollback
            operation = GitOperation(
                operation_type="stage",
                files=files,
                commit_hash=current_commit,
                branch_name=current_branch,
                timestamp=time.time(),
            )
            self.operations_history.append(operation)

            warnings.append(f"Created backup at {backup_dir}")

        except Exception as e:
            errors.append(f"Failed to stage files: {str(e)}")
            # Try to restore from backup if available
            if "backup_dir" in locals():
                try:
                    self._restore_backup(backup_dir, files)
                    warnings.append("Restored files from backup after staging failure")
                except Exception as restore_error:
                    errors.append(f"Failed to restore backup: {str(restore_error)}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def commit_changes(
        self, message: str, files: Optional[List[str]] = None
    ) -> ValidationResult:
        """Commit changes with rollback capability."""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            # Get current state before commit
            current_commit = self.get_current_commit()
            current_branch = self.get_current_branch()

            # Create backup before commit
            if files:
                backup_dir = self._create_backup(files)
            else:
                backup_dir = None

            # Commit changes
            commit_cmd = ["commit", "-m", message]
            result = self._run_git_command(commit_cmd)

            new_commit = self.get_current_commit()

            # Record operation for rollback
            operation = GitOperation(
                operation_type="commit",
                files=files or [],
                commit_hash=current_commit,
                branch_name=current_branch,
                timestamp=time.time(),
            )
            self.operations_history.append(operation)

            warnings.append(f"Committed {new_commit[:8] if new_commit else 'unknown'}")
            if backup_dir:
                warnings.append(f"Created backup at {backup_dir}")

        except Exception as e:
            errors.append(f"Failed to commit changes: {str(e)}")
            # Try to restore from backup if available
            if "backup_dir" in locals() and backup_dir and files:
                try:
                    self._restore_backup(backup_dir, files)
                    warnings.append("Restored files from backup after commit failure")
                except Exception as restore_error:
                    errors.append(f"Failed to restore backup: {str(restore_error)}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def rollback_last_operation(self) -> ValidationResult:
        """Rollback the last git operation."""
        errors: List[str] = []
        warnings: List[str] = []

        if not self.operations_history:
            errors.append("No operations to rollback")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        last_operation = self.operations_history[-1]

        try:
            if last_operation.operation_type == "commit":
                # Reset to previous commit
                if last_operation.commit_hash:
                    self._run_git_command(
                        ["reset", "--hard", last_operation.commit_hash]
                    )
                    warnings.append(
                        f"Rolled back commit to {last_operation.commit_hash[:8]}"
                    )
                else:
                    errors.append("Cannot rollback commit: no commit hash available")
                    return ValidationResult(
                        is_valid=False, errors=errors, warnings=warnings
                    )

                # Restore files from backup if available
                if self.temp_backup_dir and self.temp_backup_dir.exists():
                    backup_dirs = [
                        d for d in self.temp_backup_dir.iterdir() if d.is_dir()
                    ]
                    if backup_dirs:
                        latest_backup = max(
                            backup_dirs, key=lambda x: x.stat().st_mtime
                        )
                        self._restore_backup(latest_backup, last_operation.files)
                        warnings.append(f"Restored files from backup: {latest_backup}")

            elif last_operation.operation_type == "stage":
                # Unstage files
                for file_path in last_operation.files:
                    self._run_git_command(["reset", "HEAD", "--", file_path])
                warnings.append(f"Unstaged files: {', '.join(last_operation.files)}")

                # Restore files from backup if available
                if self.temp_backup_dir and self.temp_backup_dir.exists():
                    backup_dirs = [
                        d for d in self.temp_backup_dir.iterdir() if d.is_dir()
                    ]
                    if backup_dirs:
                        latest_backup = max(
                            backup_dirs, key=lambda x: x.stat().st_mtime
                        )
                        self._restore_backup(latest_backup, last_operation.files)
                        warnings.append(f"Restored files from backup: {latest_backup}")

            # Remove operation from history
            self.operations_history.pop()

        except Exception as e:
            errors.append(f"Failed to rollback operation: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def rollback_to_commit(
        self, commit_hash: str, files: Optional[List[str]] = None
    ) -> ValidationResult:
        """Rollback to a specific commit."""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            # Validate commit exists
            self._run_git_command(["rev-parse", commit_hash])

            # Create backup before rollback
            if files:
                backup_dir = self._create_backup(files)
            else:
                backup_dir = None

            # Reset to specified commit
            if files:
                # Reset specific files
                for file_path in files:
                    self._run_git_command(["checkout", commit_hash, "--", file_path])
                warnings.append(f"Reset files to {commit_hash[:8]}: {', '.join(files)}")
            else:
                # Reset entire repository
                self._run_git_command(["reset", "--hard", commit_hash])
                warnings.append(f"Reset repository to {commit_hash[:8]}")

            if backup_dir:
                warnings.append(f"Created backup at {backup_dir}")

        except Exception as e:
            errors.append(f"Failed to rollback to commit {commit_hash}: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def create_branch(self, branch_name: str) -> ValidationResult:
        """Create a new branch for safety."""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            # Check if branch already exists
            result = self._run_git_command(
                ["branch", "--list", branch_name], check=False
            )
            if result.stdout.strip():
                warnings.append(f"Branch {branch_name} already exists")
                return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

            # Create and checkout new branch
            self._run_git_command(["checkout", "-b", branch_name])
            warnings.append(f"Created and checked out branch: {branch_name}")

        except Exception as e:
            errors.append(f"Failed to create branch {branch_name}: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def cleanup_backups(self) -> None:
        """Clean up temporary backup directories."""
        if self.temp_backup_dir and self.temp_backup_dir.exists():
            try:
                shutil.rmtree(self.temp_backup_dir)
                self.temp_backup_dir = None
            except Exception as e:
                print(f"Warning: Failed to cleanup backup directory: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current git status."""
        try:
            result = self._run_git_command(["status", "--porcelain"])
            modified_files = [
                line[3:] for line in result.stdout.split("\n") if line.startswith(" M")
            ]

            return {
                "current_commit": self.get_current_commit(),
                "current_branch": self.get_current_branch(),
                "modified_files": modified_files,
                "operations_count": len(self.operations_history),
                "backup_available": self.temp_backup_dir is not None
                and self.temp_backup_dir.exists(),
            }
        except Exception as e:
            return {"error": str(e)}


class GitOperations:
    """Git operations interface for common git commands."""

    def __init__(self, repo_path: Union[str, Path] = "."):
        self.repo_path = Path(repo_path).resolve()

    def _run_git_command(
        self, cmd: List[str], check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        try:
            result = subprocess.run(
                ["git"] + cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check,
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitError(
                f"Git command failed: {' '.join(cmd)}\nError: {e.stderr}",
                command=" ".join(cmd),
            )

    def status(self) -> GitStatus:
        """Get git status."""
        try:
            result = self._run_git_command(["status", "--porcelain", "-b"])
            lines = result.stdout.strip().split("\n")

            branch = ""
            modified = []
            untracked = []
            staged = []

            for line in lines:
                if line.startswith("##"):
                    branch = line[3:].split("...")[0].strip()
                elif line.startswith(" M") or line.startswith("M "):
                    modified.append(line[3:])
                elif line.startswith("A "):
                    staged.append(line[3:])
                elif line.startswith("??"):
                    untracked.append(line[3:])

            is_clean = not (modified or untracked or staged)

            return GitStatus(
                branch=branch,
                modified_files=modified,
                untracked_files=untracked,
                staged_files=staged,
                is_clean=is_clean,
            )
        except Exception:
            return GitStatus()

    def add(self, files: Union[str, List[str]]) -> bool:
        """Add files to staging area."""
        if isinstance(files, str):
            files = [files]
        try:
            for f in files:
                self._run_git_command(["add", f])
            return True
        except GitError:
            return False

    def commit(self, message: str) -> bool:
        """Commit staged changes."""
        try:
            self._run_git_command(["commit", "-m", message])
            return True
        except GitError:
            return False

    def push(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Push changes to remote."""
        try:
            if branch:
                self._run_git_command(["push", remote, branch])
            else:
                self._run_git_command(["push"])
            return True
        except GitError:
            return False

    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Pull changes from remote."""
        try:
            if branch:
                self._run_git_command(["pull", remote, branch])
            else:
                self._run_git_command(["pull"])
            return True
        except GitError:
            return False

    def fetch(self, remote: str = "origin") -> bool:
        """Fetch changes from remote."""
        try:
            self._run_git_command(["fetch", remote])
            return True
        except GitError:
            return False

    def checkout(self, branch: str, create: bool = False) -> bool:
        """Checkout a branch."""
        try:
            if create:
                self._run_git_command(["checkout", "-b", branch])
            else:
                self._run_git_command(["checkout", branch])
            return True
        except GitError:
            return False

    def create_branch(self, branch_name: str) -> bool:
        """Create and checkout a new branch."""
        return self.checkout(branch_name, create=True)

    def get_log(self, max_entries: int = 10) -> List[str]:
        """Get commit log."""
        try:
            result = self._run_git_command(["log", f"-{max_entries}", "--oneline"])
            output = result.stdout.strip()
            if not output:
                return []
            return [line for line in output.split("\n") if line]
        except GitError:
            return []

    def get_remotes(self) -> Dict[str, str]:
        """Get configured remotes."""
        try:
            result = self._run_git_command(["remote", "-v"])
            remotes = {}
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        remotes[parts[0]] = parts[1]
            return remotes
        except GitError:
            return {}

    def is_clean(self) -> bool:
        """Check if working tree is clean."""
        status = self.status()
        return status.is_clean

    def get_current_branch(self) -> str:
        """Get current branch name."""
        try:
            result = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
            branch: str = result.stdout.strip()
            return branch
        except GitError:
            return ""


# Convenience functions for common operations
def safe_git_add(files: List[str], repo_path: str = ".") -> ValidationResult:
    """Safely add files to git with rollback capability."""
    manager = GitRollbackManager(repo_path)
    try:
        result = manager.stage_files(files)
        return result
    finally:
        manager.cleanup_backups()


def safe_git_commit(
    message: str, files: Optional[List[str]] = None, repo_path: str = "."
) -> ValidationResult:
    """Safely commit changes with rollback capability."""
    manager = GitRollbackManager(repo_path)
    try:
        result = manager.commit_changes(message, files)
        return result
    finally:
        manager.cleanup_backups()


def safe_git_rollback(repo_path: str = ".") -> ValidationResult:
    """Rollback the last git operation."""
    manager = GitRollbackManager(repo_path)
    try:
        result = manager.rollback_last_operation()
        return result
    finally:
        manager.cleanup_backups()


def is_git_repository(path: Union[str, Path]) -> bool:
    """Check if a path is a git repository."""
    path_obj = Path(path)
    git_dir = path_obj / ".git"
    return git_dir.exists() and git_dir.is_dir()


def get_repository_info(path: Union[str, Path]) -> RepositoryInfo:
    """Get information about a git repository."""
    if not is_git_repository(path):
        raise GitError(f"Not a git repository: {path}")

    ops = GitOperations(path)
    info = RepositoryInfo(root_path=Path(path).resolve())

    try:
        info.current_branch = ops.get_current_branch()
    except Exception:
        pass

    try:
        result = ops._run_git_command(["rev-parse", "HEAD"])
        info.last_commit_hash = result.stdout.strip()
    except Exception:
        pass

    try:
        remotes = ops.get_remotes()
        if "origin" in remotes:
            info.remote_url = remotes["origin"]
    except Exception:
        pass

    return info


def get_current_branch(repo_path: Union[str, Path] = ".") -> str:
    """Get the current branch name."""
    ops = GitOperations(repo_path)
    return ops.get_current_branch()


def create_branch(branch_name: str, repo_path: Union[str, Path] = ".") -> bool:
    """Create and checkout a new branch."""
    ops = GitOperations(repo_path)
    return ops.create_branch(branch_name)


def stage_files(
    files: Union[str, List[str]], repo_path: Union[str, Path] = "."
) -> bool:
    """Stage files for commit."""
    ops = GitOperations(repo_path)
    return ops.add(files)


def commit_changes(message: str, repo_path: Union[str, Path] = ".") -> bool:
    """Commit staged changes."""
    ops = GitOperations(repo_path)
    return ops.commit(message)


def push_changes(
    remote: str = "origin",
    branch: Optional[str] = None,
    repo_path: Union[str, Path] = ".",
) -> bool:
    """Push changes to remote."""
    ops = GitOperations(repo_path)
    return ops.push(remote, branch)
