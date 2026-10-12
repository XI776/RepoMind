"""Read public GitHub and local Git commits without checking out target code."""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

from repomind.models import Repository, RepositorySnapshot, stable_id

log = logging.getLogger(__name__)
_SKIP_DIRS = frozenset({".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", ".mypy_cache", "build", "dist"})
_GITHUB_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MAX_FILES = 2_000
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 20_000_000
MAX_CLONE_BYTES = 300_000_000  # Checked after clone; timeout is the transfer guard.
GIT_TIMEOUT = 120
INDEX_CONFIG_ID = "python-ast-v1"


class SourceError(ValueError):
    """The requested source is unsupported or invalid."""


class GitImportError(RuntimeError):
    """Git failed or the repository exceeded an import limit."""


@dataclass(frozen=True, slots=True)
class GitFile:
    path: str
    blob_oid: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ImportedRepository:
    repository: Repository
    snapshot: RepositorySnapshot
    files: tuple[GitFile, ...]
    warning: str | None = None


def _git_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"})
    return env


def _git(*args: str, cwd: Path | None = None, timeout: int = GIT_TIMEOUT) -> bytes:
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
               "-c", "submodule.recurse=false", "-c", "http.followRedirects=false", *args]
    try:
        result = subprocess.run(command, cwd=cwd, env=_git_env(), capture_output=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitImportError(f"Git command failed or timed out: {args[0]}") from exc
    if result.returncode:
        message = result.stderr.decode("utf-8", "replace").strip()[-500:]
        raise GitImportError(f"Git {args[0]} failed: {message}")
    return result.stdout


def _github_url(value: str) -> tuple[str, str]:
    parsed = urlsplit(value)
    try:
        invalid_port = parsed.port is not None
    except ValueError as exc:
        raise SourceError("Invalid GitHub URL port") from exc
    if (parsed.scheme != "https" or parsed.hostname != "github.com" or invalid_port
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise SourceError("Only public https://github.com/OWNER/REPO URLs are supported")
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
        raise SourceError("GitHub URL must identify a repository, not a file or branch")
    owner, name = parts
    name = name.removesuffix(".git")
    if not all(_GITHUB_PART.fullmatch(part) for part in (owner, name)) or name in {".", ".."}:
        raise SourceError("Invalid GitHub owner or repository name")
    return f"https://github.com/{owner}/{name}.git", name


class GitImporter:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.snapshots = self.workspace / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)

    def import_source(self, source: str) -> ImportedRepository:
        if "://" in source or source.startswith("git@"):
            clone_url, name = _github_url(source)
            source_type = "github"
            canonical = clone_url.removesuffix(".git").lower()
            clone_source = clone_url
            warning = None
        else:
            candidate = Path(source).expanduser().resolve()
            if not candidate.is_dir():
                raise SourceError(f"Local directory does not exist: {source}")
            try:
                root = Path(_git("rev-parse", "--show-toplevel", cwd=candidate).decode().strip()).resolve()
                _git("rev-parse", "--verify", "HEAD^{commit}", cwd=root)
                dirty = bool(_git("status", "--porcelain", "--untracked-files=normal", cwd=root).strip())
            except GitImportError as exc:
                raise SourceError("Local source must be a Git repository with a HEAD commit") from exc
            source_type = "local"
            canonical = str(root)
            name = root.name
            clone_source = canonical
            warning = "Uncommitted and untracked changes were not indexed" if dirty else None

        repo = Repository(stable_id(source_type, canonical), name, source_type, canonical)
        parent = self.snapshots / repo.id
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="clone-", dir=parent))
        try:
            args = ["clone", "--bare", "--no-tags", "--no-recurse-submodules"]
            if source_type == "github":
                args.extend(["--depth", "1", "--single-branch", "-c", "protocol.file.allow=never"])
            else:
                args.extend(["--no-local"])
            _git(*args, clone_source, str(staging), timeout=GIT_TIMEOUT)
            clone_size = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
            if clone_size > MAX_CLONE_BYTES:
                raise GitImportError("Cloned repository exceeds the disk limit")
            commit = _git("rev-parse", "--verify", "HEAD^{commit}", cwd=staging).decode().strip()
            final = parent / f"{commit}.git"
            if not final.exists():
                os.replace(staging, final)
                log.info("snapshot_created repo=%s commit=%s", repo.id, commit)
            snapshot = RepositorySnapshot(stable_id(repo.id, commit, INDEX_CONFIG_ID), repo.id, commit,
                                          INDEX_CONFIG_ID, str(final), datetime.now(timezone.utc).isoformat())
            files = self.list_python_files(snapshot)
            return ImportedRepository(repo, snapshot, files, warning)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def list_python_files(self, snapshot: RepositorySnapshot) -> tuple[GitFile, ...]:
        raw = _git("ls-tree", "-r", "-z", "-l", snapshot.commit_sha, cwd=Path(snapshot.git_dir))
        found: list[GitFile] = []
        total = 0
        for entry in raw.split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, oid, size = metadata.split()
            if kind != b"blob" or mode not in {b"100644", b"100755"}:
                continue
            try:
                path = raw_path.decode("utf-8")
            except UnicodeDecodeError:
                log.warning("skipped_non_utf8_path repo=%s", snapshot.repository_id)
                continue
            parts = PurePosixPath(path).parts
            if (not parts or path.startswith("/") or any(part in {".", ".."} for part in parts)
                    or any(part in _SKIP_DIRS for part in parts[:-1]) or not path.endswith(".py")):
                continue
            file_size = int(size)
            if file_size > MAX_FILE_BYTES:
                log.warning("skipped_large_file path=%s size=%d", path, file_size)
                continue
            total += file_size
            if len(found) >= MAX_FILES or total > MAX_TOTAL_BYTES:
                raise GitImportError("Python source exceeds the file count or byte limit")
            found.append(GitFile(path, oid.decode("ascii"), file_size))
        return tuple(found)

    def read_blob(self, snapshot: RepositorySnapshot, entry: GitFile) -> bytes:
        raw = _git("cat-file", "blob", entry.blob_oid, cwd=Path(snapshot.git_dir))
        if len(raw) != entry.size_bytes or len(raw) > MAX_FILE_BYTES:
            raise GitImportError(f"Git blob size changed or exceeded limit: {entry.path}")
        return raw
