from pathlib import Path
import subprocess

import pytest


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git(repo, "config", "user.name", "RepoMind Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "mod.py").write_text(
        "import os\nfrom . import util\n\n"
        "class Worker:\n    def run(self, value: int) -> int:\n        return value + 1\n\n"
        "async def fetch():\n    return 1\n", encoding="utf-8")
    (repo / "pkg" / "util.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "ignored.py").write_text("def ignored(): pass\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "ignored.py").write_text("def ignored(): pass\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "initial")
    return repo
