from pathlib import Path

import pytest

from repomind.ingestion import GitImporter, SourceError
from repomind.service import RepoMindService
from conftest import git


def test_local_import_index_query_and_duplicate(git_repo: Path, tmp_path: Path) -> None:
    service = RepoMindService(tmp_path / "workspace")
    first = service.import_source(str(git_repo))
    assert first.file_count == 2
    assert first.symbol_count == 4
    assert first.parse_error_count == 0
    assert not first.duplicate
    assert first.snapshot.commit_sha == git(git_repo, "rev-parse", "HEAD")
    assert len(service.index.list_repositories()) == 1
    files = service.index.list_files(first.snapshot.id)
    assert {f.path for f in files} == {"pkg/mod.py", "pkg/util.py"}
    worker = service.index.find_symbols(first.repository.id, "Worker", "pkg/mod.py")
    assert len(worker) == 1
    assert worker[0].qualified_name == "pkg.mod.Worker"
    symbol, source = service.read_symbol(worker[0].id)
    assert symbol.start_line == 4 and source.startswith("class Worker:")
    assert "def run" in source
    imports = service.index.list_imports(files[0].id)
    assert len(imports) == 2
    assert imports[1].kind == "from" and imports[1].level == 1
    again = service.import_source(str(git_repo))
    assert again.duplicate
    assert len(service.index.list_files(first.snapshot.id)) == 2
    assert len(service.index.find_symbols(first.repository.id)) == 4


def test_dirty_worktree_does_not_change_snapshot(git_repo: Path, tmp_path: Path) -> None:
    service = RepoMindService(tmp_path / "workspace")
    initial = service.import_source(str(git_repo))
    (git_repo / "pkg" / "mod.py").write_text("def changed_only_in_worktree(): pass\n", encoding="utf-8")
    second = service.import_source(str(git_repo))
    assert second.duplicate and second.warning
    assert service.index.find_symbols(initial.repository.id, "changed_only_in_worktree") == ()
    original = service.index.find_symbols(initial.repository.id, "Worker")[0]
    assert "class Worker" in service.read_symbol(original.id)[1]


def test_new_commit_keeps_old_snapshot_readable(git_repo: Path, tmp_path: Path) -> None:
    service = RepoMindService(tmp_path / "workspace")
    first = service.import_source(str(git_repo))
    old_symbol = service.index.find_symbols(first.repository.id, "helper")[0]
    (git_repo / "pkg" / "util.py").write_text("def replacement():\n    return 7\n", encoding="utf-8")
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-qm", "update")
    second = service.import_source(str(git_repo))
    assert second.snapshot.id != first.snapshot.id
    assert service.index.get_active_snapshot(first.repository.id).id == second.snapshot.id
    assert not service.index.find_symbols(first.repository.id, "helper")
    assert service.index.find_symbols(first.repository.id, "replacement")
    assert "return 42" in service.read_symbol(old_symbol.id)[1]


def test_invalid_sources_and_uncommitted_repo(tmp_path: Path) -> None:
    importer = GitImporter(tmp_path / "workspace")
    for url in ("http://github.com/a/b", "https://github.com/a/b/tree/main",
                "https://example.com/a/b", "https://user:pw@github.com/a/b"):
        with pytest.raises(SourceError):
            importer.import_source(url)
    with pytest.raises(SourceError):
        importer.import_source(str(tmp_path))


def test_bad_python_is_recorded_and_symlink_is_skipped(git_repo: Path, tmp_path: Path) -> None:
    (git_repo / "pkg" / "broken.py").write_text("def nope(:\n", encoding="utf-8")
    (git_repo / "pkg" / "empty.py").write_bytes(b"")
    (git_repo / "pkg" / "linked.py").symlink_to("util.py")
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-qm", "edge cases")
    service = RepoMindService(tmp_path / "workspace")
    result = service.import_source(str(git_repo))
    files = {file.path: file for file in service.index.list_files(result.snapshot.id)}
    assert result.parse_error_count == 1
    assert files["pkg/broken.py"].parse_status == "syntax_error"
    assert files["pkg/broken.py"].error
    assert files["pkg/empty.py"].parse_status == "parsed"
    assert "pkg/linked.py" not in files


def test_local_fsmonitor_config_is_not_executed(git_repo: Path, tmp_path: Path) -> None:
    marker = tmp_path / "untrusted-hook-ran"
    hook = tmp_path / "fsmonitor.sh"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n", encoding="utf-8")
    hook.chmod(0o755)
    git(git_repo, "config", "core.fsmonitor", str(hook))
    service = RepoMindService(tmp_path / "workspace")
    assert service.import_source(str(git_repo)).file_count == 2
    assert not marker.exists()
