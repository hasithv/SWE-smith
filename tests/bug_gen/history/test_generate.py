import subprocess
import textwrap

from pathlib import Path

from swesmith.bug_gen.history.generate import HISTORY_STRATEGY, HistoryBugStrategy


def _run_git(commands: list[list[str]], cwd: Path):
    for cmd in commands:
        subprocess.run(
            cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


def _init_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _run_git([["git", "init"]], cwd=repo_root)
    _run_git(
        [
            ["git", "config", "user.email", "test@example.com"],
            ["git", "config", "user.name", "History Tester"],
        ],
        cwd=repo_root,
    )
    return repo_root


def _commit_file(repo_root: Path, content: str, message: str):
    sample_path = repo_root / "sample.py"
    sample_path.write_text(textwrap.dedent(content), encoding="utf-8")
    _run_git([["git", "add", "sample.py"]], cwd=repo_root)
    _run_git([["git", "commit", "-m", message]], cwd=repo_root)


class DummyHarness:
    def __init__(self, failure_marker: str):
        self.failure_marker = failure_marker
        self.calls: list[str] = []

    def ensure_baseline(self) -> bool:
        return True

    def validate_candidate(self, candidate):
        self.calls.append(candidate.patch)
        if self.failure_marker in candidate.patch:
            return True
        return False


def test_harness_walks_until_failure(tmp_path):
    repo_root = _init_repo(tmp_path)
    _commit_file(
        repo_root,
        """
        def compute():
            return 0
        """,
        "v0",
    )
    _commit_file(
        repo_root,
        """
        def compute():
            return 1
        """,
        "v1",
    )
    _commit_file(
        repo_root,
        """
        def compute():
            return 2
        """,
        "v2-current",
    )

    harness = DummyHarness("return 0")
    strategy = HistoryBugStrategy(
        repo_root, log_dir=tmp_path / "logs", harness=harness, max_history_steps=5
    )
    candidates = strategy.generate_candidates()
    assert candidates, "Expected history candidate when harness finds a failing patch"
    candidate = candidates[0]
    assert candidate.metadata["previous_revision"] == "HEAD~2"
    assert "return 0" in candidate.patch
    assert harness.calls, "Harness should be invoked"
    assert "failure_info" not in candidate.metadata


def test_no_harness_uses_first_available_revision(tmp_path):
    repo_root = _init_repo(tmp_path)
    _commit_file(
        repo_root,
        """
        def greet():
            return "old"
        """,
        "old",
    )
    _commit_file(
        repo_root,
        """
        def greet():
            return "new"
        """,
        "new",
    )

    strategy = HistoryBugStrategy(repo_root, log_dir=tmp_path / "logs")
    candidates = strategy.generate_candidates()
    assert candidates
    candidate = candidates[0]
    assert candidate.strategy == HISTORY_STRATEGY
    assert candidate.metadata["previous_revision"] == "HEAD~1"
    assert "-    return \"new\"" in candidate.patch or "-    return 'new'" in candidate.patch
    assert "+    return \"old\"" in candidate.patch or "+    return 'old'" in candidate.patch


def test_skips_docstring_only_changes(tmp_path):
    repo_root = _init_repo(tmp_path)
    _commit_file(
        repo_root,
        """
        def greet():
            \"\"\"Doc v1\"\"\"
            return 1
        """,
        "v1",
    )
    _commit_file(
        repo_root,
        """
        def greet():
            \"\"\"Doc v2\"\"\"
            return 1
        """,
        "doc change only",
    )

    strategy = HistoryBugStrategy(repo_root, log_dir=tmp_path / "logs")
    candidates = strategy.generate_candidates()
    assert candidates == [], "Docstring-only changes should be ignored"


def test_skips_if_no_history(tmp_path):
    repo_root = _init_repo(tmp_path)
    _commit_file(
        repo_root,
        """
        def helper(x):
            return x + 1
        """,
        "initial",
    )

    strategy = HistoryBugStrategy(repo_root, log_dir=tmp_path / "logs", max_history_steps=1)
    candidates = strategy.generate_candidates()
    assert candidates == [], "Single-commit repo should not yield history candidates"
