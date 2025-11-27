"""
History-based bug generation strategy.

This strategy rewinds individual function/method bodies to an earlier Git
revision while keeping the current signature and decorators intact. It walks
back through history (up to a fixed depth; default 100) until it finds a
version whose patch causes tests to fail (via the provided harness), then
emits that candidate.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import astor
from swebench.harness.constants import FAIL_TO_PASS

from swesmith.bug_gen.adapters.python import _build_entity
from swesmith.bug_gen.utils import apply_code_change, get_patch
from swesmith.constants import LOG_DIR_BUG_GEN, PREFIX_BUG, PREFIX_METADATA, BugRewrite

HISTORY_STRATEGY = "history"
DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


@dataclass
class BugCandidate:
    """Represents a bug-inducing patch ready for validation."""
    patch: str
    metadata: dict
    strategy: str
    bug_path: Path | None = None
    metadata_path: Path | None = None

def _has_substantive_body(func: ast.FunctionDef) -> bool:
    """Return True if the function has a non-trivial body (beyond docstrings/pass)."""
    body = func.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return any(not isinstance(n, ast.Pass) for n in body)


def _get_docstring_node(func: ast.FunctionDef) -> ast.Expr | None:
    if (
        func.body
        and isinstance(func.body[0], ast.Expr)
        and isinstance(func.body[0].value, ast.Constant)
        and isinstance(func.body[0].value.value, str)
    ):
        return func.body[0]
    return None


def _sanitize_docstring_node(node: ast.Expr) -> ast.Expr:
    """Return a copy of a docstring Expr with escapes doubled to avoid SyntaxWarning."""
    new_node = copy.deepcopy(node)
    if isinstance(new_node.value, ast.Constant) and isinstance(new_node.value.value, str):
        new_node.value = ast.Constant(new_node.value.value.replace("\\", "\\\\"))
    return new_node


def _strip_doc_and_locations(fn: ast.FunctionDef) -> ast.FunctionDef:
    """Copy of a FunctionDef with docstring removed and position info stripped."""
    fn = copy.deepcopy(fn)

    # Remove docstring stmt if present
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        fn.body = fn.body[1:]

    for node in ast.walk(fn):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                setattr(node, attr, None)
    return fn


def _normalized_body_dump(fn: ast.FunctionDef) -> str:
    """Hashable form of a function without docstring/locations to spot real code changes."""
    return ast.dump(_strip_doc_and_locations(fn), include_attributes=False)


class LocalPytestHarness:
    """Lightweight harness that runs pytest locally to extract failing tests."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self._baseline_checked = False
        self._baseline_ok = False

    def ensure_baseline(self) -> bool:
        if self._baseline_checked:
            return self._baseline_ok
        self._baseline_checked = True
        self._baseline_ok, _ = self._run_pytest(return_failures=True)
        return self._baseline_ok

    def validate_candidate(self, candidate: BugCandidate):
        if not self._apply_patch(candidate):
            return False, None
        ok, failures = self._run_pytest(return_failures=True)
        self._reset_repo()
        if not ok and failures:
            return True, failures
        return False, failures

    def _run_pytest(self, return_failures: bool = False):
        result = subprocess.run(
            ["pytest", "-q", "--maxfail=10", "--disable-warnings"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        failures = self._parse_failures(result.stdout + "\n" + result.stderr)
        # pytest returns 0 for success, 1 for failures, 5 for no tests collected
        ok = result.returncode in (0, 5)
        if return_failures:
            return ok, failures
        return ok, []

    def _parse_failures(self, output: str) -> list[str]:
        failures = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("FAILED "):
                parts = line.split()
                if len(parts) >= 2:
                    failures.append(parts[1])
        return failures

    def _apply_patch(self, candidate: BugCandidate) -> bool:
        try:
            subprocess.run(
                ["git", "-C", str(self.repo_root), "apply", str(candidate.bug_path)],
                check=True,
                **DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _reset_repo(self):
        subprocess.run(["git", "-C", str(self.repo_root), "reset", "--hard"], **DEVNULL)
        subprocess.run(["git", "-C", str(self.repo_root), "clean", "-fdx"], **DEVNULL)


def _format_failure_info(failure_info: dict | list | str | None) -> str | None:
    """Normalize failure info into a short string."""
    if not failure_info:
        return None
    if isinstance(failure_info, dict):
        tests = failure_info.get(FAIL_TO_PASS) or failure_info.get("fail_to_pass") or []
        if tests:
            return f"Tests failing: {', '.join(map(str, tests))}"
        return str(failure_info)
    if isinstance(failure_info, list):
        return f"Tests failing: {', '.join(map(str, failure_info))}"
    return f"Tests failing: {failure_info}"


def _pos_arg_count(func: ast.FunctionDef) -> int:
    """Count positional (including positional-only) arguments."""
    posonly = getattr(func.args, "posonlyargs", [])
    return len(posonly) + len(func.args.args)


def _iter_function_entities(file_path: Path) -> Iterable[tuple[object, str]]:
    """Yield (CodeEntity, qualname) pairs for functions in a Python file."""
    try:
        file_content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(file_content, filename=str(file_path))
    except (OSError, SyntaxError):
        return []

    results: list[tuple[object, str]] = []

    def visit(node: ast.AST, parents: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, parents + [child.name])
            elif isinstance(child, ast.FunctionDef):
                qualname = ".".join(parents + [child.name]) if parents else child.name
                entity = _build_entity(child, file_content, str(file_path))
                results.append((entity, qualname))
                visit(child, parents + [child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return results


def _collect_prev_functions(prev_src: str) -> dict[str, ast.FunctionDef]:
    """Map qualified function names to FunctionDef nodes from previous source."""
    try:
        tree = ast.parse(prev_src)
    except SyntaxError:
        return {}

    mapping: dict[str, ast.FunctionDef] = {}

    def visit(node: ast.AST, parents: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, parents + [child.name])
            elif isinstance(child, ast.FunctionDef):
                qualname = ".".join(parents + [child.name]) if parents else child.name
                mapping[qualname] = child
                visit(child, parents + [child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return mapping


class HistoryBugStrategy:
    """Generate bugs by swapping current function bodies with the previous revision."""

    name = HISTORY_STRATEGY

    def __init__(
        self,
        repo: str | Path,
        harness=None,
        log_dir: Path | None = None,
        dirs_exclude: list[str] | None = None,
        dirs_include: list[str] | None = None,
        max_history_steps: int = 100,
        auto_harness: bool = False,
    ):
        self.harness = harness
        self.dirs_exclude = dirs_exclude or []
        self.dirs_include = dirs_include or []
        self.repo_profile = None
        self._cloned = False
        self.max_history_steps = max_history_steps

        repo_path = Path(repo)
        if repo_path.exists():
            self.repo_root = repo_path
            self.repo_name = repo_path.name
        else:
            from swesmith.profiles import registry

            self.repo_profile = registry.get(str(repo))
            repo_dir, cloned = self.repo_profile.clone()
            self.repo_root = Path(repo_dir)
            self.repo_name = self.repo_profile.repo_name
            self._cloned = cloned

        self.log_dir = log_dir or LOG_DIR_BUG_GEN / self.repo_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.harness is None and auto_harness:
            self.harness = LocalPytestHarness(self.repo_root)

    def _cleanup_repo(self):
        if self._cloned and self.repo_root.exists():
            shutil.rmtree(self.repo_root, ignore_errors=True)

    def _maybe_fetch_history(self) -> bool:
        """Fetch upstream history (if profile info is available) to make HEAD~k resolvable."""
        if not self.repo_profile:
            return False
        upstream_url = f"https://github.com/{self.repo_profile.owner}/{self.repo_profile.repo}.git"
        # Add upstream remote if needed
        remotes = subprocess.run(
            ["git", "-C", str(self.repo_root), "remote"],
            capture_output=True,
            text=True,
        )
        if remotes.returncode == 0 and "upstream" not in remotes.stdout.split():
            subprocess.run(
                ["git", "-C", str(self.repo_root), "remote", "add", "upstream", upstream_url],
                **DEVNULL,
            )
        # Fetch limited history to avoid huge downloads
        depth = max(self.max_history_steps + 1, 10)
        fetch_result = subprocess.run(
            ["git", "-C", str(self.repo_root), "fetch", "--depth", str(depth), "upstream"],
            **DEVNULL,
        )
        if fetch_result.returncode != 0:
            return False
        print(f"[history] Fetched upstream history (depth={depth}) for {self.repo_root}")
        # Align HEAD to the upstream commit (tree should be identical to mirror snapshot)
        subprocess.run(
            ["git", "-C", str(self.repo_root), "reset", "--hard", self.repo_profile.commit],
            **DEVNULL,
        )
        return True

    def _has_previous_commit(self) -> bool:
        for depth in range(1, self.max_history_steps + 1):
            cmd = [
                "git",
                "-C",
                str(self.repo_root),
                "rev-parse",
                f"HEAD~{depth}",
            ]
            if subprocess.run(cmd, **DEVNULL).returncode == 0:
                return True
        return False

    def _iter_candidate_files(self) -> Iterable[Path]:
        for root, _, files in os.walk(self.repo_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix != ".py":
                    continue
                if file_path.name.startswith("__") and file_path.name.endswith("__" + file_path.suffix):
                    continue
                if self._should_skip(root, file):
                    continue
                yield file_path

    def _should_skip(self, root: str, file: str) -> bool:
        if self.repo_profile and self.repo_profile._is_test_path(root, file):
            return True
        if not self.repo_profile:
            # Generic test heuristics if no repo profile is available
            lower = file.lower()
            if lower.startswith("test") or lower.endswith("_test.py") or "tests" in Path(
                root
            ).parts:
                return True
        if self.dirs_exclude and any(token in root for token in self.dirs_exclude):
            return True
        if self.dirs_include and not any(token in root for token in self.dirs_include):
            return True
        return False

    def _get_src_at_revision(self, rel_path: Path, depth: int) -> str | None:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(self.repo_root),
                "show",
                f"HEAD~{depth}:{rel_path.as_posix()}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout

    def _build_rewrite(self, current: ast.FunctionDef, previous: ast.FunctionDef) -> str | None:
        doc_node = _get_docstring_node(current) or _get_docstring_node(previous)
        prev_body = list(previous.body)
        if _get_docstring_node(previous):
            prev_body = prev_body[1:]

        new_body = []
        if doc_node is not None:
            new_body.append(_sanitize_docstring_node(doc_node))
        new_body.extend(copy.deepcopy(node) for node in prev_body)
        if not new_body:
            return None

        replacement = copy.deepcopy(current)
        replacement.body = new_body

        # IMPORTANT: do not print decorators here; they live outside entity span
        replacement.decorator_list = []

        ast.fix_missing_locations(replacement)
        try:
            return astor.to_source(replacement).strip()
        except Exception:
            return None

    def _materialize_candidate(
        self,
        entity,
        rewrite: str,
        qualname: str,
        relative_path: Path,
        previous_revision: str,
        failure_info: dict | list | str | None = None,
    ) -> BugCandidate | None:
        expl = _format_failure_info(failure_info) or (
            "Tests are failing after this change; inspect validation logs for failing test names and errors."
        )
        bug = BugRewrite(
            rewrite=rewrite,
            explanation=expl,
            strategy=self.name,
        )
        sig_hash = hashlib.sha256(entity.signature.encode()).hexdigest()[:8]
        bug_hash = bug.get_hash()
        file_dir_name = f"{self.repo_name}__{str(relative_path).replace(os.sep, '__')}"
        bug_dir = (
            self.log_dir
            / file_dir_name
            / f"{entity.name}_{sig_hash}"
        )
        bug_dir.mkdir(parents=True, exist_ok=True)

        uuid_str = f"{self.name}__{bug_hash}"
        metadata = {
            **bug.to_dict(),
            "file_path": str(relative_path),
            "function": qualname,
            "previous_revision": previous_revision,
        }
        if failure_info is not None:
            metadata["failure_info"] = failure_info
        metadata_path = bug_dir / f"{PREFIX_METADATA}__{uuid_str}.json"
        bug_path = bug_dir / f"{PREFIX_BUG}__{uuid_str}.diff"

        try:
            metadata_path.write_text(json.dumps(metadata, indent=2))
            apply_code_change(entity, bug)
            patch = get_patch(str(self.repo_root), reset_changes=True)
            if not patch:
                raise ValueError("Empty patch")
            bug_path.write_text(patch)
        except Exception:
            metadata_path.unlink(missing_ok=True)
            bug_path.unlink(missing_ok=True)
            # Clean up empty directory to avoid clutter
            try:
                if not any(bug_dir.iterdir()):
                    bug_dir.rmdir()
            except FileNotFoundError:
                pass
            return None

        return BugCandidate(
            patch=patch,
            metadata=metadata,
            strategy=self.name,
            bug_path=bug_path,
            metadata_path=metadata_path,
        )

    def generate_candidates(self, max_candidates: int = -1) -> list[BugCandidate]:
        if not self._has_previous_commit():
            fetched = self._maybe_fetch_history()
            if not self._has_previous_commit():
                print(
                    f"[history] No prior commits within {self.max_history_steps} steps "
                    f"for {self.repo_root}.{' Tried fetching upstream history.' if fetched else ''}"
                )
                self._cleanup_repo()
                return []

        if self.harness and hasattr(self.harness, "ensure_baseline"):
            if not self.harness.ensure_baseline():
                print("[history] Baseline check failed; aborting history generation.")
                self._cleanup_repo()
                return []

        candidates: list[BugCandidate] = []
        try:
            file_paths = list(self._iter_candidate_files())
            if not file_paths:
                print(f"[history] No candidate files found in {self.repo_root}")
                return []

            print(f"[history] Scanning {len(file_paths)} files in {self.repo_root}")
            for idx, file_path in enumerate(file_paths, start=1):
                rel_path = Path(os.path.relpath(file_path, self.repo_root))
                print(f"[history] [{idx}/{len(file_paths)}] {rel_path}")
                history_funcs = [
                    (depth, funcs)
                    for depth in range(1, self.max_history_steps + 1)
                    if (src := self._get_src_at_revision(rel_path, depth))
                    and (funcs := _collect_prev_functions(src))
                ]
                if not history_funcs:
                    continue
                for entity, qualname in _iter_function_entities(file_path):
                    if not isinstance(entity.node, ast.FunctionDef):
                        continue
                    if not _has_substantive_body(entity.node):
                        continue
                    curr_norm = _normalized_body_dump(entity.node)

                    for depth, prev_funcs in history_funcs:
                        prev_node = prev_funcs.get(qualname)
                        if not prev_node:
                            continue
                        if _pos_arg_count(prev_node) != _pos_arg_count(entity.node):
                            continue
                        if not _has_substantive_body(prev_node):
                            continue
                        prev_norm = _normalized_body_dump(prev_node)
                        if curr_norm == prev_norm:
                            continue

                        rewrite = self._build_rewrite(entity.node, prev_node)
                        if not rewrite:
                            continue

                        prev_rev = f"HEAD~{depth}"
                        candidate = self._materialize_candidate(
                            entity, rewrite, qualname, rel_path, prev_rev, None
                        )
                        if not candidate:
                            continue

                        if self.harness and hasattr(self.harness, "validate_candidate"):
                            result = self.harness.validate_candidate(candidate)
                            proceed, failure_info = (
                                (result[0], result[1])
                                if isinstance(result, (list, tuple)) and len(result) >= 2
                                else (bool(result), None)
                            )
                            if not proceed:
                                continue
                            if failure_info is not None:
                                candidate.metadata["failure_info"] = failure_info
                            formatted = _format_failure_info(failure_info)
                            if formatted:
                                candidate.metadata["explanation"] = formatted
                                if candidate.metadata_path:
                                    candidate.metadata_path.write_text(
                                        json.dumps(candidate.metadata, indent=2)
                                    )

                        candidates.append(candidate)
                        print(
                            f"[history] Found candidate for {qualname} at {prev_rev} "
                            f"in {rel_path}"
                        )
                        if max_candidates != -1 and len(candidates) >= max_candidates:
                            return candidates
                        break
        finally:
            self._cleanup_repo()

        return candidates


def main(
    repo: str,
    max_bugs: int = -1,
    dirs_exclude: list[str] | None = None,
    dirs_include: list[str] | None = None,
):
    strategy = HistoryBugStrategy(
        repo,
        dirs_exclude=dirs_exclude,
        dirs_include=dirs_include,
    )
    candidates = strategy.generate_candidates(max_candidates=max_bugs)
    print(
        f"Generated {len(candidates)} history candidates for {strategy.repo_name} at {strategy.log_dir}"
    )
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate bugs by restoring function bodies from earlier commits "
            "until tests fail (searching backward up to max_history_steps)."
        )
    )
    parser.add_argument(
        "repo",
        type=str,
        help="Repository name (registered in profiles) or path to a local git repo.",
    )
    parser.add_argument(
        "--max_bugs",
        type=int,
        default=-1,
        help="Maximum number of history-based bugs to emit (default: unlimited).",
    )
    parser.add_argument(
        "--dirs_exclude",
        nargs="+",
        default=[],
        help="Directory name filters to exclude.",
    )
    parser.add_argument(
        "--dirs_include",
        nargs="+",
        default=[],
        help="Directory name filters to include (if set, only these are considered).",
    )
    args = parser.parse_args()
    main(**vars(args))
