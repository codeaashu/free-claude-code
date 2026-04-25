from __future__ import annotations

import re
from pathlib import Path


def test_architecture_plan_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    plan = repo_root / "PLAN.md"

    assert plan.exists()
    text = plan.read_text(encoding="utf-8")
    assert "Intended Dependency Direction" in text
    assert "Smoke Coverage Policy" in text


def test_env_examples_are_in_sync() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_example = repo_root / ".env.example"
    packaged_example = repo_root / "config" / "env.example"

    assert _env_keys(root_example) == _env_keys(packaged_example)


def test_pyproject_first_party_packages_match_packaged_roots() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"known-first-party = \[(?P<items>[^\]]+)\]", pyproject)

    assert match is not None
    configured = {
        item.strip().strip('"')
        for item in match.group("items").split(",")
        if item.strip()
    }
    expected = {"api", "cli", "config", "core", "messaging", "providers", "smoke"}
    assert configured == expected


def _env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, _value = stripped.partition("=")
        keys.add(key.strip())
    return keys
