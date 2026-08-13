"""Every third-party import in `src/` must be a declared dependency.

This exists because it did not, and CI caught it the hard way: `omegaconf` and
`hydra-core` were imported by `config.py` and `cli.py` but never declared, so a
fresh `pip install -e .` produced an install that failed at import. It worked
locally only because those packages had been pip-installed by hand into the
development environment.

That failure mode is invisible to a normal test suite — the tests pass in the
environment where someone already installed the missing package. The check has
to compare the *source's* imports against the *manifest*, which is what this
does, in milliseconds and with no network.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# Imports that are deliberately optional. Each must be guarded at its import
# site with an error message telling the user which extra to install.
OPTIONAL_IMPORTS = {"datasets"}

# Fallback module -> distribution mapping, used when importlib cannot resolve a
# module because it is not installed in the current environment.
MODULE_TO_DISTRIBUTION = {
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "yaml": "pyyaml",
    "torch": "torch",
    "numpy": "numpy",
    "regex": "regex",
    "datasets": "datasets",
}


def _declared_dependencies() -> set[str]:
    """Distribution names from [project.dependencies], normalized."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    declared = set()
    for spec in pyproject["project"]["dependencies"]:
        # Strip version specifiers and extras: "hydra-core>=1.3" -> "hydra-core"
        name = spec.split(";")[0].strip()
        for sep in ("[", "=", ">", "<", "!", "~", " "):
            name = name.split(sep)[0]
        declared.add(name.strip().lower().replace("_", "-"))
    return declared


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative (first-party) import.
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _third_party_imports() -> dict[str, set[str]]:
    """Map third-party module name -> set of files importing it."""
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for module in _top_level_imports(path):
            if module in sys.stdlib_module_names or module == "microlab":
                continue
            found.setdefault(module, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


def _distribution_for(module: str) -> str:
    installed = packages_distributions().get(module)
    if installed:
        return installed[0].lower().replace("_", "-")
    return MODULE_TO_DISTRIBUTION.get(module, module).lower().replace("_", "-")


def test_all_third_party_imports_are_declared():
    declared = _declared_dependencies()
    undeclared: dict[str, set[str]] = {}

    for module, files in _third_party_imports().items():
        if module in OPTIONAL_IMPORTS:
            continue
        if _distribution_for(module) not in declared:
            undeclared[module] = files

    assert not undeclared, (
        "imports in src/ that are not declared in [project.dependencies]:\n"
        + "\n".join(f"  {m} (in {', '.join(sorted(f))})" for m, f in sorted(undeclared.items()))
        + "\nA fresh `pip install -e .` would fail at import."
    )


def test_declared_dependencies_are_actually_imported():
    """Flag dependencies nothing imports.

    A spurious dependency is a smaller problem than a missing one, but it still
    costs install time on every Kaggle session and misleads anyone reading the
    manifest to understand what the project needs. `pyyaml` was declared here
    for a while although nothing imported it — OmegaConf handles the YAML.
    """
    imported = {_distribution_for(m) for m in _third_party_imports()}
    unused = _declared_dependencies() - imported
    assert not unused, f"declared but never imported in src/: {sorted(unused)}"


@pytest.mark.parametrize("module", sorted(OPTIONAL_IMPORTS))
def test_optional_imports_are_guarded(module: str):
    """An optional import must fail with instructions, not a bare ImportError."""
    importing_files = _third_party_imports().get(module, set())
    assert importing_files, f"{module} is listed as optional but nothing imports it"
    for rel in importing_files:
        source = (REPO_ROOT / rel).read_text()
        assert "ImportError" in source and "pip install" in source, (
            f"{rel} imports optional dependency {module!r} without a guarded, "
            "actionable error message"
        )
