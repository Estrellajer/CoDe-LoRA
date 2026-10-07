"""Tables and figures from run directories (artifact schema: docs/RESULTS.md); CLI: ``python -m codelora.analysis``."""

from .loader import Run, discover, group_runs

__all__ = ["Run", "discover", "group_runs"]
