"""Canonical public API: one model and one graph type."""

from .edge_free_api import ELA
from .graph import ELAGraph

__all__ = ["ELA", "ELAGraph"]
