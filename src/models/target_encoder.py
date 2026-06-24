from __future__ import annotations

from .context_encoder import ContextEncoder


class TargetEncoder(ContextEncoder):
    """Same architecture as the context encoder, updated by EMA in NuJEPA."""
