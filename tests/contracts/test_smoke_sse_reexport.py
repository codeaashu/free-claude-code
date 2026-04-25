"""Ensure smoke re-exports stay aligned with :mod:`core.anthropic.stream_contracts`."""

from __future__ import annotations

import core.anthropic.stream_contracts as core_sc
import smoke.lib.sse as smoke_sse


def test_smoke_lib_sse_reexports_core_stream_contracts() -> None:
    for name in smoke_sse.__all__:
        assert getattr(smoke_sse, name) is getattr(core_sc, name)
