"""Process launchers for CPU tests (gloo)."""

from __future__ import annotations

import socket
import sys


def python(world: int = 1) -> list[str]:
    """``python`` or a ``torch.distributed.run`` launcher with ``world`` local ranks.

    The rendezvous address is pinned to 127.0.0.1: ``--standalone`` resolves the machine's hostname, which
    hangs on hosts whose hostname is not resolvable (e.g. some macOS setups).
    """

    if world == 1:
        return [sys.executable]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--master_addr=127.0.0.1", f"--master_port={port}", f"--nproc_per_node={world}",
    ]  # fmt: skip
