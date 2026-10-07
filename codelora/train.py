"""``python -m codelora.train FRAGMENT... [key=value ...]`` (single GPU) or
``torchrun --nproc_per_node=N -m codelora.train FRAGMENT... [key=value ...]`` (single-node data parallel).

A fragment is a YAML path or a name under ``configs/`` (e.g. ``orders/standard-order1``); fragments are
merged left to right and ``key=value`` arguments override the result."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from . import distributed
from .config import load_config
from .runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "items", nargs="+", help="config fragments, then dotted overrides such as train.lr=5e-4 seed=43"
    )
    args = parser.parse_args(argv)
    fragments = [item for item in args.items if "=" not in item]
    overrides = [item for item in args.items if "=" in item]
    distributed.init_from_env()
    try:
        run(load_config(fragments, overrides))
    except BaseException:
        if not distributed.get_context().enabled:
            raise
        # A rank that dies must exit at once: a normal shutdown can block in NCCL teardown while the
        # other ranks wait in a collective.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    distributed.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
