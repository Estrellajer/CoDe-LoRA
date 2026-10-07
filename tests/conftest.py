import pytest

from tests import tinylab


@pytest.fixture(scope="session")
def lab(tmp_path_factory):
    """Offline tiny models and benchmarks shared by the whole session."""

    root = tmp_path_factory.mktemp("lab")
    return {
        "t5": tinylab.make_tiny_t5(root / "t5"),
        "qwen": tinylab.make_tiny_qwen(root / "qwen"),
        "data": tinylab.make_benchmark(root / "data"),
        "trace": tinylab.make_trace_benchmark(root / "trace"),
        "root": root,
    }
