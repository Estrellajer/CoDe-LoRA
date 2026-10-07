import pytest
import torch
import torch.nn.functional as F

from codelora.config import RouterCfg
from codelora.routing.prototype import build_prototype
from codelora.routing.router import ROUTERS, Router, build_router, register_router


def test_prototype_is_a_normalised_mean_of_normalised_embeddings():
    support = torch.tensor([[3.0, 0.0], [0.0, 0.1]])
    prototype = build_prototype(support)
    assert torch.allclose(prototype.norm(), torch.tensor(1.0))
    assert torch.allclose(prototype, F.normalize(torch.tensor([1.0, 1.0]), dim=0))  # not dominated by the long vector


def test_cosine_router_returns_best_task_and_clamped_cosine():
    router = build_router(RouterCfg(kind="cosine"))
    router.add_task("a", torch.tensor([[1.0, 0.0]]))
    router.add_task("b", torch.tensor([[0.0, 1.0]]))
    task, confidence = router.route(torch.tensor([0.9, 0.1]))
    assert task == "a" and 0.9 < confidence <= 1.0
    task, confidence = router.route(torch.tensor([0.0, 5.0]))
    assert task == "b" and confidence <= 1.0


def test_lda_router_separates_gaussian_tasks_and_confidence_is_a_posterior():
    torch.manual_seed(0)
    router = build_router(RouterCfg(kind="lda", lda_shrinkage=0.01))
    centres = {
        "a": torch.tensor([4.0, 0.0, 0.0]),
        "b": torch.tensor([0.0, 4.0, 0.0]),
        "c": torch.tensor([0.0, 0.0, 4.0]),
    }
    for task, centre in centres.items():
        router.add_task(task, centre + torch.randn(200, 3))
    for task, centre in centres.items():
        hits = sum(router.route(centre + torch.randn(3))[0] == task for _ in range(50))
        assert hits >= 48
    _, confidence = router.route(centres["a"])
    assert 1 / 3 < confidence <= 1.0


@pytest.mark.parametrize("kind", ["cosine", "lda"])
def test_router_state_round_trip(kind):
    router = build_router(RouterCfg(kind=kind))
    router.add_task("a", torch.randn(20, 4))
    router.add_task("b", torch.randn(20, 4) + 3)
    restored = build_router(RouterCfg(kind=kind))
    restored.load_state_dict(router.state_dict())
    query = torch.randn(4)
    assert restored.route(query) == router.route(query)


def test_a_registered_router_is_selected_by_kind():
    @register_router("always-first")
    class AlwaysFirst(Router):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.tasks = []

        def add_task(self, task, support):
            self.tasks.append(task)

        def route(self, embedding):
            return self.tasks[0], 1.0

    try:
        assert isinstance(build_router(RouterCfg(kind="always-first")), AlwaysFirst)
    finally:
        del ROUTERS["always-first"]
