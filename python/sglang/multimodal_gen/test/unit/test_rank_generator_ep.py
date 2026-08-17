"""The expert-parallel axis must be a real orthogonal dimension of the mesh.

EP was folded onto the TP group in Phase 1 (`ep_size == tp_size`). It is now its
own `RankGenerator` axis, so `num_gpus = dp * cfg * tp * ep * sp` and an EP group
is orthogonal to the TP/SP groups. These are pure combinatorics on the rank
factorizer -- no torch, no GPUs.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.utils.distributed import RankGenerator

_ORDER = "tp-ep-sp-pp-cfg-dp"


def _generator(*, tp=1, ep=1, sp=1, cfg=1, dp=1):
    return RankGenerator(tp=tp, sp=sp, pp=1, cfg=cfg, dp=dp, order=_ORDER, ep=ep)


def _assert_partition(groups, world_size, group_size):
    """Every rank appears in exactly one group of the expected size."""
    assert all(len(g) == group_size for g in groups), groups
    flat = [r for g in groups for r in g]
    assert sorted(flat) == list(range(world_size)), groups
    assert len(groups) == world_size // group_size, groups


def test_ep_axis_enters_world_size():
    gen = _generator(tp=2, ep=4)
    assert gen.world_size == 8
    assert gen.name_to_size["ep"] == 4


def test_ep_groups_partition_the_world():
    # 8 = tp(2) x ep(4): four TP groups of 2, two EP groups of 4.
    gen = _generator(tp=2, ep=4)
    _assert_partition(gen.get_ranks("ep"), world_size=8, group_size=4)
    _assert_partition(gen.get_ranks("tp"), world_size=8, group_size=2)


def test_ep_and_tp_are_orthogonal():
    # Orthogonality: a rank's TP group and EP group meet in exactly that rank.
    gen = _generator(tp=2, ep=4)
    ep_groups = gen.get_ranks("ep")
    tp_groups = gen.get_ranks("tp")
    for rank in range(gen.world_size):
        ep = next(g for g in ep_groups if rank in g)
        tp = next(g for g in tp_groups if rank in g)
        assert set(ep) & set(tp) == {rank}, (rank, ep, tp)


def test_ep_composes_with_sp():
    # The target layout: 8 = ep(4) x sp(2), tp folded to 1.
    gen = _generator(ep=4, sp=2)
    _assert_partition(gen.get_ranks("ep"), world_size=8, group_size=4)
    _assert_partition(gen.get_ranks("sp"), world_size=8, group_size=2)


def test_ep_disabled_is_a_singleton_axis():
    gen = _generator(tp=2, sp=2)
    assert gen.world_size == 4
    # ep=1 -> every rank is its own EP group, i.e. experts fully replicated.
    _assert_partition(gen.get_ranks("ep"), world_size=4, group_size=1)
