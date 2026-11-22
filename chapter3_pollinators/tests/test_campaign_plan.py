"""The campaign plan: tasks, shared warm-ups, dependencies, dispatch order and cost."""

from __future__ import annotations

import pytest

from chapter3_pollinators.campaign.plan import (
    StepCosts,
    Task,
    build_plan,
    dispatch_key,
    estimate_seconds,
    parse_task_id,
    ready_tasks,
)


def test_full_plan_counts() -> None:
    plan = build_plan(seeds=range(10), group3_seeds=range(10))
    kinds = [t.kind for t in plan]
    assert kinds.count("warmup") == 100
    assert kinds.count("no_regulator") == 100
    assert kinds.count("branch") == 160 + 120 + 120 + 120 + 80
    assert len({t.task_id for t in plan}) == len(plan)


def test_group3_with_five_seeds() -> None:
    plan = build_plan(seeds=range(10), group3_seeds=range(5))
    hm = [t for t in plan if t.community.startswith("HM")]
    assert {t.seed for t in hm} == set(range(5))
    assert sum(t.kind == "warmup" for t in hm) == 30
    assert sum(t.kind == "branch" for t in hm) == 120


def test_warmups_and_no_regulator_runs_are_shared_between_governments() -> None:
    plan = build_plan(seeds=[0], group3_seeds=[0])
    ht1 = [t for t in plan if t.community == "HT1"]
    assert sum(t.kind == "warmup" for t in ht1) == 1
    assert sum(t.kind == "no_regulator" for t in ht1) == 1
    branches = {(t.government, t.condition) for t in ht1 if t.kind == "branch"}
    assert branches == {
        ("HTC", "reward_shape"), ("HTC", "oraa_mb"), ("HTC", "oraa_omni"), ("HTC", "ltos"),
        ("HMC", "reward_shape"), ("HMC", "oraa_mb"), ("HMC", "oraa_omni"),
        ("HTC", "oraa_mb_b3"), ("HTC", "oraa_omni_b3"),
    }
    hm3 = {(t.government, t.condition) for t in plan if t.community == "HM3" and t.kind == "branch"}
    assert hm3 == {("HMC", "reward_shape"), ("HMC", "oraa_mb"), ("HTC", "reward_shape"), ("HTC", "oraa_mb")}


def test_priorities_follow_the_protocol_groups() -> None:
    plan = {t.task_id: t for t in build_plan(seeds=[0], group3_seeds=[0])}
    assert plan["warmup:HT1:s0"].priority == 1
    assert plan["no_regulator:HT1:s0"].priority == 1
    assert plan["branch:HT1:HMC:oraa_omni:s0"].priority == 2
    assert plan["branch:HM2:HTC:oraa_mb:s0"].priority == 3
    assert plan["warmup:HM2:s0"].priority == 3


def test_task_ids_round_trip() -> None:
    for task in build_plan(seeds=[3], group3_seeds=[3]):
        assert parse_task_id(task.task_id) == task


def test_dependencies_and_readiness() -> None:
    plan = build_plan(seeds=[0], group3_seeds=[0])
    branch = next(t for t in plan if t.kind == "branch")
    assert branch.dependency == f"warmup:{branch.community}:s0"
    assert next(t for t in plan if t.kind == "warmup").dependency is None
    ready = ready_tasks(plan, done=set(), running=set())
    assert {t.kind for t in ready} == {"warmup"}
    ready = ready_tasks(plan, done={"warmup:HT1:s0"}, running={"branch:HT1:HTC:ltos:s0"})
    ht1 = {t.task_id for t in ready if t.community == "HT1"}
    assert "branch:HT1:HTC:ltos:s0" not in ht1
    assert "branch:HT1:HTC:oraa_mb:s0" in ht1 and "no_regulator:HT1:s0" in ht1


def test_dispatch_order_runs_warmups_first_then_seed_by_seed() -> None:
    costs = StepCosts(warmup=0.6, no_regulator=0.3, reward_shape=0.3, oraa_mb=6.0, oraa_omni=5.8, ltos=0.35)
    plan = build_plan(seeds=[0, 1], group3_seeds=[0, 1])
    order = sorted(plan, key=lambda t: dispatch_key(t, costs))
    assert [t.priority for t in order] == sorted(t.priority for t in order)
    group1 = [t for t in order if t.priority == 1]
    first_run = next(i for i, t in enumerate(group1) if t.kind != "warmup")
    assert all(t.kind == "warmup" for t in group1[:first_run])
    assert [t.seed for t in group1[:first_run]] == sorted(t.seed for t in group1[:first_run])
    runs = group1[first_run:]
    assert all(t.kind != "warmup" for t in runs)
    first_seed1 = next(i for i, t in enumerate(runs) if t.seed == 1)
    assert all(t.seed == 0 for t in runs[:first_seed1])
    assert runs[0].condition.startswith("oraa")


def test_cost_estimate_scales_with_run_lengths() -> None:
    costs = StepCosts(warmup=1.0, no_regulator=1.0, reward_shape=1.0, oraa_mb=1.0, oraa_omni=1.0, ltos=1.0)
    assert estimate_seconds(Task("warmup", "HT1", 0), costs) == pytest.approx(7015, rel=0.01)
    assert estimate_seconds(Task("branch", "HT1", 0, "HTC", "oraa_mb"), costs) == 1200
    assert estimate_seconds(Task("no_regulator", "HT1", 0), costs) == pytest.approx(6199, rel=0.01)


def test_group_four_adds_the_bound_branches_after_everything_else() -> None:
    from collections import Counter

    from chapter3_pollinators.campaign.plan import StepCosts, estimate_seconds

    plan = build_plan(range(10), range(5))
    by_priority = Counter(t.priority for t in plan)
    assert dict(by_priority) == {1: 240, 2: 120, 3: 180, 4: 80}
    group4 = [t for t in plan if t.priority == 4]
    assert {t.condition for t in group4} == {"oraa_mb_b3", "oraa_omni_b3"}
    assert {t.government for t in group4} == {"HTC"} and {t.kind for t in group4} == {"branch"}
    assert {t.community for t in group4} == {"HT1", "HT2", "HT3", "HT4"}
    assert all(t.dependency == f"warmup:{t.community}:s{t.seed}" for t in group4)
    warmups = {t.task_id: t.priority for t in plan if t.kind == "warmup"}
    assert all(warmups[t.dependency] == 1 for t in group4)
    costs = StepCosts(warmup=0.6, no_regulator=0.3, reward_shape=0.3, oraa_mb=7.0, oraa_omni=6.0, ltos=0.3)
    mb3 = next(t for t in group4 if t.condition == "oraa_mb_b3")
    omni3 = next(t for t in group4 if t.condition == "oraa_omni_b3")
    assert estimate_seconds(mb3, costs) == 1200 * 7.0 and estimate_seconds(omni3, costs) == 1200 * 6.0
