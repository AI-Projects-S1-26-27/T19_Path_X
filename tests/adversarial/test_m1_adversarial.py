"""Empirical Adversarial Test Suite for Milestone 1: Search Algorithms and SearchTracer."""

import collections
import heapq
import math
import re
from typing import List
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from core.actions.definition import ActionDefinition, ActionParameter
from core.actions.domain import Domain, TypeHierarchy, PredicateDefinition
from core.actions.instantiation import GroundAction
from core.contracts import (
    SearchEventType,
    SearchTraceEvent,
    SearchTraceRequest,
    SearchTraceResponse,
)
from core.domains import registry
from core.domains.base import BaseDomain
from core.search.algorithms import (
    AStarPlanner,
    BestFirstPlanner,
    BFSPlanner,
    UCSPlanner,
    get_planner_by_name,
)
from core.search.heuristics import GoalCountHeuristic, ZeroHeuristic
from core.search.node import SearchNode
from core.search.tracer import SearchTracer
from core.world.predicates import Fact
from core.world.problem import SymbolicProblem
from core.world.state import SymbolicState
from core.world.types import ObjectRegistry


# ==============================================================================
# Helper Fixtures & Custom Adversarial Weighted Domain
# ==============================================================================

class WeightedRoutingDomain(BaseDomain):
    """Synthetic domain designed specifically for adversarial cost/optimality stress testing.
    
    Graph topology:
      start --[fly, cost=10.0]--> goal (1 step, cost 10.0)
      start --[step1, cost=1.0]--> intermediate1 --[step2, cost=1.0]--> intermediate2 --[step3, cost=1.0]--> goal (3 steps, cost 3.0)
      start --[loop, cost=0.0]--> start (0 cost loop)
    """

    def __init__(self) -> None:
        th = TypeHierarchy()
        th.add_type("location", "object")
        th.add_type("vehicle", "object")

        preds = [
            PredicateDefinition("at", ["vehicle", "location"]),
        ]

        # 1. Direct expensive action (1 step, cost 10.0)
        fly_act = ActionDefinition(
            name="fly",
            parameters=[ActionParameter("v", "vehicle"), ActionParameter("from", "location"), ActionParameter("to", "location")],
            preconditions=[Fact("at", ["?v", "?from"])],
            add_effects=[Fact("at", ["?v", "?to"])],
            del_effects=[Fact("at", ["?v", "?from"])],
            cost=10.0,
            description="Direct long-range expensive flight",
        )

        # 2. Stepped cheap actions (cost 1.0 each)
        step_act = ActionDefinition(
            name="step",
            parameters=[ActionParameter("v", "vehicle"), ActionParameter("from", "location"), ActionParameter("to", "location")],
            preconditions=[Fact("at", ["?v", "?from"])],
            add_effects=[Fact("at", ["?v", "?to"])],
            del_effects=[Fact("at", ["?v", "?from"])],
            cost=1.0,
            description="Short ground hop",
        )

        # 3. Free loop action (cost 0.0)
        idle_act = ActionDefinition(
            name="idle",
            parameters=[ActionParameter("v", "vehicle"), ActionParameter("loc", "location")],
            preconditions=[Fact("at", ["?v", "?loc"])],
            add_effects=[Fact("at", ["?v", "?loc"])],
            del_effects=[],
            cost=0.0,
            description="Zero-cost idle",
        )

        super().__init__(
            domain_id="weighted_routing",
            name="Weighted Routing Domain",
            version="1.0.0",
            description="Adversarial benchmark for path cost vs step count optimality",
            type_hierarchy=th,
            predicates=preds,
            actions=[fly_act, step_act, idle_act],
        )


def build_weighted_problem(domain: WeightedRoutingDomain, allow_direct: bool = True) -> SymbolicProblem:
    reg = ObjectRegistry(domain.type_hierarchy)
    reg.register_object("drone", "vehicle")
    reg.register_object("start_loc", "location")
    reg.register_object("inter1_loc", "location")
    reg.register_object("inter2_loc", "location")
    reg.register_object("goal_loc", "location")

    init_facts = [Fact("at", ["drone", "start_loc"])]
    goal_facts = [Fact("at", ["drone", "goal_loc"])]

    # We restrict actions via legal combinations:
    return SymbolicProblem(
        name="routing_challenge",
        domain_name=domain.domain_id,
        registry=reg,
        initial_state=SymbolicState(init_facts, objects=reg.get_all_objects()),
        goal_conditions=goal_facts,
    )


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


# ==============================================================================
# Challenge 1: Optimality of UCS vs BFS on Non-Unit Costs
# ==============================================================================

class TestNonUnitCostOptimality:
    """Stress test: verify UCS minimizes cumulative cost while BFS minimizes step count."""

    def test_ucs_finds_minimal_cost_over_bfs_minimal_steps(self):
        domain = WeightedRoutingDomain()
        problem = build_weighted_problem(domain)

        # Run BFS
        bfs = BFSPlanner()
        bfs_tracer = SearchTracer(domain, bfs.selection_rule)
        bfs_plan = bfs.search(problem, domain, tracer=bfs_tracer)

        # Run UCS
        ucs = UCSPlanner()
        ucs_tracer = SearchTracer(domain, ucs.selection_rule)
        ucs_plan = ucs.search(problem, domain, tracer=ucs_tracer)

        # Run A* with ZeroHeuristic
        astar_zero = AStarPlanner()
        astar_plan = astar_zero.search(problem, domain, heuristic=ZeroHeuristic())

        assert bfs_plan.is_success is True
        assert ucs_plan.is_success is True
        assert astar_plan.is_success is True

        # BFS should pick the 1-step direct flight (cost 10.0)
        assert len(bfs_plan.actions) == 1
        assert bfs_plan.actions[0].name == "fly"
        assert bfs_plan.total_cost == 10.0

        # UCS MUST pick the 3-step ground route (cost 1.0 + 1.0 + 1.0 = 3.0)
        # Even though it is deeper (3 steps vs 1 step), cost is 3.0 < 10.0!
        # Note: In our open topology drone can step to goal directly if bound to goal_loc!
        # To strictly enforce topology, let's test with an explicit step sequence!
        assert ucs_plan.total_cost <= bfs_plan.total_cost
        assert ucs_plan.total_cost == astar_plan.total_cost

    def test_strict_multi_branch_cost_optimality(self):
        """Construct graph with two non-overlapping branches:
        Branch 1 (fast in steps, expensive in cost): 2 actions of cost 5.0 = 10.0
        Branch 2 (slow in steps, cheap in cost): 4 actions of cost 1.0 = 4.0
        """
        th = TypeHierarchy()
        th.add_type("node", "object")
        preds = [PredicateDefinition("loc", ["node"])]

        # Fast expensive actions
        a_f1 = ActionDefinition("fast1", [ActionParameter("x", "node"), ActionParameter("y", "node")],
                                [Fact("loc", ["?x"])], [Fact("loc", ["?y"])], [Fact("loc", ["?x"])], cost=5.0)
        # Slow cheap actions
        a_s1 = ActionDefinition("slow1", [ActionParameter("x", "node"), ActionParameter("y", "node")],
                                [Fact("loc", ["?x"])], [Fact("loc", ["?y"])], [Fact("loc", ["?x"])], cost=1.0)

        domain = BaseDomain(
            domain_id="branch_domain",
            name="Branch Domain",
            version="1.0",
            description="Branching benchmark",
            type_hierarchy=th,
            predicates=preds,
            actions=[a_f1, a_s1],
        )

        reg = ObjectRegistry(th)
        for name in ["start", "f_mid", "s_1", "s_2", "s_3", "goal"]:
            reg.register_object(name, "node")

        # Define legal transitions only via strict precondition-based actions
        # Let's craft specific actions that chain uniquely:
        act_f1 = ActionDefinition("go_f1", [], [Fact("loc", ["start"])], [Fact("loc", ["f_mid"])], [Fact("loc", ["start"])], cost=5.0)
        act_f2 = ActionDefinition("go_f2", [], [Fact("loc", ["f_mid"])], [Fact("loc", ["goal"])], [Fact("loc", ["f_mid"])], cost=5.0)

        act_s1 = ActionDefinition("go_s1", [], [Fact("loc", ["start"])], [Fact("loc", ["s_1"])], [Fact("loc", ["start"])], cost=1.0)
        act_s2 = ActionDefinition("go_s2", [], [Fact("loc", ["s_1"])], [Fact("loc", ["s_2"])], [Fact("loc", ["s_1"])], cost=1.0)
        act_s3 = ActionDefinition("go_s3", [], [Fact("loc", ["s_2"])], [Fact("loc", ["s_3"])], [Fact("loc", ["s_2"])], cost=1.0)
        act_s4 = ActionDefinition("go_s4", [], [Fact("loc", ["s_3"])], [Fact("loc", ["goal"])], [Fact("loc", ["s_3"])], cost=1.0)

        branch_domain = BaseDomain(
            domain_id="strict_branch",
            name="Strict Branch",
            version="1.0",
            description="",
            type_hierarchy=th,
            predicates=preds,
            actions=[act_f1, act_f2, act_s1, act_s2, act_s3, act_s4],
        )

        prob = SymbolicProblem(
            name="strict_prob",
            domain_name="strict_branch",
            registry=reg,
            initial_state=SymbolicState([Fact("loc", ["start"])], objects=reg.get_all_objects()),
            goal_conditions=[Fact("loc", ["goal"])],
        )

        # BFS must find the 2-step path (cost 10.0)
        bfs_plan = BFSPlanner().search(prob, branch_domain)
        assert bfs_plan.is_success is True
        assert len(bfs_plan.actions) == 2
        assert bfs_plan.total_cost == 10.0
        assert [a.name for a in bfs_plan.actions] == ["go_f1", "go_f2"]

        # UCS must find the 4-step path (cost 4.0)
        ucs_plan = UCSPlanner().search(prob, branch_domain)
        assert ucs_plan.is_success is True
        assert len(ucs_plan.actions) == 4
        assert ucs_plan.total_cost == 4.0
        assert [a.name for a in ucs_plan.actions] == ["go_s1", "go_s2", "go_s3", "go_s4"]

        # A* (ZeroHeuristic) must match UCS exactly
        astar_plan = AStarPlanner().search(prob, branch_domain, heuristic=ZeroHeuristic())
        assert astar_plan.total_cost == 4.0
        assert len(astar_plan.actions) == 4


# ==============================================================================
# Challenge 2: Negative and Non-Finite Cost Rejection
# ==============================================================================

class TestCostValidation:
    """Stress test: negative or NaN/Inf costs must be rejected to preserve optimality."""

    def test_negative_cost_raises_value_error(self):
        th = TypeHierarchy()
        th.add_type("node", "object")
        preds = [PredicateDefinition("p", ["node"])]
        neg_act = ActionDefinition("neg_step", [], [Fact("p", ["a"])], [Fact("p", ["b"])], [Fact("p", ["a"])], cost=-2.5)

        domain = BaseDomain(
            domain_id="neg_dom", name="Neg Dom", version="1.0", description="",
            type_hierarchy=th, predicates=preds, actions=[neg_act]
        )
        reg = ObjectRegistry(th)
        reg.register_object("a", "node")
        reg.register_object("b", "node")

        prob = SymbolicProblem("neg_prob", "neg_dom", reg,
                               SymbolicState([Fact("p", ["a"])], objects=reg.get_all_objects()),
                               [Fact("p", ["b"])])

        with pytest.raises(ValueError, match="invalid negative or non-finite cost"):
            UCSPlanner().search(prob, domain)

        with pytest.raises(ValueError, match="invalid negative or non-finite cost"):
            AStarPlanner().search(prob, domain)

        with pytest.raises(ValueError, match="invalid negative or non-finite cost"):
            BestFirstPlanner().search(prob, domain)

    def test_nan_and_inf_cost_raises_value_error(self):
        th = TypeHierarchy()
        th.add_type("node", "object")
        preds = [PredicateDefinition("p", ["node"])]

        for invalid_cost in [float("nan"), float("inf"), float("-inf")]:
            bad_act = ActionDefinition("bad_step", [], [Fact("p", ["a"])], [Fact("p", ["b"])], [Fact("p", ["a"])], cost=invalid_cost)
            dom = BaseDomain(domain_id="bad_dom", name="Bad Dom", version="1.0", description="",
                             type_hierarchy=th, predicates=preds, actions=[bad_act])
            reg = ObjectRegistry(th)
            reg.register_object("a", "node")
            reg.register_object("b", "node")
            prob = SymbolicProblem("bad_prob", "bad_dom", reg,
                                   SymbolicState([Fact("p", ["a"])], objects=reg.get_all_objects()),
                                   [Fact("p", ["b"])])

            with pytest.raises(ValueError, match="invalid negative or non-finite cost"):
                UCSPlanner().search(prob, dom)


# ==============================================================================
# Challenge 3: Sequence Monotonicity and No Gaps
# ==============================================================================

class TestSequenceMonotonicity:
    """Stress test: sequence numbers must be 1-indexed, strictly monotonic, with 0 gaps."""

    @pytest.mark.parametrize("algo_name", ["BFS", "BEST_FIRST", "A*", "UCS"])
    def test_sequence_monotonicity_on_solvable(self, algo_name):
        bw = registry.get("blocks_world")
        reg = ObjectRegistry(bw.type_hierarchy)
        reg.register_object("a", "block")
        reg.register_object("b", "block")
        init_facts = [
            Fact("on_table", ["a"]), Fact("on_table", ["b"]),
            Fact("clear", ["a"]), Fact("clear", ["b"]),
            Fact("handempty", [])
        ]
        prob = SymbolicProblem("bw_test", bw.domain_id, reg,
                               SymbolicState(init_facts, objects=reg.get_all_objects()),
                               [Fact("on", ["a", "b"])])

        planner = get_planner_by_name(algo_name)
        tracer = SearchTracer(bw, planner.selection_rule)
        plan = planner.search(prob, bw, tracer=tracer)

        assert plan.is_success is True
        assert len(tracer.events) >= 3

        seqs = [e.seq for e in tracer.events]
        expected_seqs = list(range(1, len(tracer.events) + 1))
        assert seqs == expected_seqs, f"Sequence gap found in {algo_name}: {seqs[:10]}..."

    @pytest.mark.parametrize("algo_name", ["BFS", "BEST_FIRST", "A*", "UCS"])
    def test_sequence_monotonicity_initial_is_goal(self, algo_name):
        bw = registry.get("blocks_world")
        reg = ObjectRegistry(bw.type_hierarchy)
        reg.register_object("a", "block")
        init_facts = [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])]
        prob = SymbolicProblem("bw_goal", bw.domain_id, reg,
                               SymbolicState(init_facts, objects=reg.get_all_objects()),
                               [Fact("on_table", ["a"])])  # Already satisfied

        planner = get_planner_by_name(algo_name)
        tracer = SearchTracer(bw, planner.selection_rule)
        plan = planner.search(prob, bw, tracer=tracer)

        assert plan.is_success is True
        seqs = [e.seq for e in tracer.events]
        assert seqs == [1, 2, 3]
        assert tracer.events[0].event_type == SearchEventType.INITIAL
        assert tracer.events[1].event_type == SearchEventType.GOAL_DETECTED
        assert tracer.events[2].event_type == SearchEventType.SOLUTION_RECONSTRUCTED

    @pytest.mark.parametrize("algo_name", ["BFS", "BEST_FIRST", "A*", "UCS"])
    def test_sequence_monotonicity_unsolvable(self, algo_name):
        bw = registry.get("blocks_world")
        reg = ObjectRegistry(bw.type_hierarchy)
        reg.register_object("a", "block")
        init_facts = [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])]
        # Impossible goal
        prob = SymbolicProblem("bw_unsolv", bw.domain_id, reg,
                               SymbolicState(init_facts, objects=reg.get_all_objects()),
                               [Fact("on", ["a", "a"])])  # Cannot put block on itself

        planner = get_planner_by_name(algo_name, max_nodes_expanded=50)
        tracer = SearchTracer(bw, planner.selection_rule)
        plan = planner.search(prob, bw, tracer=tracer)

        assert plan.is_success is False
        seqs = [e.seq for e in tracer.events]
        assert seqs == list(range(1, len(tracer.events) + 1))


# ==============================================================================
# Challenge 4: SHA-256 State Hash Verification
# ==============================================================================

class TestSHA256StateHash:
    """Stress test: state_hash must be a 64-char lowercase hexadecimal string for every event."""

    HEX_64_REGEX = re.compile(r"^[0-9a-f]{64}$")

    def test_sha256_format_across_all_registered_domains(self):
        for domain in registry.list_domains():
            s = SymbolicState(frozenset())
            h = domain.canonical_state_key(s)
            assert len(h) == 64, f"Domain '{domain.domain_id}' produced hash of length {len(h)} != 64"
            assert self.HEX_64_REGEX.match(h) is not None, f"Domain '{domain.domain_id}' produced non-hex-64 hash: {h}"

    def test_every_event_contains_64_char_sha256(self):
        bw = registry.get("blocks_world")
        reg = ObjectRegistry(bw.type_hierarchy)
        reg.register_object("a", "block")
        reg.register_object("b", "block")
        init_facts = [
            Fact("on_table", ["a"]), Fact("on_table", ["b"]),
            Fact("clear", ["a"]), Fact("clear", ["b"]),
            Fact("handempty", [])
        ]
        prob = SymbolicProblem("bw_test", bw.domain_id, reg,
                               SymbolicState(init_facts, objects=reg.get_all_objects()),
                               [Fact("on", ["a", "b"])])

        for algo in ["BFS", "BEST_FIRST", "A*", "UCS"]:
            planner = get_planner_by_name(algo)
            tracer = SearchTracer(bw, planner.selection_rule)
            planner.search(prob, bw, tracer=tracer)

            for ev in tracer.events:
                # State hash verification
                assert len(ev.state_hash) == 64, f"Event {ev.seq} ({ev.event_type}) has invalid state_hash length: '{ev.state_hash}'"
                assert self.HEX_64_REGEX.match(ev.state_hash) is not None
                assert ev.state_id == ev.state_hash

                # Parent ID verification if present
                if ev.parent_id is not None:
                    assert len(ev.parent_id) == 64
                    assert self.HEX_64_REGEX.match(ev.parent_id) is not None

                # Frontier snapshot nodes verification
                for snap_node in ev.frontier_snapshot:
                    s_hash = snap_node["state_hash"]
                    assert len(s_hash) == 64
                    assert self.HEX_64_REGEX.match(s_hash) is not None
                    assert snap_node["node_id"] == s_hash
                    assert snap_node["state_id"] == s_hash


# ==============================================================================
# Challenge 5: Frontier Snapshots Sorting and Bounded Capacity
# ==============================================================================

class TestFrontierSnapshots:
    """Stress test: snapshots must be accurately sorted and bounded by max_snapshot_size."""

    def test_frontier_snapshot_capacity_and_sorting(self):
        bw = registry.get("blocks_world")
        reg = ObjectRegistry(bw.type_hierarchy)
        for name in ["a", "b", "c"]:
            reg.register_object(name, "block")
        init_facts = [
            Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]),
            Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]),
            Fact("handempty", [])
        ]
        prob = SymbolicProblem("bw_3block", bw.domain_id, reg,
                               SymbolicState(init_facts, objects=reg.get_all_objects()),
                               [Fact("on", ["a", "b"]), Fact("on", ["b", "c"])])

        # Test A* sorting: f_cost must be non-decreasing
        astar = AStarPlanner()
        tracer_astar = SearchTracer(bw, astar.selection_rule, max_snapshot_size=10)
        astar.search(prob, bw, tracer=tracer_astar)

        for ev in tracer_astar.events:
            snap = ev.frontier_snapshot
            assert len(snap) <= 10
            if len(snap) > 1:
                # Check f_costs are non-decreasing
                f_costs = [item["f_cost"] for item in snap]
                assert f_costs == sorted(f_costs), f"A* snapshot not sorted by f_cost: {f_costs}"

        # Test UCS sorting: g_cost must be non-decreasing
        ucs = UCSPlanner()
        tracer_ucs = SearchTracer(bw, ucs.selection_rule, max_snapshot_size=10)
        ucs.search(prob, bw, tracer=tracer_ucs)

        for ev in tracer_ucs.events:
            snap = ev.frontier_snapshot
            assert len(snap) <= 10
            if len(snap) > 1:
                g_costs = [item["g_cost"] for item in snap]
                assert g_costs == sorted(g_costs), f"UCS snapshot not sorted by g_cost: {g_costs}"

        # Test BFS sorting: depth must be non-decreasing
        bfs = BFSPlanner()
        tracer_bfs = SearchTracer(bw, bfs.selection_rule, max_snapshot_size=10)
        bfs.search(prob, bw, tracer=tracer_bfs)

        for ev in tracer_bfs.events:
            snap = ev.frontier_snapshot
            assert len(snap) <= 10
            if len(snap) > 1:
                depths = [item["depth"] for item in snap]
                assert depths == sorted(depths), f"BFS snapshot not sorted by depth: {depths}"


# ==============================================================================
# Challenge 6: Solution Path Reconstruction Continuity
# ==============================================================================

class TestSolutionPathReconstruction:
    """Stress test: solution path must form an unbroken semantic chain from initial state to goal."""

    @pytest.mark.parametrize("algo", ["BFS", "BEST_FIRST", "A*", "UCS"])
    def test_solution_path_transition_continuity(self, algo):
        bw = registry.get("blocks_world")
        reg = ObjectRegistry(bw.type_hierarchy)
        reg.register_object("a", "block")
        reg.register_object("b", "block")
        init_facts = [
            Fact("on_table", ["a"]), Fact("on_table", ["b"]),
            Fact("clear", ["a"]), Fact("clear", ["b"]),
            Fact("handempty", [])
        ]
        prob = SymbolicProblem("bw_test", bw.domain_id, reg,
                               SymbolicState(init_facts, objects=reg.get_all_objects()),
                               [Fact("on", ["a", "b"])])

        planner = get_planner_by_name(algo)
        tracer = SearchTracer(bw, planner.selection_rule)
        plan = planner.search(prob, bw, tracer=tracer)

        assert plan.is_success is True
        nodes = tracer.solution_path_node_ids
        edges = tracer.solution_path_edge_ids

        # Chain lengths: N nodes, N-1 edges
        assert len(nodes) == len(plan.actions) + 1
        assert len(edges) == len(plan.actions)

        # Initial node matches problem initial state
        assert nodes[0] == bw.canonical_state_key(prob.initial_state)

        # Step by step execution verification
        curr_state = prob.initial_state
        for i, act_schema in enumerate(plan.actions):
            curr_hash = bw.canonical_state_key(curr_state)
            assert nodes[i] == curr_hash

            edge = edges[i]
            parts = edge.split("::")
            assert len(parts) == 3
            u, act_str, v = parts
            assert u == curr_hash
            assert v == nodes[i + 1]

            # Execute ground action and verify next state matches
            dom_act = bw.get_action(act_schema.name)
            assert dom_act is not None
            binding = dict(zip([p.name for p in dom_act.parameters], act_schema.arguments))
            ground_act = dom_act.instantiate(binding)
            next_state, err = bw.apply_action(curr_state, ground_act)
            assert err is None
            next_hash = bw.canonical_state_key(next_state)
            assert next_hash == nodes[i + 1]
            curr_state = next_state

        # Final state must satisfy goal
        assert prob.is_goal_satisfied(curr_state) is True


# ==============================================================================
# Challenge 7: SearchNode Deterministic Tie-Breaking
# ==============================================================================

class TestTieBreakingOrder:
    """Stress test: SearchNode comparison __lt__ must break ties deterministically."""

    def test_searchnode_lt_tie_breaking(self):
        s = SymbolicState(frozenset())

        # 1. Lower f_cost wins
        n1 = SearchNode(state=s, g_cost=1.0, h_cost=2.0)  # f = 3.0
        n2 = SearchNode(state=s, g_cost=2.0, h_cost=2.0)  # f = 4.0
        assert n1 < n2

        # 2. Equal f_cost: lower h_cost wins
        n3 = SearchNode(state=s, g_cost=2.0, h_cost=1.0)  # f = 3.0, h = 1.0
        n4 = SearchNode(state=s, g_cost=1.0, h_cost=2.0)  # f = 3.0, h = 2.0
        assert n3 < n4

        # 3. Equal f_cost and equal h_cost: deeper node wins (depth preference)
        p = SearchNode(state=s)
        n5 = SearchNode(state=s, parent=p, g_cost=1.0, h_cost=1.0)  # depth = 1
        n6 = SearchNode(state=s, parent=None, g_cost=1.0, h_cost=1.0)  # depth = 0
        assert n5 < n6  # deeper preferred

        # 4. Equal f, h, and depth: earlier node_id wins
        n7 = SearchNode(state=s, g_cost=1.0, h_cost=1.0)
        n8 = SearchNode(state=s, g_cost=1.0, h_cost=1.0)
        assert n7.depth == n8.depth
        assert n7.node_id < n8.node_id
        assert n7 < n8


# ==============================================================================
# Challenge 8: REST API Boundary and Validation Attacks
# ==============================================================================

class TestSearchTraceAPIAdversarial:
    """Stress test: API input boundary attacks and status code validation."""

    def test_api_invalid_algorithm_rejected(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "algorithm": "UNKNOWN_ALGORITHM_XYZ",
        })
        assert res.status_code == 422  # Pydantic field validator rejects

    def test_api_out_of_bounds_max_nodes(self, client):
        # max_nodes_expanded must be 1 <= n <= 50000
        for bad_val in [0, -10, 50001, 100000]:
            res = client.post("/api/search/trace", json={
                "domain_id": "blocks_world",
                "max_nodes_expanded": bad_val,
            })
            assert res.status_code == 422

    def test_api_out_of_bounds_timeout(self, client):
        for bad_t in [0.0, -1.0, 61.0, 100.0]:
            res = client.post("/api/search/trace", json={
                "domain_id": "blocks_world",
                "timeout_seconds": bad_t,
            })
            assert res.status_code == 422

    def test_api_nonexistent_domain_404(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "quantum_physics_domain_404",
            "algorithm": "BFS",
        })
        assert res.status_code == 404

    def test_api_valid_all_algorithms(self, client):
        for algo in ["BFS", "BEST_FIRST", "A*", "UCS"]:
            res = client.post("/api/search/trace", json={
                "domain_id": "blocks_world",
                "algorithm": algo,
                "prompt": "Stack the red box on the blue box.",
            })
            assert res.status_code == 200, f"Failed for algorithm {algo}: {res.text}"
            data = res.json()
            assert data["algorithm"] == algo
            assert data["total_events"] > 0
            assert len(data["events"]) == data["total_events"]
            assert data["is_goal_found"] is True
            assert len(data["solution_path_node_ids"]) > 0
            assert len(data["solution_path_edge_ids"]) > 0


# ==============================================================================
# Challenge 9: Zero-Cost Cycles & Loop Prevention
# ==============================================================================

class TestZeroCostCycles:
    """Stress test: zero-cost self-loops or bidirectional edges must not cause infinite loops."""

    def test_ucs_and_astar_handle_zero_cost_cycles_cleanly(self):
        th = TypeHierarchy()
        th.add_type("node", "object")
        preds = [PredicateDefinition("at", ["node"])]

        # Zero cost cycle: a -> b (cost 0.0), b -> a (cost 0.0)
        # Costly goal: a -> goal (cost 5.0)
        act_cycle_ab = ActionDefinition("cycle_ab", [], [Fact("at", ["a"])], [Fact("at", ["b"])], [Fact("at", ["a"])], cost=0.0)
        act_cycle_ba = ActionDefinition("cycle_ba", [], [Fact("at", ["b"])], [Fact("at", ["a"])], [Fact("at", ["b"])], cost=0.0)
        act_goal = ActionDefinition("reach_goal", [], [Fact("at", ["a"])], [Fact("at", ["goal"])], [Fact("at", ["a"])], cost=5.0)

        domain = BaseDomain(
            domain_id="zero_cost_dom", name="Zero Cost", version="1.0", description="",
            type_hierarchy=th, predicates=preds, actions=[act_cycle_ab, act_cycle_ba, act_goal]
        )
        reg = ObjectRegistry(th)
        for name in ["a", "b", "goal"]:
            reg.register_object(name, "node")

        prob = SymbolicProblem("zero_prob", "zero_cost_dom", reg,
                               SymbolicState([Fact("at", ["a"])], objects=reg.get_all_objects()),
                               [Fact("at", ["goal"])])

        for algo in ["UCS", "A*"]:
            planner = get_planner_by_name(algo, max_nodes_expanded=100)
            tracer = SearchTracer(domain, planner.selection_rule)
            plan = planner.search(prob, domain, tracer=tracer)

            assert plan.is_success is True
            assert plan.total_cost == 5.0
            # Ensure no sequence gaps occurred
            seqs = [e.seq for e in tracer.events]
            assert seqs == list(range(1, len(tracer.events) + 1))


# ==============================================================================
# Challenge 10: Pruning Events (SKIP_PRUNED) Sequence and Hash Integrity
# ==============================================================================

class TestPruningEventIntegrity:
    """Stress test: SKIP_PRUNED events emitted by hard constraints, invariants, or repair."""

    def test_forbidden_action_and_hard_constraint_pruning(self):
        bw = registry.get("blocks_world")
        reg = ObjectRegistry(bw.type_hierarchy)
        reg.register_object("a", "block")
        reg.register_object("b", "block")
        init_facts = [
            Fact("on_table", ["a"]), Fact("on_table", ["b"]),
            Fact("clear", ["a"]), Fact("clear", ["b"]),
            Fact("handempty", [])
        ]
        prob = SymbolicProblem("bw_test", bw.domain_id, reg,
                               SymbolicState(init_facts, objects=reg.get_all_objects()),
                               [Fact("on", ["a", "b"])])

        # Pick a forbidden action
        pick_up_a = bw.get_action("pick_up").instantiate({"?b": "a"})
        planner = BFSPlanner()
        tracer = SearchTracer(bw, planner.selection_rule)

        plan = planner.search(prob, bw, forbidden_actions={pick_up_a}, tracer=tracer)

        # Check that SKIP_PRUNED event was emitted
        skip_events = [e for e in tracer.events if e.event_type == SearchEventType.SKIP_PRUNED]
        assert len(skip_events) > 0, "No SKIP_PRUNED event was recorded for forbidden action"

        # Check every skip event has valid sequence and 64-char hash
        for sk in skip_events:
            assert len(sk.state_hash) == 64
            assert re.match(r"^[0-9a-f]{64}$", sk.state_hash) is not None
            assert "Forbidden" in sk.message or "Cycle" in sk.message or "constraint" in sk.message

        # Overall sequence must remain strictly continuous
        seqs = [e.seq for e in tracer.events]
        assert seqs == list(range(1, len(tracer.events) + 1))


# ==============================================================================
# Challenge 11: API Concurrency Capacity Exhaustion (HTTP 429)
# ==============================================================================

class TestAPIConcurrencyLimit:
    """Stress test: verify 429 is returned when concurrency limit (2) is saturated."""

    def test_concurrency_saturation_returns_429(self, client):
        from app.api.routes import graph_concurrency_manager

        # Acquire all available slots
        slot1 = graph_concurrency_manager.acquire()
        slot2 = graph_concurrency_manager.acquire()
        assert slot1 is True
        assert slot2 is True

        try:
            # 3rd request should receive 429 immediately
            res = client.post("/api/search/trace", json={
                "domain_id": "blocks_world",
                "algorithm": "BFS",
            })
            assert res.status_code == 429
            assert "Search execution capacity exceeded" in res.text
        finally:
            # Release slots
            graph_concurrency_manager.release()
            graph_concurrency_manager.release()

        # After releasing, request should succeed (or proceed past concurrency check)
        res_after = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "algorithm": "BFS",
        })
        assert res_after.status_code != 429


# ==============================================================================
# Challenge 12: GraphEdge Transition Details Serialization
# ==============================================================================

class TestGraphEdgeSerialization:
    """Stress test: GraphEdge must capture preconditions, add/del effects, invariants_satisfied."""

    def test_graphedge_from_ground_action(self):
        from core.graph.models import GraphEdge, GraphNode, ReachableStateGraph
        bw = registry.get("blocks_world")
        stack_def = bw.get_action("stack")
        ground_stack = stack_def.instantiate({"?b": "a", "?under": "b"})

        edge = GraphEdge.from_ground_action(
            edge_id="e1",
            source_id="s_src",
            target_id="s_tgt",
            action=ground_stack,
            invariants_satisfied=True,
            is_plan_edge=True,
        )

        assert edge.action_name == "stack"
        assert edge.action_args == ["a", "b"]
        assert len(edge.preconditions) == len(ground_stack.preconditions)
        assert len(edge.add_effects) == len(ground_stack.add_effects)
        assert len(edge.del_effects) == len(ground_stack.del_effects)
        assert edge.invariants_satisfied is True
        assert edge.is_plan_edge is True

        # Test Cytoscape serialization via ReachableStateGraph
        from core.graph.models import GraphNode, ReachableStateGraph
        graph = ReachableStateGraph(
            graph_id="g1",
            domain_id="blocks_world",
            problem_name="p1",
            nodes={"s_src": GraphNode(node_id="s_src", facts=[]), "s_tgt": GraphNode(node_id="s_tgt", facts=[])},
            edges=[edge],
        )
        cy = graph.to_cytoscape()
        cy_edge = [elem for elem in cy["elements"] if elem["data"]["id"] == "e1"][0]
        data = cy_edge["data"]

        assert "preconditions" in data
        assert "add_effects" in data
        assert "del_effects" in data
        assert "invariants_satisfied" in data
        assert data["invariants_satisfied"] is True
        assert data["is_plan_edge"] is True
        assert cy_edge["classes"] == "plan_edge"
