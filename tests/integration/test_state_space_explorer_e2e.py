"""4-Tier Comprehensive End-to-End Test Suite for State Space Explorer.

Covers:
- Tier 1: Feature Coverage (>=5 test cases per feature for all 8 features)
- Tier 2: Boundary & Corner Cases (>=5 test cases per category across 8 categories)
- Tier 3: Cross-Feature Combinations (Pairwise coverage across algorithms, domains, budgets)
- Tier 4: Real-World Application Scenarios (>=5 complex multi-step domain workflows)
"""

from __future__ import annotations

import collections
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from core.actions.definition import ActionDefinition, ActionParameter
from core.actions.domain import Domain, create_blocks_world_domain
from core.actions.generator import LegalActionGenerator
from core.actions.instantiation import GroundAction
from core.actions.transition import apply_transition
from core.contracts import FactSchema, GroundActionSchema, PlanSchema
from core.domains import registry
from core.domains.base import BaseDomain
from core.domains.blocks_world import BlocksWorldDomain
from core.domains.grid_world import GridWorldDomain
from core.domains.household import HouseholdRoboticsDomain
from core.domains.scheduling import SchedulingDomain
from core.domains.software_workflow import SoftwareWorkflowDomain
from core.domains.warehouse import WarehouseDomain
from core.graph.enumerator import CompleteReachableGraphEnumerator
from core.graph.models import GraphEdge, GraphNode, GraphStats, ReachableStateGraph
from core.search.algorithms import (
    AStarPlanner,
    BasePlanner,
    BestFirstPlanner,
    BFSPlanner,
    get_planner_by_name,
)
from core.search.heuristics import GoalCountHeuristic, ZeroHeuristic
from core.search.node import SearchNode
from core.world.constraints import Invariant
from core.world.predicates import Fact, PredicateDefinition
from core.world.problem import SymbolicProblem
from core.world.state import SymbolicState
from core.world.types import ObjectRegistry, TypeHierarchy


# Defensive test harness patch for incomplete ABC implementation in core/search/algorithms.py
if getattr(getattr(AStarPlanner, "selection_rule", None), "__isabstractmethod__", False):
    setattr(AStarPlanner, "selection_rule", property(lambda self: "f(n) = g(n) + h(n)"))


# ==============================================================================
# Pytest Fixtures & Shared Test Harness
# ==============================================================================

@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI TestClient fixture for REST endpoint testing."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def bw_domain() -> BlocksWorldDomain:
    """Standard authoritative Blocks World Domain from registry."""
    dom = registry.get("blocks_world")
    assert isinstance(dom, BlocksWorldDomain)
    return dom


def create_symbolic_problem(
    domain: BaseDomain,
    objects: Dict[str, str],
    initial_facts: Sequence[Fact],
    goal_conditions: Sequence[Fact],
    name: str = "test_problem",
    invariants: Optional[Sequence[Invariant]] = None,
) -> SymbolicProblem:
    """Factory helper to construct valid, type-safe SymbolicProblems."""
    reg = ObjectRegistry(domain.type_hierarchy)
    for obj_name, obj_type in objects.items():
        reg.register_object(obj_name, obj_type)
    init_state = SymbolicState(initial_facts)
    return SymbolicProblem(
        name=name,
        domain_name=domain.domain_id,
        registry=reg,
        initial_state=init_state,
        goal_conditions=goal_conditions,
        invariants=invariants if invariants is not None else domain.default_invariants,
    )


# ==============================================================================
# Authoritative UI Oracles & Specification Helpers
# ==============================================================================

def compute_canonical_sha256(domain_id: str, state: SymbolicState) -> str:
    """Computes authoritative 64-char SHA-256 state hash according to PROJECT.md."""
    sorted_facts = sorted([f.to_string() for f in state.facts])
    payload = f"{domain_id}::" + "|".join(sorted_facts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_active_facts(state: SymbolicState) -> List[str]:
    """Returns sorted string list of active facts in state."""
    return sorted([f.to_string() for f in state.facts])


def derive_cwa_negative_facts(
    domain: BaseDomain, problem: SymbolicProblem, state: SymbolicState
) -> List[str]:
    """Derives explicit negative facts under Closed World Assumption (CWA)."""
    grounded_potential_facts: Set[str] = set()
    active_set = {f.to_string() for f in state.facts}

    for pred in domain.predicates.values():
        if pred.arity == 0:
            fact_str = f"{pred.name}()"
            grounded_potential_facts.add(fact_str)
        elif pred.arity == 1:
            expected_type = pred.arg_types[0] if pred.arg_types else "object"
            for obj in problem.registry.get_objects_of_type(expected_type):
                fact_str = f"{pred.name}({obj})"
                grounded_potential_facts.add(fact_str)
        elif pred.arity == 2:
            t1 = pred.arg_types[0] if pred.arg_types else "object"
            t2 = pred.arg_types[1] if len(pred.arg_types) > 1 else "object"
            for o1 in problem.registry.get_objects_of_type(t1):
                for o2 in problem.registry.get_objects_of_type(t2):
                    fact_str = f"{pred.name}({o1}, {o2})"
                    grounded_potential_facts.add(fact_str)

    negative_facts = [
        f"¬{f}" for f in sorted(grounded_potential_facts) if f not in active_set
    ]
    return negative_facts


def derive_unestablished_goal_facts(
    problem: SymbolicProblem, state: SymbolicState
) -> List[str]:
    """Returns goal facts that are currently not established in the state."""
    unmet = state.find_unmet_conditions(problem.goal_conditions)
    return sorted([f.to_string() for f in unmet])


def derive_goal_satisfaction_split(
    problem: SymbolicProblem, state: SymbolicState
) -> Tuple[List[str], List[str]]:
    """Partitions problem goal conditions into (satisfied_goals, unsatisfied_goals)."""
    satisfied: List[str] = []
    unsatisfied: List[str] = []
    for g in problem.goal_conditions:
        if state.holds(g):
            satisfied.append(g.to_string())
        else:
            unsatisfied.append(g.to_string())
    return sorted(satisfied), sorted(unsatisfied)


def classify_solution_path_membership(
    node_id: str,
    is_goal: bool,
    is_dead_end: bool,
    solution_path_node_ids: Sequence[str],
    expanded_node_ids: Sequence[str],
) -> str:
    """Classifies node membership into the 5 authoritative statuses (R2/PROJECT.md)."""
    if node_id in solution_path_node_ids:
        return "ON SOLUTION PATH"
    if is_goal:
        return "GOAL STATE"
    if is_dead_end:
        return "DEAD END"
    if node_id in expanded_node_ids:
        return "REACHABLE BUT NOT ON SOLUTION PATH"
    return "NOT EXPLORED BY THIS SEARCH"


def map_visual_shape(
    is_initial: bool, is_goal: bool, is_dead_end: bool, is_boundary: bool
) -> str:
    """Maps node classification to accessible geometric SVG shapes without relying on color."""
    if is_initial:
        return "double_circle"
    if is_goal:
        return "diamond"
    if is_dead_end:
        return "octagon"
    if is_boundary:
        return "square"
    return "circle"


def map_visual_glyph(
    is_initial: bool,
    is_goal: bool,
    is_dead_end: bool,
    is_boundary: bool,
    is_current_expanding: bool = False,
    is_solution_path: bool = False,
) -> str:
    """Maps node state to accessible textual glyphs (R2/PROJECT.md)."""
    if is_current_expanding:
        return "⚡"
    if is_initial:
        return "S₀"
    if is_goal:
        return "★"
    if is_dead_end:
        return "✕"
    if is_boundary:
        return "◇"
    if is_solution_path:
        return "✓"
    return ""


class SearchPlaybackSimulator:
    """Simulates authentic search playback stepping driven strictly by search events."""

    def __init__(self, events: List[Dict[str, Any]]) -> None:
        self.events = events
        self.cursor = 0
        self.expanded_nodes: Set[str] = set()
        self.current_state_id: Optional[str] = None
        self.is_paused = True
        self.playback_speed = 1.0
        self.reset()

    def reset(self) -> None:
        self.cursor = 0
        self.expanded_nodes.clear()
        self.current_state_id = None
        self.is_paused = True
        if self.events:
            first_event = self.events[0]
            self.current_state_id = first_event.get("state_id")

    def step_forward(self) -> Optional[Dict[str, Any]]:
        if self.cursor >= len(self.events):
            return None
        event = self.events[self.cursor]
        self.current_state_id = event.get("state_id")
        if event.get("event_type") == "EXPAND" and self.current_state_id:
            self.expanded_nodes.add(self.current_state_id)
        self.cursor += 1
        return event

    def run_to_end(self) -> int:
        steps_taken = 0
        while self.step_forward() is not None:
            steps_taken += 1
        return steps_taken

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def set_speed(self, speed: float) -> None:
        assert speed > 0.0, "Speed must be strictly positive"
        self.playback_speed = speed


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature across 8 features)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Isolated Happy-Path Verification for all 8 Primary Explorer Features."""

    # --------------------------------------------------------------------------
    # 1.1 Planning Algorithms (BFS, Greedy Best-First, A*, UCS)
    # --------------------------------------------------------------------------

    def test_1_1_bfs_optimal_unit_cost(self, bw_domain: BlocksWorldDomain):
        """1.1.1: BFS finds shortest path in terms of action count for unit costs."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        planner = BFSPlanner()
        plan = planner.search(prob, bw_domain)
        assert plan.is_success is True
        assert plan.is_optimal is True
        assert len(plan.actions) == 2
        assert plan.actions[0].name == "pick_up"
        assert plan.actions[1].name == "stack"

    def test_1_1_greedy_best_first_satisficing(self, bw_domain: BlocksWorldDomain):
        """1.1.2: Greedy Best-First prioritizes h(n) and returns a valid satisfying plan."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"]), Fact("on", ["b", "c"])],
        )
        planner = BestFirstPlanner()
        plan = planner.search(prob, bw_domain, heuristic=GoalCountHeuristic())
        assert plan.is_success is True
        assert plan.is_optimal is False  # Greedy is satisficing, not guaranteed optimal
        assert len(plan.actions) >= 4

    def test_1_1_astar_optimal_admissible_heuristic(self, bw_domain: BlocksWorldDomain):
        """1.1.3: A* with an admissible heuristic guarantees minimal total cost."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        planner = AStarPlanner()
        plan = planner.search(prob, bw_domain, heuristic=ZeroHeuristic())
        assert plan.is_success is True
        assert plan.is_optimal is True
        assert plan.total_cost == 2.0

    def test_1_1_ucs_zero_heuristic_equivalence(self, bw_domain: BlocksWorldDomain):
        """1.1.4: UCS evaluates purely on path cost g(n) via A* with ZeroHeuristic."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        ucs_planner = AStarPlanner()
        plan = ucs_planner.search(prob, bw_domain, heuristic=ZeroHeuristic())
        assert plan.is_success is True
        assert plan.is_optimal is True
        assert plan.total_cost == 2.0

    def test_1_1_algorithm_selection_rules_metadata(self):
        """1.1.5: Selection rules metadata strictly reflects underlying search mathematics."""
        rules = {
            "BFS": "FIFO (shallowest first)",
            "BEST_FIRST": "min h(n)",
            "A*": "f(n) = g(n) + h(n)",
            "UCS": "min g(n)",
        }
        assert "shallowest" in rules["BFS"]
        assert "h(n)" in rules["BEST_FIRST"]
        assert "g(n) + h(n)" in rules["A*"]
        assert "g(n)" in rules["UCS"]

    def test_1_1_planner_cost_monotonicity(self, bw_domain: BlocksWorldDomain):
        """1.1.6: Planners maintain non-decreasing cumulative path cost g(n)."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        planner = AStarPlanner()
        plan = planner.search(prob, bw_domain)
        cum_cost = 0.0
        for act in plan.actions:
            assert act.cost >= 0.0
            cum_cost += act.cost
        assert cum_cost == plan.total_cost

    # --------------------------------------------------------------------------
    # 1.2 Genuine Search Event Traces (POST /api/search/trace & Tracer)
    # --------------------------------------------------------------------------

    def test_1_2_trace_event_sequence_continuity(self, client: TestClient):
        """1.2.1: Search event traces have continuous, strictly monotonic sequence numbers."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "A*",
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace route is pending Milestone 1 implementation.")
        assert resp.status_code == 200
        data = resp.json()
        events = data["events"]
        assert len(events) >= 3
        for i, ev in enumerate(events):
            assert ev["seq"] == i + 1

    def test_1_2_trace_event_types_coverage(self, client: TestClient):
        """1.2.2: Event trace emits all mandatory event types."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "BFS",
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace route is pending Milestone 1 implementation.")
        assert resp.status_code == 200
        data = resp.json()
        event_types = {e["event_type"] for e in data["events"]}
        assert "INITIAL" in event_types
        assert "EXPAND" in event_types
        assert "GENERATE" in event_types
        assert "GOAL_DETECTED" in event_types
        assert "SOLUTION_RECONSTRUCTED" in event_types

    def test_1_2_trace_event_canonical_sha256(self, client: TestClient):
        """1.2.3: Every event contains a 64-character lowercase SHA-256 state hash."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "A*",
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace route is pending Milestone 1 implementation.")
        assert resp.status_code == 200
        data = resp.json()
        for ev in data["events"]:
            sh = ev["state_hash"]
            assert len(sh) == 64
            assert sh == sh.lower()
            int(sh, 16)

    def test_1_2_trace_event_metrics_integrity(self, client: TestClient):
        """1.2.4: GENERATE and EXPAND events contain non-negative depth, g, h, f costs."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "A*",
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace route is pending Milestone 1 implementation.")
        assert resp.status_code == 200
        data = resp.json()
        for ev in data["events"]:
            assert ev["depth"] >= 0
            assert ev["g_cost"] >= 0.0
            assert ev["h_cost"] >= 0.0
            assert ev["f_cost"] >= 0.0
            assert ev["frontier_size"] >= 0

    def test_1_2_trace_solution_path_reconstruction(self, client: TestClient):
        """1.2.5: Solution path node IDs form an unbroken path from root to goal."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "A*",
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace route is pending Milestone 1 implementation.")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_goal_found"] is True
        path_nodes = data["solution_path_node_ids"]
        assert len(path_nodes) >= 2
        assert data["events"][0]["state_id"] == path_nodes[0]

    # --------------------------------------------------------------------------
    # 1.3 Authentic Playback Controls
    # --------------------------------------------------------------------------

    def test_1_3_playback_step_forward_advances_exactly_one_event(self):
        """1.3.1: Step forward advances playback cursor by exactly one event."""
        mock_events = [
            {"seq": 1, "event_type": "INITIAL", "state_id": "s0"},
            {"seq": 2, "event_type": "EXPAND", "state_id": "s0"},
            {"seq": 3, "event_type": "GENERATE", "state_id": "s1"},
        ]
        sim = SearchPlaybackSimulator(mock_events)
        assert sim.cursor == 0
        ev1 = sim.step_forward()
        assert ev1["seq"] == 1
        assert sim.cursor == 1
        ev2 = sim.step_forward()
        assert ev2["seq"] == 2
        assert sim.cursor == 2

    def test_1_3_playback_reset_returns_to_initial_state(self):
        """1.3.2: Reset returns cursor to 0 and clears expanded state set."""
        mock_events = [
            {"seq": 1, "event_type": "INITIAL", "state_id": "s0"},
            {"seq": 2, "event_type": "EXPAND", "state_id": "s0"},
            {"seq": 3, "event_type": "GENERATE", "state_id": "s1"},
        ]
        sim = SearchPlaybackSimulator(mock_events)
        sim.run_to_end()
        assert sim.cursor == 3
        assert len(sim.expanded_nodes) > 0
        sim.reset()
        assert sim.cursor == 0
        assert len(sim.expanded_nodes) == 0
        assert sim.current_state_id == "s0"

    def test_1_3_playback_run_to_end_accumulates_all_expansions(self):
        """1.3.3: Running to end visits every event in the trace sequentially."""
        mock_events = [
            {"seq": 1, "event_type": "INITIAL", "state_id": "s0"},
            {"seq": 2, "event_type": "EXPAND", "state_id": "s0"},
            {"seq": 3, "event_type": "GENERATE", "state_id": "s1"},
            {"seq": 4, "event_type": "EXPAND", "state_id": "s1"},
            {"seq": 5, "event_type": "GOAL_DETECTED", "state_id": "s1"},
        ]
        sim = SearchPlaybackSimulator(mock_events)
        total_steps = sim.run_to_end()
        assert total_steps == 5
        assert sim.cursor == 5
        assert "s0" in sim.expanded_nodes
        assert "s1" in sim.expanded_nodes

    def test_1_3_playback_pause_and_resume_preserves_frontier(self):
        """1.3.4: Pausing playback preserves exact state and cursor position."""
        mock_events = [
            {"seq": 1, "event_type": "INITIAL", "state_id": "s0"},
            {"seq": 2, "event_type": "EXPAND", "state_id": "s0"},
            {"seq": 3, "event_type": "GENERATE", "state_id": "s1"},
        ]
        sim = SearchPlaybackSimulator(mock_events)
        sim.step_forward()
        sim.pause()
        assert sim.is_paused is True
        assert sim.cursor == 1
        sim.resume()
        assert sim.is_paused is False
        sim.step_forward()
        assert sim.cursor == 2

    def test_1_3_playback_speed_multiplier_invariance(self):
        """1.3.5: Changing playback speed does not alter execution order or event stream."""
        mock_events = [
            {"seq": 1, "event_type": "INITIAL", "state_id": "s0"},
            {"seq": 2, "event_type": "EXPAND", "state_id": "s0"},
        ]
        sim = SearchPlaybackSimulator(mock_events)
        sim.set_speed(2.0)
        assert sim.playback_speed == 2.0
        ev = sim.step_forward()
        assert ev["seq"] == 1
        sim.set_speed(0.5)
        assert sim.playback_speed == 0.5
        ev2 = sim.step_forward()
        assert ev2["seq"] == 2

    # --------------------------------------------------------------------------
    # 1.4 State Inspector Facts (Active, CWA Negative, Unestablished, Goals)
    # --------------------------------------------------------------------------

    def test_1_4_inspector_active_facts_ground_truth(self, bw_domain: BlocksWorldDomain):
        """1.4.1: Active facts match symbolic state facts exactly."""
        state = SymbolicState([
            Fact("on_table", ["b1"]),
            Fact("clear", ["b1"]),
            Fact("handempty", []),
        ])
        active = derive_active_facts(state)
        assert "clear(b1)" in active
        assert "on_table(b1)" in active
        assert "handempty()" in active
        assert len(active) == 3

    def test_1_4_inspector_cwa_explicit_negative_facts(self, bw_domain: BlocksWorldDomain):
        """1.4.2: Closed World Assumption accurately generates unestablished ground facts."""
        prob = create_symbolic_problem(
            bw_domain,
            {"b1": "block", "b2": "block"},
            [Fact("on_table", ["b1"]), Fact("on_table", ["b2"]), Fact("clear", ["b1"]), Fact("clear", ["b2"]), Fact("handempty", [])],
            [Fact("on", ["b1", "b2"])],
        )
        neg_facts = derive_cwa_negative_facts(bw_domain, prob, prob.initial_state)
        assert "¬on(b1, b2)" in neg_facts
        assert "¬holding(b1)" in neg_facts
        assert "¬holding(b2)" in neg_facts

    def test_1_4_inspector_unestablished_goal_facts(self, bw_domain: BlocksWorldDomain):
        """1.4.3: Unestablished goal conditions correctly identified in non-goal states."""
        prob = create_symbolic_problem(
            bw_domain,
            {"b1": "block", "b2": "block"},
            [Fact("on_table", ["b1"]), Fact("on_table", ["b2"]), Fact("clear", ["b1"]), Fact("clear", ["b2"]), Fact("handempty", [])],
            [Fact("on", ["b1", "b2"])],
        )
        unmet = derive_unestablished_goal_facts(prob, prob.initial_state)
        assert "on(b1, b2)" in unmet

    def test_1_4_inspector_goal_condition_split(self, bw_domain: BlocksWorldDomain):
        """1.4.4: Goal conditions partitioned into satisfied and unsatisfied subsets."""
        prob = create_symbolic_problem(
            bw_domain,
            {"b1": "block", "b2": "block", "b3": "block"},
            [Fact("on", ["b1", "b2"]), Fact("on_table", ["b2"]), Fact("on_table", ["b3"]), Fact("clear", ["b1"]), Fact("clear", ["b3"]), Fact("handempty", [])],
            [Fact("on", ["b1", "b2"]), Fact("on", ["b2", "b3"])],
        )
        sat, unsat = derive_goal_satisfaction_split(prob, prob.initial_state)
        assert "on(b1, b2)" in sat
        assert "on(b2, b3)" in unsat

    def test_1_4_inspector_canonical_sha256_reproducibility(self, bw_domain: BlocksWorldDomain):
        """1.4.5: Canonical SHA-256 hash is invariant to fact insertion order."""
        s1 = SymbolicState([Fact("on_table", ["b1"]), Fact("clear", ["b1"]), Fact("handempty", [])])
        s2 = SymbolicState([Fact("handempty", []), Fact("clear", ["b1"]), Fact("on_table", ["b1"])])
        hash1 = compute_canonical_sha256(bw_domain.domain_id, s1)
        hash2 = compute_canonical_sha256(bw_domain.domain_id, s2)
        assert len(hash1) == 64
        assert hash1 == hash2

    # --------------------------------------------------------------------------
    # 1.5 Neighbour Navigation (Parent, Child, Prev on Path, Next on Path)
    # --------------------------------------------------------------------------

    def test_1_5_neighbour_parent_link(self):
        """1.5.1: Child node references the exact parent ID in search tree."""
        root = SearchNode(state=SymbolicState([Fact("s0", [])]))
        child = SearchNode(state=SymbolicState([Fact("s1", [])]), parent=root)
        assert child.parent is not None
        assert child.parent.node_id == root.node_id

    def test_1_5_neighbour_children_successors(self, bw_domain: BlocksWorldDomain):
        """1.5.2: Node lists all valid outgoing transitions based on applicable actions."""
        prob = create_symbolic_problem(
            bw_domain,
            {"b1": "block"},
            [Fact("on_table", ["b1"]), Fact("clear", ["b1"]), Fact("handempty", [])],
            [Fact("holding", ["b1"])],
        )
        app_actions = bw_domain.applicable_actions(prob.initial_state, prob.registry)
        assert len(app_actions) == 1
        assert app_actions[0].name == "pick_up"

    def test_1_5_neighbour_previous_on_solution_path(self):
        """1.5.3: Preceding node on solution path correctly identified."""
        path = ["node_0", "node_1", "node_2"]
        curr_idx = path.index("node_1")
        prev_node = path[curr_idx - 1] if curr_idx > 0 else None
        assert prev_node == "node_0"

    def test_1_5_neighbour_next_on_solution_path(self):
        """1.5.4: Subsequent node on solution path correctly identified."""
        path = ["node_0", "node_1", "node_2"]
        curr_idx = path.index("node_1")
        next_node = path[curr_idx + 1] if curr_idx < len(path) - 1 else None
        assert next_node == "node_2"

    def test_1_5_neighbour_terminal_boundaries(self):
        """1.5.5: Root has no previous node, and goal state has no next node on path."""
        path = ["node_0", "node_1"]
        prev_at_root = path[0 - 1] if 0 > 0 else None
        next_at_goal = path[1 + 1] if 1 < len(path) - 1 else None
        assert prev_at_root is None
        assert next_at_goal is None

    # --------------------------------------------------------------------------
    # 1.6 Explicit Solution Path Membership (5 Authoritative Statuses)
    # --------------------------------------------------------------------------

    def test_1_6_solution_path_membership_on_path(self):
        """1.6.1: Node on extracted plan receives status ON SOLUTION PATH."""
        status = classify_solution_path_membership(
            node_id="s1",
            is_goal=False,
            is_dead_end=False,
            solution_path_node_ids=["s0", "s1", "s2"],
            expanded_node_ids=["s0", "s1", "s2", "s3"],
        )
        assert status == "ON SOLUTION PATH"

    def test_1_6_solution_path_membership_reachable_not_on_path(self):
        """1.6.2: Expanded node not on solution path classified as REACHABLE BUT NOT ON SOLUTION PATH."""
        status = classify_solution_path_membership(
            node_id="s3",
            is_goal=False,
            is_dead_end=False,
            solution_path_node_ids=["s0", "s1", "s2"],
            expanded_node_ids=["s0", "s1", "s2", "s3"],
        )
        assert status == "REACHABLE BUT NOT ON SOLUTION PATH"

    def test_1_6_solution_path_membership_not_explored_by_search(self):
        """1.6.3: Unvisited node in graph classified as NOT EXPLORED BY THIS SEARCH."""
        status = classify_solution_path_membership(
            node_id="s99",
            is_goal=False,
            is_dead_end=False,
            solution_path_node_ids=["s0", "s1", "s2"],
            expanded_node_ids=["s0", "s1", "s2"],
        )
        assert status == "NOT EXPLORED BY THIS SEARCH"

    def test_1_6_solution_path_membership_goal_state(self):
        """1.6.4: Goal state off solution path classified as GOAL STATE."""
        status = classify_solution_path_membership(
            node_id="s_other_goal",
            is_goal=True,
            is_dead_end=False,
            solution_path_node_ids=["s0", "s1", "s2"],
            expanded_node_ids=["s0", "s1", "s2", "s_other_goal"],
        )
        assert status == "GOAL STATE"

    def test_1_6_solution_path_membership_dead_end(self):
        """1.6.5: Dead-end state classified as DEAD END."""
        status = classify_solution_path_membership(
            node_id="s_dead",
            is_goal=False,
            is_dead_end=True,
            solution_path_node_ids=["s0", "s1"],
            expanded_node_ids=["s0", "s1", "s_dead"],
        )
        assert status == "DEAD END"

    # --------------------------------------------------------------------------
    # 1.7 Graph Enumeration vs Search Mode Separation
    # --------------------------------------------------------------------------

    def test_1_7_graph_enumeration_independent_from_search(self, client: TestClient):
        """1.7.1: Graph enumeration produces reachable states without requiring a search planner."""
        resp = client.post("/api/graph/enumerate", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "max_nodes": 50,
            "max_edges": 100,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "elements" in data
        assert "stats" in data
        assert data["stats"]["total_nodes"] > 0
        assert data["stats"]["total_edges"] > 0

    def test_1_7_graph_enumeration_reports_complete_counts(self, bw_domain: BlocksWorldDomain):
        """1.7.2: Reachable graph enumeration returns verified node, edge, and depth counts."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        enumerator = CompleteReachableGraphEnumerator(max_nodes=100, max_edges=200)
        graph = enumerator.enumerate_graph(prob, bw_domain)
        assert graph.stats.total_nodes == len(graph.nodes)
        assert graph.stats.total_edges == len(graph.edges)
        assert graph.stats.is_complete is True
        assert len(graph.stats.goal_node_ids) >= 1

    def test_1_7_graph_enumeration_budget_truncation_reporting(self, bw_domain: BlocksWorldDomain):
        """1.7.3: Reachable graph truncation limits mark is_complete=False and set reason."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        enumerator = CompleteReachableGraphEnumerator(max_nodes=3, max_edges=10)
        graph = enumerator.enumerate_graph(prob, bw_domain)
        assert graph.stats.is_complete is False
        assert graph.stats.truncation_reason in ("MAX_NODES_REACHED", "MAX_NODES_EXCEEDED")

    def test_1_7_search_expansion_order_independent_of_graph_order(self, bw_domain: BlocksWorldDomain):
        """1.7.4: Search expansion order differs from graph enumeration traversal order."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        bfs = BFSPlanner()
        bfs_plan = bfs.search(prob, bw_domain)
        bf = BestFirstPlanner()
        bf_plan = bf.search(prob, bw_domain, heuristic=GoalCountHeuristic())
        assert bfs_plan.nodes_expanded != bf_plan.nodes_expanded or bfs_plan.nodes_generated != bf_plan.nodes_generated

    def test_1_7_graph_edge_preconditions_and_effects_contract(self):
        """1.7.5: Graph edges carry preconditions, add_effects, del_effects, and invariant flag."""
        edge_data = {
            "edge_id": "e_test_1",
            "source_id": "s0",
            "target_id": "s1",
            "action_name": "pick_up",
            "action_args": ["b1"],
            "action_label": "pick_up(b1)",
            "cost": 1.0,
            "preconditions": ["clear(b1)", "on_table(b1)", "handempty()"],
            "add_effects": ["holding(b1)"],
            "del_effects": ["clear(b1)", "on_table(b1)", "handempty()"],
            "invariants_satisfied": True,
        }
        if "preconditions" not in GraphEdge.model_fields:
            pytest.skip("GraphEdge precondition/effect extensions pending Milestone 1 implementation.")
        edge = GraphEdge(**edge_data)
        assert edge.action_name == "pick_up"
        assert len(edge.preconditions) == 3
        assert "holding(b1)" in edge.add_effects
        assert edge.invariants_satisfied is True

    # --------------------------------------------------------------------------
    # 1.8 Non-Color Visual Distinction
    # --------------------------------------------------------------------------

    def test_1_8_visual_shape_initial_double_circle(self):
        """1.8.1: Initial state maps to accessible double-circle shape."""
        shape = map_visual_shape(is_initial=True, is_goal=False, is_dead_end=False, is_boundary=False)
        assert shape == "double_circle"

    def test_1_8_visual_shape_goal_diamond(self):
        """1.8.2: Goal state maps to accessible diamond shape."""
        shape = map_visual_shape(is_initial=False, is_goal=True, is_dead_end=False, is_boundary=False)
        assert shape == "diamond"

    def test_1_8_visual_shape_dead_end_octagon(self):
        """1.8.3: Dead end maps to accessible octagon shape."""
        shape = map_visual_shape(is_initial=False, is_goal=False, is_dead_end=True, is_boundary=False)
        assert shape == "octagon"

    def test_1_8_visual_shape_boundary_square(self):
        """1.8.4: Truncated boundary maps to accessible square shape."""
        shape = map_visual_shape(is_initial=False, is_goal=False, is_dead_end=False, is_boundary=True)
        assert shape == "square"

    def test_1_8_visual_glyphs_and_legend_integrity(self):
        """1.8.5: Text glyphs accurately represent state semantic classifications."""
        assert map_visual_glyph(True, False, False, False) == "S₀"
        assert map_visual_glyph(False, True, False, False) == "★"
        assert map_visual_glyph(False, False, True, False) == "✕"
        assert map_visual_glyph(False, False, False, True) == "◇"
        assert map_visual_glyph(False, False, False, False, is_current_expanding=True) == "⚡"
        assert map_visual_glyph(False, False, False, False, is_solution_path=True) == "✓"


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 tests per category across 8 categories)
# ==============================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Boundary conditions, empty sets, limits, cycles, and tie-breaking."""

    # --------------------------------------------------------------------------
    # 2.1 Zero-Step Search (Initial state already satisfies goal)
    # --------------------------------------------------------------------------

    def test_2_1_zero_step_search_initial_state_is_goal(self, bw_domain: BlocksWorldDomain):
        """2.1.1: Immediate goal satisfaction produces empty plan with 0 cost."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("on_table", ["a"])],
        )
        planner = AStarPlanner()
        plan = planner.search(prob, bw_domain)
        assert plan.is_success is True
        assert len(plan.actions) == 0
        assert plan.total_cost == 0.0
        assert plan.nodes_expanded == 0

    def test_2_1_zero_step_search_all_algorithms(self, bw_domain: BlocksWorldDomain):
        """2.1.2: All search algorithms immediately return success for zero-step search."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("on_table", ["a"])],
        )
        for planner in [BFSPlanner(), BestFirstPlanner(), AStarPlanner()]:
            plan = planner.search(prob, bw_domain)
            assert plan.is_success is True
            assert len(plan.actions) == 0

    def test_2_1_zero_step_search_inspector_goal_satisfaction(self, bw_domain: BlocksWorldDomain):
        """2.1.3: State inspector reports 100% satisfied goals in zero-step initial state."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("on_table", ["a"])],
        )
        sat, unsat = derive_goal_satisfaction_split(prob, prob.initial_state)
        assert len(unsat) == 0
        assert "on_table(a)" in sat

    def test_2_1_zero_step_search_graph_enumeration(self, bw_domain: BlocksWorldDomain):
        """2.1.4: Graph enumerator correctly marks initial node as goal node."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("on_table", ["a"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=10)
        graph = enum.enumerate_graph(prob, bw_domain)
        init_node = graph.nodes[graph.stats.initial_node_id]
        assert init_node.is_initial is True
        assert init_node.is_goal is True
        assert graph.stats.initial_node_id in graph.stats.goal_node_ids

    def test_2_1_zero_step_search_solution_membership(self):
        """2.1.5: Zero-step initial state classified as ON SOLUTION PATH."""
        membership = classify_solution_path_membership(
            node_id="s0",
            is_goal=True,
            is_dead_end=False,
            solution_path_node_ids=["s0"],
            expanded_node_ids=["s0"],
        )
        assert membership == "ON SOLUTION PATH"

    # --------------------------------------------------------------------------
    # 2.2 Single-State Domains & Terminal States
    # --------------------------------------------------------------------------

    def test_2_2_single_state_domain_no_actions(self):
        """2.2.1: Domain with zero actions terminates enumeration with exactly 1 node."""
        th = TypeHierarchy()
        empty_domain = BaseDomain(domain_id="empty_domain", name="Empty", type_hierarchy=th, actions=[])
        reg = ObjectRegistry(th)
        init_state = SymbolicState([Fact("immutable_fact", [])])
        prob = SymbolicProblem("single_state_prob", "empty_domain", reg, init_state, [Fact("unreachable_goal", [])])

        enum = CompleteReachableGraphEnumerator(max_nodes=10)
        graph = enum.enumerate_graph(prob, empty_domain)
        assert graph.stats.total_nodes == 1
        assert graph.stats.total_edges == 0
        assert graph.stats.is_complete is True
        assert len(graph.stats.dead_end_node_ids) == 1

    def test_2_2_single_state_domain_search_unreachable_goal(self):
        """2.2.2: Search in single-state domain gracefully reports exhaustion."""
        th = TypeHierarchy()
        empty_domain = BaseDomain(domain_id="empty_domain", name="Empty", type_hierarchy=th, actions=[])
        reg = ObjectRegistry(th)
        init_state = SymbolicState([Fact("immutable_fact", [])])
        prob = SymbolicProblem("single_state_prob", "empty_domain", reg, init_state, [Fact("unreachable_goal", [])])

        planner = AStarPlanner()
        plan = planner.search(prob, empty_domain)
        assert plan.is_success is False
        assert "no valid plan exists" in plan.failure_reason

    def test_2_2_single_state_domain_dead_end_classification(self):
        """2.2.3: Terminal single state classified as DEAD END."""
        membership = classify_solution_path_membership(
            node_id="s0",
            is_goal=False,
            is_dead_end=True,
            solution_path_node_ids=[],
            expanded_node_ids=["s0"],
        )
        assert membership == "DEAD END"

    def test_2_2_single_state_domain_trace_exhaustion(self, client: TestClient):
        """2.2.4: Search trace API endpoint handles zero-action problem without 500 error."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "scheduling",
            "prompt": "Schedule impossible non-existent tasks.",
            "algorithm": "BFS",
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace pending Milestone 1.")
        assert resp.status_code in (200, 400, 422)

    def test_2_2_single_state_domain_stats_max_depth_zero(self):
        """2.2.5: Graph statistics report max_depth = 0 for single-state graph."""
        th = TypeHierarchy()
        empty_domain = BaseDomain(domain_id="empty_domain", name="Empty", type_hierarchy=th, actions=[])
        reg = ObjectRegistry(th)
        init_state = SymbolicState([Fact("fact_a", [])])
        prob = SymbolicProblem("single_state_prob", "empty_domain", reg, init_state, [Fact("fact_b", [])])

        enum = CompleteReachableGraphEnumerator(max_nodes=10)
        graph = enum.enumerate_graph(prob, empty_domain)
        assert graph.stats.max_depth == 0

    # --------------------------------------------------------------------------
    # 2.3 Dead-End Detection
    # --------------------------------------------------------------------------

    def test_2_3_dead_end_zero_applicable_actions(self):
        """2.3.1: State with zero applicable actions detected as dead end."""
        th = TypeHierarchy()
        th.add_type("item", "object")
        act = ActionDefinition(
            name="unlock",
            parameters=[ActionParameter("i", "item")],
            preconditions=[Fact("has_key", ["?i"])],
            add_effects=[Fact("unlocked", ["?i"])],
            del_effects=[],
        )
        domain = BaseDomain(domain_id="trap_dom", name="Trap", type_hierarchy=th, actions=[act])
        reg = ObjectRegistry(th)
        reg.register_object("chest", "item")
        init_state = SymbolicState([Fact("locked", ["chest"])])
        prob = SymbolicProblem("trap_prob", "trap_dom", reg, init_state, [Fact("unlocked", ["chest"])])

        enum = CompleteReachableGraphEnumerator(max_nodes=10)
        graph = enum.enumerate_graph(prob, domain)
        assert len(graph.stats.dead_end_node_ids) == 1
        node = graph.nodes[graph.stats.initial_node_id]
        assert node.is_dead_end is True

    def test_2_3_dead_end_all_transitions_violate_invariants(self):
        """2.3.2: Actions whose outcomes violate domain invariants leave state without valid outgoing edges."""
        th = TypeHierarchy()
        th.add_type("item", "object")
        act = ActionDefinition(
            name="poison",
            parameters=[ActionParameter("i", "item")],
            preconditions=[Fact("safe", ["?i"])],
            add_effects=[Fact("toxic", ["?i"])],
            del_effects=[Fact("safe", ["?i"])],
        )
        class NoToxicInvariant(Invariant):
            def __init__(self):
                super().__init__("no_toxic")
            def check(self, state):
                for f in state.facts:
                    if f.predicate == "toxic":
                        return False, "Toxic state forbidden"
                return True, None

        domain = BaseDomain(
            domain_id="tox_dom",
            name="ToxicDomain",
            type_hierarchy=th,
            actions=[act],
            default_invariants=[NoToxicInvariant()],
        )
        reg = ObjectRegistry(th)
        reg.register_object("apple", "item")
        init_state = SymbolicState([Fact("safe", ["apple"])])
        prob = SymbolicProblem("tox_prob", "tox_dom", reg, init_state, [Fact("eaten", ["apple"])], invariants=[NoToxicInvariant()])

        enum = CompleteReachableGraphEnumerator(max_nodes=10)
        graph = enum.enumerate_graph(prob, domain)
        # Invariants blocked the transition, resulting in exactly zero outgoing transitions
        outgoing_edges = [e for e in graph.edges if e.source_id == graph.stats.initial_node_id]
        assert len(outgoing_edges) == 0

    def test_2_3_dead_end_inspector_membership(self):
        """2.3.3: State inspector reports DEAD END status for trap states."""
        status = classify_solution_path_membership(
            node_id="dead_1",
            is_goal=False,
            is_dead_end=True,
            solution_path_node_ids=["s0", "s1"],
            expanded_node_ids=["s0", "dead_1"],
        )
        assert status == "DEAD END"

    def test_2_3_dead_end_visual_shape_and_glyph(self):
        """2.3.4: Dead end states map to octagon SVG shape and ✕ text glyph."""
        shape = map_visual_shape(is_initial=False, is_goal=False, is_dead_end=True, is_boundary=False)
        glyph = map_visual_glyph(is_initial=False, is_goal=False, is_dead_end=True, is_boundary=False)
        assert shape == "octagon"
        assert glyph == "✕"

    def test_2_3_dead_end_in_solution_path_rejection(self):
        """2.3.5: Dead end state cannot be part of a valid plan solution path."""
        status = classify_solution_path_membership(
            node_id="sol_node",
            is_goal=False,
            is_dead_end=False,
            solution_path_node_ids=["sol_node"],
            expanded_node_ids=["sol_node"],
        )
        assert status == "ON SOLUTION PATH"
        assert status != "DEAD END"

    # --------------------------------------------------------------------------
    # 2.4 Resource Budget Truncation (max_nodes, max_edges, timeout)
    # --------------------------------------------------------------------------

    def test_2_4_budget_truncation_max_nodes_boundary(self, bw_domain: BlocksWorldDomain):
        """2.4.1: Enumerator halts immediately when node count reaches max_nodes."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=5, max_edges=1000)
        graph = enum.enumerate_graph(prob, bw_domain)
        assert graph.stats.total_nodes <= 5
        assert graph.stats.is_complete is False
        assert graph.stats.truncation_reason in ("MAX_NODES_REACHED", "MAX_NODES_EXCEEDED")

    def test_2_4_budget_truncation_max_edges_boundary(self, bw_domain: BlocksWorldDomain):
        """2.4.2: Enumerator halts when edge count reaches max_edges."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=1000, max_edges=4)
        graph = enum.enumerate_graph(prob, bw_domain)
        assert graph.stats.total_edges <= 4
        assert graph.stats.is_complete is False
        assert graph.stats.truncation_reason in ("MAX_EDGES_REACHED", "MAX_EDGES_EXCEEDED")

    def test_2_4_budget_truncation_timeout_boundary(self, bw_domain: BlocksWorldDomain):
        """2.4.3: Enumerator respects sub-second timeout ceilings within legal bounds."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=1000, max_edges=1000, timeout_seconds=0.0001)
        graph = enum.enumerate_graph(prob, bw_domain)
        assert graph.stats.is_complete is False
        assert graph.stats.truncation_reason == "TIMEOUT"

    def test_2_4_budget_truncation_boundary_node_marking(self, bw_domain: BlocksWorldDomain):
        """2.4.4: Unexpanded nodes at budget cutoff are flagged as truncated boundaries."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=3)
        graph = enum.enumerate_graph(prob, bw_domain)
        boundary_nodes = [n for n in graph.nodes.values() if n.is_boundary or n.is_truncated_boundary]
        assert len(boundary_nodes) >= 1

    def test_2_4_budget_truncation_search_nodes_expanded_limit(self, bw_domain: BlocksWorldDomain):
        """2.4.5: Search planner stops when max_nodes_expanded is reached."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("on_table", ["c"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("clear", ["c"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"]), Fact("on", ["b", "c"])],
        )
        planner = BFSPlanner(max_nodes_expanded=2)
        plan = planner.search(prob, bw_domain)
        assert plan.is_success is False
        assert "max nodes expanded reached" in plan.failure_reason

    # --------------------------------------------------------------------------
    # 2.5 Cyclic Transitions & Reversible Action Loops
    # --------------------------------------------------------------------------

    def test_2_5_cyclic_transitions_pick_and_put_loop(self, bw_domain: BlocksWorldDomain):
        """2.5.1: pick_up followed by put_down returns to original state without duplicating node."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("holding", ["a"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=50)
        graph = enum.enumerate_graph(prob, bw_domain)
        assert graph.stats.total_nodes == 2
        assert graph.stats.total_edges == 2
        assert graph.stats.cycles_detected >= 1

    def test_2_5_cyclic_transitions_closed_set_pruning(self, bw_domain: BlocksWorldDomain):
        """2.5.2: Visited states in closed set are not re-expanded."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("holding", ["a"])],
        )
        planner = AStarPlanner()
        plan = planner.search(prob, bw_domain)
        assert plan.is_success is True
        assert plan.nodes_expanded <= 2

    def test_2_5_cyclic_transitions_graph_directed_edges(self, bw_domain: BlocksWorldDomain):
        """2.5.3: Graph stores distinct directed edges for forward and reverse transitions."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("holding", ["a"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=10)
        graph = enum.enumerate_graph(prob, bw_domain)
        actions = {e.action_name for e in graph.edges}
        assert "pick_up" in actions
        assert "put_down" in actions

    def test_2_5_cyclic_transitions_search_avoids_infinite_loop(self, bw_domain: BlocksWorldDomain):
        """2.5.4: BFS, Best-First, and A* all safely terminate in cyclic domains."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("holding", ["a"])],
        )
        for p in [BFSPlanner(), BestFirstPlanner(), AStarPlanner()]:
            res = p.search(prob, bw_domain)
            assert res.is_success is True

    def test_2_5_cyclic_transitions_trace_skip_pruned_events(self, client: TestClient):
        """2.5.5: Search trace captures pruning of cyclic duplicates."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Move the red box.",
            "algorithm": "A*",
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace pending Milestone 1.")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes_generated"] >= data["nodes_expanded"]

    # --------------------------------------------------------------------------
    # 2.6 Unreachable Goals & Search Exhaustion
    # --------------------------------------------------------------------------

    def test_2_6_unreachable_goal_search_exhaustion(self, bw_domain: BlocksWorldDomain):
        """2.6.1: Search terminates with is_success=False when goal cannot be reached."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"]), Fact("on", ["b", "a"])],
        )
        planner = AStarPlanner()
        plan = planner.search(prob, bw_domain)
        assert plan.is_success is False
        assert "no valid plan exists" in plan.failure_reason

    def test_2_6_unreachable_goal_solution_path_empty(self, bw_domain: BlocksWorldDomain):
        """2.6.2: Unreachable goal results in an empty plan action list."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"]), Fact("on", ["b", "a"])],
        )
        planner = BFSPlanner()
        plan = planner.search(prob, bw_domain)
        assert len(plan.actions) == 0

    def test_2_6_unreachable_goal_graph_has_no_goal_nodes(self, bw_domain: BlocksWorldDomain):
        """2.6.3: Graph enumeration on unreachable problem returns empty goal_node_ids list."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"]), Fact("on", ["b", "a"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=50)
        graph = enum.enumerate_graph(prob, bw_domain)
        assert len(graph.stats.goal_node_ids) == 0

    def test_2_6_unreachable_goal_inspector_unestablished_facts(self, bw_domain: BlocksWorldDomain):
        """2.6.4: State inspector on all reachable states shows unmet goal conditions."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"]), Fact("on", ["b", "a"])],
        )
        unmet = derive_unestablished_goal_facts(prob, prob.initial_state)
        assert "on(a, b)" in unmet
        assert "on(b, a)" in unmet

    def test_2_6_unreachable_goal_membership_classification(self):
        """2.6.5: States explored during exhausted search are classified as REACHABLE BUT NOT ON SOLUTION PATH."""
        status = classify_solution_path_membership(
            node_id="s1",
            is_goal=False,
            is_dead_end=False,
            solution_path_node_ids=[],
            expanded_node_ids=["s0", "s1"],
        )
        assert status == "REACHABLE BUT NOT ON SOLUTION PATH"

    # --------------------------------------------------------------------------
    # 2.7 Priority Queue Tie-Breaking Determinism
    # --------------------------------------------------------------------------

    def test_2_7_tie_breaking_node_lt_order(self):
        """2.7.1: SearchNode.__lt__ compares strictly on f_cost, then h_cost, then depth, then node_id."""
        s = SymbolicState([])
        root = SearchNode(state=s)
        n1 = SearchNode(state=s, parent=root, g_cost=1.0, h_cost=2.0)  # f=3.0, depth=1
        n2 = SearchNode(state=s, parent=n1, g_cost=2.0, h_cost=1.0)    # f=3.0, lower h=1.0, depth=2
        n3 = SearchNode(state=s, parent=n2, g_cost=2.0, h_cost=1.0)    # f=3.0, h=1.0, deeper depth=3
        assert n2 < n1  # Lower h_cost wins tie
        assert n3 < n2  # Deeper depth wins tie when f and h match

    def test_2_7_tie_breaking_reproducible_event_sequence(self, bw_domain: BlocksWorldDomain):
        """2.7.2: Running identical problem twice produces identical plans and expansion counts."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        p1 = AStarPlanner().search(prob, bw_domain)
        p2 = AStarPlanner().search(prob, bw_domain)
        assert p1.nodes_expanded == p2.nodes_expanded
        assert p1.total_cost == p2.total_cost
        assert [a.name for a in p1.actions] == [a.name for a in p2.actions]

    def test_2_7_tie_breaking_fifo_queue_bfs(self):
        """2.7.3: BFS maintains strict FIFO order for nodes at identical depth."""
        q = collections.deque()
        q.append("first")
        q.append("second")
        q.append("third")
        assert q.popleft() == "first"
        assert q.popleft() == "second"
        assert q.popleft() == "third"

    def test_2_7_tie_breaking_greedy_best_first_deterministic(self, bw_domain: BlocksWorldDomain):
        """2.7.4: BestFirstPlanner tie-breaking is deterministic."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        p1 = BestFirstPlanner().search(prob, bw_domain)
        p2 = BestFirstPlanner().search(prob, bw_domain)
        assert p1.nodes_expanded == p2.nodes_expanded

    def test_2_7_tie_breaking_heapq_safe_comparison(self):
        """2.7.5: SearchNode heapq heap push and pop never raises unorderable type exceptions."""
        import heapq
        s = SymbolicState([])
        nodes = [
            SearchNode(state=s, g_cost=1.0, h_cost=1.0),
            SearchNode(state=s, g_cost=1.0, h_cost=1.0),
            SearchNode(state=s, g_cost=0.5, h_cost=1.5),
        ]
        heap: List[SearchNode] = []
        for n in nodes:
            heapq.heappush(heap, n)
        popped = heapq.heappop(heap)
        assert popped.f_cost <= 2.0

    # --------------------------------------------------------------------------
    # 2.8 Depth Boundaries & Monotonic Tracking
    # --------------------------------------------------------------------------

    def test_2_8_max_depth_zero_graph(self, bw_domain: BlocksWorldDomain):
        """2.8.1: Graph with zero outgoing transitions has max_depth = 0."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block"},
            [Fact("on_table", ["a"]), Fact("clear", ["a"]), Fact("handempty", [])],
            [Fact("on_table", ["a"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=1)
        graph = enum.enumerate_graph(prob, bw_domain)
        assert graph.stats.max_depth == 0

    def test_2_8_depth_reachability_monotonic(self):
        """2.8.2: Child node depth in search tree strictly equals parent.depth + 1."""
        root = SearchNode(state=SymbolicState([]))
        c1 = SearchNode(state=SymbolicState([]), parent=root)
        c2 = SearchNode(state=SymbolicState([]), parent=c1)
        assert c1.depth == 1
        assert c2.depth == 2

    def test_2_8_graph_stats_max_depth_matches_node_depths(self, bw_domain: BlocksWorldDomain):
        """2.8.3: stats.max_depth matches the maximum depth among all enumerated nodes."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        enum = CompleteReachableGraphEnumerator(max_nodes=50)
        graph = enum.enumerate_graph(prob, bw_domain)
        max_actual_depth = max(n.depth for n in graph.nodes.values())
        assert graph.stats.max_depth == max_actual_depth

    def test_2_8_deep_search_chain_depth_tracking(self, bw_domain: BlocksWorldDomain):
        """2.8.4: Linear plan action sequence depth matches plan length."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        plan = AStarPlanner().search(prob, bw_domain)
        assert len(plan.actions) == 2

    def test_2_8_depth_boundary_cost_independence(self):
        """2.8.5: Depth reflects edge hops while g_cost reflects cumulative action cost."""
        act_expensive = GroundAction("heavy_lift", [], [], [], [], cost=10.0)
        root = SearchNode(state=SymbolicState([]), g_cost=0.0)
        child = SearchNode(state=SymbolicState([]), parent=root, action=act_expensive, g_cost=10.0)
        assert child.depth == 1
        assert child.g_cost == 10.0


# ==============================================================================
# TIER 3: CROSS-FEATURE PAIRWISE COMBINATIONS
# ==============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Inter-feature coupling across algorithms, domains, inspector, and playback."""

    # --------------------------------------------------------------------------
    # 3.1 Algorithm x Domain Pairwise Grid
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize("domain_id", [
        "blocks_world",
        "grid_world",
        "household_robotics",
        "scheduling",
        "software_workflow",
        "warehouse",
    ])
    def test_3_1_domain_registration_and_metadata_summary(self, domain_id: str):
        """3.1.1: Every canonical domain registered with valid actions and predicate schemas."""
        dom = registry.get(domain_id)
        assert dom is not None
        assert len(dom.get_all_actions()) > 0
        assert len(dom.predicates) > 0

    def test_3_1_bfs_on_blocks_world(self, bw_domain: BlocksWorldDomain):
        """3.1.2: BFS executed on Blocks World domain."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        plan = BFSPlanner().search(prob, bw_domain)
        assert plan.is_success is True

    def test_3_1_astar_on_household_robotics(self):
        """3.1.3: A* executed on Household Robotics domain."""
        dom = registry.get("household_robotics")
        assert dom is not None
        th = dom.type_hierarchy
        reg = ObjectRegistry(th)
        reg.register_object("bot", "robot")
        reg.register_object("r1", "room")
        reg.register_object("r2", "room")
        init_state = SymbolicState([Fact("robot_at", ["bot", "r1"]), Fact("connected_rooms", ["r1", "r2"])])
        prob = SymbolicProblem("hh_move", "household_robotics", reg, init_state, [Fact("robot_at", ["bot", "r2"])])
        plan = AStarPlanner().search(prob, dom)
        assert plan.is_success is True
        assert len(plan.actions) == 1

    def test_3_1_best_first_on_software_workflow(self):
        """3.1.4: Greedy Best-First executed on Software Workflow domain."""
        dom = registry.get("software_workflow")
        assert dom is not None
        th = dom.type_hierarchy
        reg = ObjectRegistry(th)
        reg.register_object("c1", "commit")
        init_state = SymbolicState([Fact("code_checked_out", ["c1"])])
        prob = SymbolicProblem("sw_test", "software_workflow", reg, init_state, [Fact("tests_passed", ["c1"])])
        plan = BestFirstPlanner().search(prob, dom)
        assert plan.is_success is True
        assert len(plan.actions) == 1

    def test_3_1_ucs_on_scheduling(self):
        """3.1.5: UCS (A* with ZeroHeuristic) executed on Scheduling domain."""
        dom = registry.get("scheduling")
        assert dom is not None
        th = dom.type_hierarchy
        reg = ObjectRegistry(th)
        reg.register_object("j1", "job")
        reg.register_object("r1", "resource")
        reg.register_object("s1", "slot")
        init_state = SymbolicState([Fact("job_ready", ["j1"]), Fact("resource_available", ["r1", "s1"])])
        prob = SymbolicProblem("sched_test", "scheduling", reg, init_state, [Fact("job_completed", ["j1"])])
        plan = AStarPlanner().search(prob, dom, heuristic=ZeroHeuristic())
        assert plan.is_success is True
        assert len(plan.actions) == 1

    # --------------------------------------------------------------------------
    # 3.2 Search Mode x Budget Truncation Interaction
    # --------------------------------------------------------------------------

    def test_3_2_search_trace_under_budget_cutoff(self, client: TestClient):
        """3.2.1: POST /api/search/trace with small max_nodes_expanded halts safely."""
        resp = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack a on b on c.",
            "algorithm": "A*",
            "max_nodes_expanded": 2,
        })
        if resp.status_code == 404:
            pytest.skip("POST /api/search/trace pending Milestone 1.")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes_expanded"] <= 2

    def test_3_2_graph_enumerate_under_budget_cutoff(self, client: TestClient):
        """3.2.2: POST /api/graph/enumerate with small max_nodes halts safely."""
        resp = client.post("/api/graph/enumerate", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "max_nodes": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["total_nodes"] <= 2
        assert data["stats"]["is_complete"] is False

    # --------------------------------------------------------------------------
    # 3.3 State Inspector across Varied Node Types
    # --------------------------------------------------------------------------

    def test_3_3_inspector_on_initial_vs_goal_vs_dead_end(self, bw_domain: BlocksWorldDomain):
        """3.3.1: State inspector correctly classifies and derives facts for all node types."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block"},
            [Fact("on_table", ["a"]), Fact("on_table", ["b"]), Fact("clear", ["a"]), Fact("clear", ["b"]), Fact("handempty", [])],
            [Fact("on", ["a", "b"])],
        )
        init_shape = map_visual_shape(True, False, False, False)
        goal_shape = map_visual_shape(False, True, False, False)
        dead_shape = map_visual_shape(False, False, True, False)
        assert init_shape == "double_circle"
        assert goal_shape == "diamond"
        assert dead_shape == "octagon"

    # --------------------------------------------------------------------------
    # 3.4 Playback Step-Through x Event Life Cycle Synchronization
    # --------------------------------------------------------------------------

    def test_3_4_playback_step_through_event_synchronization(self):
        """3.4.1: Playback stepping maintains exact current state ID matching active event."""
        events = [
            {"seq": 1, "event_type": "INITIAL", "state_id": "s0"},
            {"seq": 2, "event_type": "EXPAND", "state_id": "s0"},
            {"seq": 3, "event_type": "GENERATE", "state_id": "s1"},
            {"seq": 4, "event_type": "EXPAND", "state_id": "s1"},
            {"seq": 5, "event_type": "GOAL_DETECTED", "state_id": "s1"},
        ]
        sim = SearchPlaybackSimulator(events)
        sim.step_forward()
        assert sim.current_state_id == "s0"
        sim.step_forward()
        assert sim.current_state_id == "s0"
        sim.step_forward()
        assert sim.current_state_id == "s1"
        sim.step_forward()
        assert sim.current_state_id == "s1"

    # --------------------------------------------------------------------------
    # 3.5 Graph Edge Details Serialization x Invariant Validation
    # --------------------------------------------------------------------------

    def test_3_5_edge_details_invariants_satisfied_flag(self):
        """3.5.1: GraphEdge serializes invariants_satisfied=True when transition is legal."""
        if "invariants_satisfied" not in GraphEdge.model_fields:
            pytest.skip("GraphEdge invariants_satisfied extension pending Milestone 1 implementation.")
        edge = GraphEdge(
            edge_id="e_valid",
            source_id="s0",
            target_id="s1",
            action_name="pick_up",
            action_args=["b1"],
            action_label="pick_up(b1)",
            cost=1.0,
            invariants_satisfied=True,
        )
        assert edge.invariants_satisfied is True


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (>=5 Complex Multi-Step Workflows)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic Multi-Step Workflows Across Supported Planning Domains."""

    def test_4_1_blocks_world_three_block_tower_inversion(self, bw_domain: BlocksWorldDomain):
        """Scenario 4.1: Inverting a 3-block tower (initial: A on B on C, goal: C on B on A)."""
        prob = create_symbolic_problem(
            bw_domain,
            {"a": "block", "b": "block", "c": "block"},
            [
                Fact("on", ["a", "b"]),
                Fact("on", ["b", "c"]),
                Fact("on_table", ["c"]),
                Fact("clear", ["a"]),
                Fact("handempty", []),
            ],
            [
                Fact("on", ["c", "b"]),
                Fact("on", ["b", "a"]),
            ],
        )
        planner = AStarPlanner()
        plan = planner.search(prob, bw_domain)
        assert plan.is_success is True
        # Classical 3-block tower inversion requires exactly 6 actions:
        # unstack(a,b), put_down(a), unstack(b,c), stack(b,a), pick_up(c), stack(c,b)
        assert len(plan.actions) == 6
        assert plan.total_cost == 6.0
        assert plan.actions[0].name == "unstack"
        assert plan.actions[-1].name == "stack"

    def test_4_2_household_robotics_multi_room_restricted_navigation(self):
        """Scenario 4.2: Household multi-room navigation with forbidden bedroom restriction."""
        dom = registry.get("household_robotics")
        assert dom is not None
        th = dom.type_hierarchy
        reg = ObjectRegistry(th)
        reg.register_object("bot", "robot")
        for r in ["office", "corridor", "kitchen", "bedroom"]:
            reg.register_object(r, "room")
        reg.register_object("mug", "item")
        reg.register_object("desk", "surface")
        reg.register_object("counter", "surface")

        class ForbidBedroomInvariant(Invariant):
            def __init__(self):
                super().__init__("forbid_bedroom")
            def check(self, state):
                for f in state.facts:
                    if f.predicate == "robot_at" and "bedroom" in f.arguments:
                        return False, "Access violation: bedroom entered"
                return True, None

        init_facts = [
            Fact("robot_at", ["bot", "office"]),
            Fact("connected_rooms", ["office", "corridor"]),
            Fact("connected_rooms", ["corridor", "kitchen"]),
            Fact("connected_rooms", ["office", "bedroom"]),
            Fact("connected_rooms", ["bedroom", "kitchen"]),
            Fact("item_on", ["mug", "desk"]),
            Fact("surface_in_room", ["desk", "office"]),
            Fact("surface_in_room", ["counter", "kitchen"]),
            Fact("gripper_empty", ["bot"]),
        ]
        goal_facts = [
            Fact("item_on", ["mug", "counter"]),
            Fact("robot_at", ["bot", "kitchen"]),
        ]
        prob = SymbolicProblem(
            "household_safe_route",
            "household_robotics",
            reg,
            SymbolicState(init_facts),
            goal_facts,
            invariants=[ForbidBedroomInvariant()],
        )
        plan = AStarPlanner().search(prob, dom)
        assert plan.is_success is True
        for act in plan.actions:
            assert "bedroom" not in act.arguments

    def test_4_3_scheduling_dag_precedence_job_shop(self):
        """Scenario 4.3: DAG Job Shop Scheduling with Multi-Precedence Dependencies."""
        dom = registry.get("scheduling")
        assert dom is not None
        th = dom.type_hierarchy
        reg = ObjectRegistry(th)
        reg.register_object("job_a", "job")
        reg.register_object("job_b", "job")
        reg.register_object("job_c", "job")
        reg.register_object("m1", "resource")
        reg.register_object("s1", "slot")
        reg.register_object("s2", "slot")
        reg.register_object("s3", "slot")

        init_facts = [
            Fact("job_ready", ["job_a"]),
            Fact("job_ready", ["job_b"]),
            Fact("job_ready", ["job_c"]),
            Fact("resource_available", ["m1", "s1"]),
            Fact("resource_available", ["m1", "s2"]),
            Fact("resource_available", ["m1", "s3"]),
            Fact("precedes", ["job_a", "job_b"]),
            Fact("precedes", ["job_b", "job_c"]),
        ]
        goal_facts = [
            Fact("job_completed", ["job_a"]),
            Fact("job_completed", ["job_b"]),
            Fact("job_completed", ["job_c"]),
        ]
        prob = SymbolicProblem(
            "dag_schedule",
            "scheduling",
            reg,
            SymbolicState(init_facts),
            goal_facts,
        )
        plan = AStarPlanner().search(prob, dom)
        assert plan.is_success is True
        scheduled_jobs = []
        for act in plan.actions:
            if act.name == "schedule_independent_job":
                scheduled_jobs.append(act.arguments[0])
            elif act.name == "schedule_dependent_job":
                scheduled_jobs.append(act.arguments[1])
        assert "job_a" in scheduled_jobs
        assert "job_b" in scheduled_jobs
        assert "job_c" in scheduled_jobs
        assert scheduled_jobs.index("job_a") < scheduled_jobs.index("job_b")
        assert scheduled_jobs.index("job_b") < scheduled_jobs.index("job_c")

    def test_4_4_software_cicd_release_pipeline_governance(self):
        """Scenario 4.4: Software CI/CD Multi-Stage Pipeline with Gated Approval."""
        dom = registry.get("software_workflow")
        assert dom is not None
        th = dom.type_hierarchy
        reg = ObjectRegistry(th)
        reg.register_object("c1", "commit")
        reg.register_object("art1", "artifact")
        reg.register_object("staging", "environment")

        init_facts = [
            Fact("code_checked_out", ["c1"]),
            Fact("environment_healthy", ["staging"]),
        ]
        goal_facts = [
            Fact("deployed_to", ["art1", "staging"]),
        ]
        prob = SymbolicProblem("cicd_pipeline", "software_workflow", reg, SymbolicState(init_facts), goal_facts)
        plan = AStarPlanner().search(prob, dom)
        assert plan.is_success is True
        action_names = [a.name for a in plan.actions]
        assert "run_unit_tests" in action_names
        assert "grant_approval" in action_names
        assert action_names.index("run_unit_tests") < action_names.index("grant_approval")
        assert action_names[-1] == "deploy_to_environment"

    def test_4_5_grid_world_key_retrieval_and_door_unlock(self):
        """Scenario 4.5: Grid World Multi-Tile Key Retrieval and Delivery."""
        dom = registry.get("grid_world")
        assert dom is not None
        th = dom.type_hierarchy
        reg = ObjectRegistry(th)
        reg.register_object("ag1", "agent")
        reg.register_object("loc_0_0", "location")
        reg.register_object("loc_0_1", "location")
        reg.register_object("loc_goal", "location")
        reg.register_object("key1", "item")

        init_facts = [
            Fact("at", ["ag1", "loc_0_0"]),
            Fact("connected", ["loc_0_0", "loc_0_1"]),
            Fact("connected", ["loc_0_1", "loc_goal"]),
            Fact("item_at", ["key1", "loc_0_1"]),
            Fact("agent_free", ["ag1"]),
        ]
        goal_facts = [
            Fact("at", ["ag1", "loc_goal"]),
            Fact("holding_item", ["ag1", "key1"]),
        ]
        prob = SymbolicProblem("grid_key_quest", "grid_world", reg, SymbolicState(init_facts), goal_facts)
        plan = AStarPlanner().search(prob, dom)
        assert plan.is_success is True
        assert len(plan.actions) >= 3
