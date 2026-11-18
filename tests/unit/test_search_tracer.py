"""Unit tests for SearchTracer, UCSPlanner, search trace API endpoint, and GraphEdge transition details."""

import math
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from core.actions.domain import create_blocks_world_domain
from core.actions.instantiation import GroundAction
from core.contracts import (
    FactSchema,
    SearchEventType,
    SearchTraceEvent,
    SearchTraceRequest,
    SearchTraceResponse,
)
from core.domains import registry
from core.graph.enumerator import CompleteReachableGraphEnumerator
from core.graph.models import GraphEdge, ReachableStateGraph
from core.search.algorithms import (
    AStarPlanner,
    BestFirstPlanner,
    BFSPlanner,
    UCSPlanner,
    get_planner_by_name,
)
from core.search.heuristics import ZeroHeuristic
from core.search.node import SearchNode
from core.search.tracer import SearchTracer
from core.world.predicates import Fact
from core.world.problem import SymbolicProblem
from core.world.state import SymbolicState
from core.world.types import ObjectRegistry


@pytest.fixture
def bw_domain():
    return registry.get("blocks_world")


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def create_simple_problem(domain):
    reg = ObjectRegistry(domain.type_hierarchy)
    reg.register_object("a", "block")
    reg.register_object("b", "block")
    init_facts = [
        Fact("on_table", ["a"]),
        Fact("on_table", ["b"]),
        Fact("clear", ["a"]),
        Fact("clear", ["b"]),
        Fact("handempty", []),
    ]
    goal_facts = [Fact("on", ["a", "b"])]
    return SymbolicProblem(
        name="test_prob",
        domain_name=domain.domain_id,
        registry=reg,
        initial_state=SymbolicState(init_facts, objects=reg.get_all_objects()),
        goal_conditions=goal_facts,
    )


# ==============================================================================
# 1. UCS Registration and Optimality Tests
# ==============================================================================

class TestUCSPlanner:
    def test_ucs_registration_aliases(self):
        for alias in ("UCS", "UNIFORM_COST", "UNIFORM-COST", "uniform_cost", "dijkstra"):
            planner = get_planner_by_name(alias)
            assert isinstance(planner, UCSPlanner)
            assert planner.name == "UCS"
            assert "g(n)" in planner.selection_rule

    def test_ucs_optimal_cost(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        planner = UCSPlanner()
        plan = planner.search(prob, bw_domain)
        assert plan.is_success is True
        assert plan.is_optimal is True
        assert plan.total_cost == 2.0
        assert len(plan.actions) == 2
        assert plan.actions[0].name == "pick_up"
        assert plan.actions[1].name == "stack"

    def test_ucs_equivalence_to_astar_zero_heuristic(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        ucs_plan = UCSPlanner().search(prob, bw_domain)
        astar_plan = AStarPlanner().search(prob, bw_domain, heuristic=ZeroHeuristic())
        assert ucs_plan.total_cost == astar_plan.total_cost
        assert [a.name for a in ucs_plan.actions] == [a.name for a in astar_plan.actions]


# ==============================================================================
# 2. SearchTracer Instrumentation Tests
# ==============================================================================

class TestSearchTracer:
    def test_tracer_bfs_event_sequence(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        planner = BFSPlanner()
        tracer = SearchTracer(bw_domain, planner.selection_rule)
        plan = planner.search(prob, bw_domain, tracer=tracer)

        assert plan.is_success is True
        assert len(tracer.events) >= 4
        # Verify sequence numbers are continuous and 1-indexed
        for idx, ev in enumerate(tracer.events):
            assert ev.seq == idx + 1
            assert len(ev.state_hash) == 64
            assert ev.state_hash == ev.state_hash.lower()

        # Check event types emitted
        event_types = {e.event_type for e in tracer.events}
        assert SearchEventType.INITIAL in event_types
        assert SearchEventType.EXPAND in event_types
        assert SearchEventType.GENERATE in event_types
        assert SearchEventType.GOAL_DETECTED in event_types
        assert SearchEventType.SOLUTION_RECONSTRUCTED in event_types

    def test_tracer_astar_event_sequence(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        planner = AStarPlanner()
        tracer = SearchTracer(bw_domain, planner.selection_rule)
        plan = planner.search(prob, bw_domain, tracer=tracer)

        assert plan.is_success is True
        assert len(tracer.events) >= 4
        assert len(tracer.solution_path_node_ids) >= 2
        assert len(tracer.solution_path_edge_ids) == len(plan.actions)

    def test_tracer_ucs_event_sequence(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        planner = UCSPlanner()
        tracer = SearchTracer(bw_domain, planner.selection_rule)
        plan = planner.search(prob, bw_domain, tracer=tracer)

        assert plan.is_success is True
        assert tracer.is_goal_found is True
        assert len(tracer.events) > 0

    def test_tracer_best_first_event_sequence(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        planner = BestFirstPlanner()
        tracer = SearchTracer(bw_domain, planner.selection_rule)
        plan = planner.search(prob, bw_domain, tracer=tracer)

        assert plan.is_success is True
        assert len(tracer.events) >= 4

    def test_frontier_snapshot_bounding(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        tracer = SearchTracer(bw_domain, "test_rule", max_snapshot_size=2)
        node1 = SearchNode(state=prob.initial_state)
        snap = tracer.snapshot_deque([node1] * 10)
        assert len(snap) == 2


# ==============================================================================
# 3. POST /api/search/trace Endpoint Tests
# ==============================================================================

class TestSearchTraceAPI:
    def test_post_search_trace_bfs(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "BFS",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["algorithm"] == "BFS"
        assert data["is_goal_found"] is True
        assert data["total_events"] > 0
        assert len(data["solution_path_node_ids"]) >= 2

    def test_post_search_trace_astar(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "A*",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["algorithm"] == "A*"
        assert data["selection_rule"] == "f(n) = g(n) + h(n)"
        assert data["is_goal_found"] is True

    def test_post_search_trace_ucs(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "prompt": "Stack the red box on the blue box.",
            "algorithm": "UCS",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["algorithm"] == "UCS"
        assert "g(n)" in data["selection_rule"]
        assert data["is_goal_found"] is True

    def test_post_search_trace_unknown_domain_404(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "nonexistent_domain",
            "algorithm": "A*",
        })
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_post_search_trace_unknown_algo_422_or_400(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "algorithm": "GENETIC_ALGO",
        })
        assert res.status_code in (400, 422)

    def test_post_search_trace_budget_validation(self, client):
        res = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "max_nodes_expanded": 0,
        })
        assert res.status_code == 422

        res2 = client.post("/api/search/trace", json={
            "domain_id": "blocks_world",
            "timeout_seconds": -5.0,
        })
        assert res2.status_code == 422

    def test_post_search_trace_concurrency_429(self, client):
        with patch("app.api.routes.graph_concurrency_manager.acquire", return_value=False):
            res = client.post("/api/search/trace", json={
                "domain_id": "blocks_world",
                "algorithm": "A*",
            })
            assert res.status_code == 429
            assert "capacity exceeded" in res.json()["detail"].lower()


# ==============================================================================
# 4. GraphEdge Transition Details Tests
# ==============================================================================

class TestGraphEdgeTransitionDetails:
    def test_graph_edge_details_construction(self):
        edge = GraphEdge(
            edge_id="e1",
            source_id="s0",
            target_id="s1",
            action_name="pick_up",
            action_args=["b1"],
            cost=1.0,
            preconditions=["clear(b1)", "on_table(b1)", "handempty()"],
            add_effects=["holding(b1)"],
            del_effects=["clear(b1)", "on_table(b1)", "handempty()"],
            invariants_satisfied=True,
        )
        assert edge.preconditions == ["clear(b1)", "on_table(b1)", "handempty()"]
        assert edge.add_effects == ["holding(b1)"]
        assert edge.del_effects == ["clear(b1)", "on_table(b1)", "handempty()"]
        assert edge.invariants_satisfied is True
        assert edge.action_label == "pick_up(b1)"

    def test_graph_edge_from_ground_action(self):
        action = GroundAction(
            name="pick_up",
            arguments=["b1"],
            preconditions=[Fact("clear", ["b1"]), Fact("on_table", ["b1"])],
            add_effects=[Fact("holding", ["b1"])],
            del_effects=[Fact("clear", ["b1"])],
            cost=1.5,
        )
        edge = GraphEdge.from_ground_action(
            edge_id="e_test",
            source_id="src_hash",
            target_id="dst_hash",
            action=action,
            invariants_satisfied=True,
        )
        assert edge.edge_id == "e_test"
        assert edge.source_id == "src_hash"
        assert edge.target_id == "dst_hash"
        assert edge.action_name == "pick_up"
        assert edge.action_args == ["b1"]
        assert edge.cost == 1.5
        assert edge.preconditions == ["clear(b1)", "on_table(b1)"]
        assert edge.add_effects == ["holding(b1)"]
        assert edge.del_effects == ["clear(b1)"]
        assert edge.invariants_satisfied is True

    def test_ground_action_string_properties(self):
        action = GroundAction(
            name="stack",
            arguments=["b1", "b2"],
            preconditions=[Fact("holding", ["b1"]), Fact("clear", ["b2"])],
            add_effects=[Fact("on", ["b1", "b2"])],
            del_effects=[Fact("holding", ["b1"])],
        )
        assert action.precondition_strings == ["holding(b1)", "clear(b2)"]
        assert action.add_effect_strings == ["on(b1, b2)"]
        assert action.del_effect_strings == ["holding(b1)"]

    def test_complete_reachable_graph_enumerator_populates_edge_details(self, bw_domain):
        prob = create_simple_problem(bw_domain)
        enumerator = CompleteReachableGraphEnumerator(max_nodes=10, max_edges=20)
        graph = enumerator.enumerate_graph(prob, bw_domain)

        assert len(graph.edges) > 0
        for edge in graph.edges:
            assert isinstance(edge.preconditions, list)
            assert isinstance(edge.add_effects, list)
            assert isinstance(edge.del_effects, list)
            assert edge.invariants_satisfied is True
            assert len(edge.preconditions) > 0
            assert len(edge.add_effects) > 0
            assert len(edge.del_effects) > 0

        # Cytoscape export validation
        cy = graph.to_cytoscape()
        cy_edges = [el for el in cy["elements"] if "source" in el["data"]]
        assert len(cy_edges) > 0
        for ce in cy_edges:
            d = ce["data"]
            assert "preconditions" in d
            assert "add_effects" in d
            assert "del_effects" in d
            assert "invariants_satisfied" in d
            assert d["invariants_satisfied"] is True
