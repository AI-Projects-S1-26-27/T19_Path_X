"""Authoritative Search Event Tracing and Frontier Recording Engine."""

from __future__ import annotations

import collections
import heapq
import itertools
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional
from pydantic import BaseModel, Field

from core.actions.domain import Domain
from core.actions.instantiation import GroundAction
from core.contracts import SearchEventType, SearchTraceEvent
from core.search.node import SearchNode
from core.world.predicates import Fact
from core.world.problem import SymbolicProblem
from core.world.state import SymbolicState

__all__ = [
    "SearchEventType",
    "SearchTraceEvent",
    "SearchTracer",
]


class SearchTracer:
    """Non-intrusive tracer recording authentic backend search events."""

    def __init__(
        self,
        domain: Domain,
        selection_rule: str,
        max_snapshot_size: int = 25,
        algorithm: Optional[str] = None,
        problem: Optional[SymbolicProblem] = None,
    ) -> None:
        self.domain = domain
        self.selection_rule = selection_rule
        self.max_snapshot_size = max_snapshot_size
        self.algorithm = algorithm
        self.problem = problem
        self.events: List[SearchTraceEvent] = []
        self.solution_path_node_ids: List[str] = []
        self.solution_path_edge_ids: List[str] = []
        self.nodes_expanded: int = 0
        self.nodes_generated: int = 0
        self.is_goal_found: bool = False
        self._seq: int = 0
        self._hash_cache: Dict[FrozenSet[Fact], str] = {}

    def get_state_hash(self, state: SymbolicState) -> str:
        h = self._hash_cache.get(state.facts)
        if h is None:
            h = self.domain.canonical_state_key(state)
            self._hash_cache[state.facts] = h
        return h

    def snapshot_node(self, node: SearchNode) -> Dict[str, Any]:
        h = self.get_state_hash(node.state)
        act_str = node.action.to_string() if node.action else None
        return {
            "node_id": h,
            "state_id": h,
            "state_hash": h,
            "depth": node.depth,
            "g_cost": round(node.g_cost, 2),
            "h_cost": round(node.h_cost, 2),
            "f_cost": round(node.f_cost, 2),
            "action_from_parent": act_str,
            "action_label": act_str,
        }

    def snapshot_deque(self, queue: collections.deque[SearchNode]) -> List[Dict[str, Any]]:
        return [self.snapshot_node(n) for n in itertools.islice(queue, self.max_snapshot_size)]

    def snapshot_best_first_heap(self, frontier: List[Any]) -> List[Dict[str, Any]]:
        top = heapq.nsmallest(self.max_snapshot_size, frontier)
        return [self.snapshot_node(item[3]) for item in top]

    def snapshot_astar_heap(self, frontier: List[SearchNode]) -> List[Dict[str, Any]]:
        top = heapq.nsmallest(self.max_snapshot_size, frontier)
        return [self.snapshot_node(n) for n in top]

    def record_initial(
        self,
        node: SearchNode,
        frontier_size: int,
        frontier_snapshot: List[Dict[str, Any]],
        is_goal: bool,
    ) -> None:
        self._seq += 1
        self.nodes_generated += 1
        s_hash = self.get_state_hash(node.state)
        node.state_hash = s_hash
        self.events.append(
            SearchTraceEvent(
                seq=self._seq,
                event_type="INITIAL",
                state_id=s_hash,
                state_hash=s_hash,
                parent_id=None,
                action_name=None,
                action_args=[],
                depth=node.depth,
                g_cost=node.g_cost,
                h_cost=node.h_cost,
                f_cost=node.f_cost,
                selection_rule=self.selection_rule,
                frontier_size=frontier_size,
                frontier_snapshot=frontier_snapshot,
                is_goal=is_goal,
                message=f"Initial state S0 initialized. Placed on frontier (g={node.g_cost:.1f}, h={node.h_cost:.1f}, f={node.f_cost:.1f}).",
            )
        )

    def record_expand(
        self,
        node: SearchNode,
        frontier_size: int,
        frontier_snapshot: List[Dict[str, Any]],
        is_goal: bool,
    ) -> None:
        self._seq += 1
        self.nodes_expanded += 1
        s_hash = self.get_state_hash(node.state)
        node.state_hash = s_hash
        p_hash = self.get_state_hash(node.parent.state) if node.parent else None
        act_name = node.action.name if node.action else None
        act_args = list(node.action.arguments) if node.action else []
        self.events.append(
            SearchTraceEvent(
                seq=self._seq,
                event_type="EXPAND",
                state_id=s_hash,
                state_hash=s_hash,
                parent_id=p_hash,
                action_name=act_name,
                action_args=act_args,
                depth=node.depth,
                g_cost=node.g_cost,
                h_cost=node.h_cost,
                f_cost=node.f_cost,
                selection_rule=self.selection_rule,
                frontier_size=frontier_size,
                frontier_snapshot=frontier_snapshot,
                is_goal=is_goal,
                message=f"Expanding state at depth {node.depth} (g={node.g_cost:.1f}, h={node.h_cost:.1f}, f={node.f_cost:.1f}). Selection rule: {self.selection_rule}.",
            )
        )

    def record_generate(
        self,
        node: SearchNode,
        parent: SearchNode,
        action: GroundAction,
        frontier_size: int,
        frontier_snapshot: List[Dict[str, Any]],
        is_goal: bool,
    ) -> None:
        self._seq += 1
        self.nodes_generated += 1
        s_hash = self.get_state_hash(node.state)
        node.state_hash = s_hash
        p_hash = self.get_state_hash(parent.state)
        self.events.append(
            SearchTraceEvent(
                seq=self._seq,
                event_type="GENERATE",
                state_id=s_hash,
                state_hash=s_hash,
                parent_id=p_hash,
                action_name=action.name,
                action_args=list(action.arguments),
                depth=node.depth,
                g_cost=node.g_cost,
                h_cost=node.h_cost,
                f_cost=node.f_cost,
                selection_rule=self.selection_rule,
                frontier_size=frontier_size,
                frontier_snapshot=frontier_snapshot,
                is_goal=is_goal,
                message=f"Generated successor via {action.to_string()} (step cost {action.cost:.1f}, new g={node.g_cost:.1f}, h={node.h_cost:.1f}, f={node.f_cost:.1f}). Enqueued to frontier.",
            )
        )

    def record_skip_pruned(
        self,
        state: Optional[SymbolicState],
        parent: Optional[SearchNode],
        action: Optional[GroundAction],
        reason: str,
        frontier_size: int,
        g_cost: float = 0.0,
        h_cost: float = 0.0,
        f_cost: float = 0.0,
    ) -> None:
        self._seq += 1
        s_hash = self.get_state_hash(state) if state else (parent.state_hash if parent and parent.state_hash else "unknown")
        p_hash = self.get_state_hash(parent.state) if parent else None
        act_name = action.name if action else None
        act_args = list(action.arguments) if action else []
        self.events.append(
            SearchTraceEvent(
                seq=self._seq,
                event_type="SKIP_PRUNED",
                state_id=s_hash,
                state_hash=s_hash,
                parent_id=p_hash,
                action_name=act_name,
                action_args=act_args,
                depth=(parent.depth + 1) if parent else 0,
                g_cost=g_cost,
                h_cost=h_cost,
                f_cost=f_cost,
                selection_rule=self.selection_rule,
                frontier_size=frontier_size,
                frontier_snapshot=[],
                is_goal=False,
                message=f"Pruned candidate transition via {action.to_string() if action else 'unknown'}: {reason}.",
            )
        )

    def record_goal_detected(
        self,
        node: SearchNode,
        frontier_size: int,
    ) -> None:
        self._seq += 1
        self.is_goal_found = True
        s_hash = self.get_state_hash(node.state)
        p_hash = self.get_state_hash(node.parent.state) if node.parent else None
        act_name = node.action.name if node.action else None
        act_args = list(node.action.arguments) if node.action else []
        self.events.append(
            SearchTraceEvent(
                seq=self._seq,
                event_type="GOAL_DETECTED",
                state_id=s_hash,
                state_hash=s_hash,
                parent_id=p_hash,
                action_name=act_name,
                action_args=act_args,
                depth=node.depth,
                g_cost=node.g_cost,
                h_cost=node.h_cost,
                f_cost=node.f_cost,
                selection_rule=self.selection_rule,
                frontier_size=frontier_size,
                frontier_snapshot=[],
                is_goal=True,
                message=f"Goal detected at state depth {node.depth}! All goal conditions satisfied with total cost {node.g_cost:.1f}.",
            )
        )

    def record_solution_reconstructed(
        self,
        goal_node: SearchNode,
        plan_actions: List[GroundAction],
    ) -> None:
        self._seq += 1
        nodes_rev: List[str] = []
        edges_rev: List[str] = []
        curr: Optional[SearchNode] = goal_node
        while curr is not None:
            curr_hash = self.get_state_hash(curr.state)
            nodes_rev.append(curr_hash)
            if curr.parent is not None and curr.action is not None:
                parent_hash = self.get_state_hash(curr.parent.state)
                edge_sig = f"{parent_hash}::{curr.action.to_string()}::{curr_hash}"
                edges_rev.append(edge_sig)
            curr = curr.parent

        nodes_rev.reverse()
        edges_rev.reverse()
        self.solution_path_node_ids = nodes_rev
        self.solution_path_edge_ids = edges_rev

        goal_hash = self.get_state_hash(goal_node.state)
        self.events.append(
            SearchTraceEvent(
                seq=self._seq,
                event_type="SOLUTION_RECONSTRUCTED",
                state_id=goal_hash,
                state_hash=goal_hash,
                parent_id=self.get_state_hash(goal_node.parent.state) if goal_node.parent else None,
                action_name=goal_node.action.name if goal_node.action else None,
                action_args=list(goal_node.action.arguments) if goal_node.action else [],
                depth=goal_node.depth,
                g_cost=goal_node.g_cost,
                h_cost=goal_node.h_cost,
                f_cost=goal_node.f_cost,
                selection_rule=self.selection_rule,
                frontier_size=0,
                frontier_snapshot=[],
                is_goal=True,
                message=f"Solution path reconstructed: {len(nodes_rev)} states, {len(plan_actions)} actions, total cost {goal_node.g_cost:.1f}.",
            )
        )
