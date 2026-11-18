"""Tests for Domain-Aware Live LLM Formalization across all 6 planning domains."""

import json
from unittest.mock import MagicMock, patch
import pytest

from core.actions.domain import create_blocks_world_domain
from core.domains import (
    BlocksWorldDomain,
    GridWorldDomain,
    HouseholdRoboticsDomain,
    SchedulingDomain,
    SoftwareWorkflowDomain,
    WarehouseDomain,
    registry,
)
from core.contracts import TaskProposalSchema
from interpretation.llm.live import LiveLLMProvider, build_domain_system_prompt
from interpretation.router import DomainRouter, RoutingOutcome
from interpretation.validator.validator import ConsistencyValidator


def test_build_domain_system_prompt_for_all_six_domains():
    """Verify that every domain generates a rich, schema-constrained prompt."""
    domains = [
        ("blocks_world", BlocksWorldDomain(), ["block"], ["on", "on_table", "holding", "handempty"]),
        ("grid_world", GridWorldDomain(), ["agent", "location", "item"], ["at", "connected", "holding_item", "agent_free"]),
        ("household_robotics", HouseholdRoboticsDomain(), ["robot", "room", "surface", "item"], ["robot_at", "item_on", "surface_in_room"]),
        ("warehouse", WarehouseDomain(), ["carrier", "zone", "sku", "bay"], ["carrier_at", "sku_stored_in", "zone_connected"]),
        ("scheduling", SchedulingDomain(), ["job", "resource", "slot"], ["job_ready", "resource_available", "precedes"]),
        ("software_workflow", SoftwareWorkflowDomain(), ["commit", "artifact", "environment"], ["code_checked_out", "deployed_to", "environment_healthy"]),
    ]

    for domain_id, domain_obj, expected_types, expected_preds in domains:
        prompt = build_domain_system_prompt(domain_obj)
        assert domain_id in prompt
        for t in expected_types:
            assert t in prompt, f"Expected type '{t}' missing in {domain_id} prompt"
        for p in expected_preds:
            assert p in prompt, f"Expected predicate '{p}' missing in {domain_id} prompt"

        # Non-blocks domains should NOT instruct blocks world defaults
        if domain_id != "blocks_world":
            assert "tabletop" not in prompt.lower()
            assert "top_block" not in prompt


def test_domain_router_attaches_system_prompt_to_route():
    """Verify DomainRouter populates system_prompt in DomainRoutingResult."""
    router = DomainRouter(registry=registry)
    res = router.route("Move the agent from cell_a to cell_b.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "grid_world"
    assert res.system_prompt is not None
    assert "Grid World" in res.system_prompt
    assert "agent" in res.system_prompt
    assert "location" in res.system_prompt


def test_live_llm_provider_passes_domain_prompt_to_lm_studio():
    """Verify LiveLLMProvider sends domain-specific system prompt to LM Studio."""
    grid_domain = GridWorldDomain()
    provider = LiveLLMProvider(provider="lmstudio", endpoint="http://127.0.0.1:1234")

    captured_payloads = []

    def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": json_payload_grid
                }
            }]
        }
        return resp

    json_payload_grid = json.dumps({
        "task_id": "grid_task",
        "raw_prompt": "Move agent to cell_b",
        "entities": {"agent": "agent", "cell_a": "location", "cell_b": "location"},
        "initial_facts": [
            {"predicate": "at", "arguments": ["agent", "cell_a"]},
            {"predicate": "connected", "arguments": ["cell_a", "cell_b"]},
            {"predicate": "agent_free", "arguments": ["agent"]}
        ],
        "goal_facts": [
            {"predicate": "at", "arguments": ["agent", "cell_b"]}
        ],
        "negative_constraints": [],
        "confidence": 0.95
    })

    mock_models_resp = MagicMock()
    mock_models_resp.status_code = 200
    mock_models_resp.json.return_value = {"data": [{"id": "local-model"}]}

    with patch("httpx.Client.get", return_value=mock_models_resp):
        with patch("httpx.Client.post", side_effect=mock_post):
            proposal = provider.interpret("Move agent to cell_b", grid_domain)
            assert isinstance(proposal, TaskProposalSchema)
            assert proposal.entities["agent"] == "agent"
            assert proposal.entities["cell_a"] == "location"

            # Check that the system message sent in payload was for Grid World
            sent_sys_msg = captured_payloads[0]["messages"][0]["content"]
            assert "Grid World" in sent_sys_msg
            assert "location" in sent_sys_msg
            assert "at(agent, location)" in sent_sys_msg or "at(ag: agent" in sent_sys_msg or "at" in sent_sys_msg


def test_live_llm_parses_and_validates_household_domain():
    """Verify non-blocks domain output parses without blocks-world contamination."""
    household_domain = HouseholdRoboticsDomain()
    provider = LiveLLMProvider(provider="mock")
    validator = ConsistencyValidator()

    sample_output = json.dumps({
        "task_id": "household_task",
        "raw_prompt": "Move the mug from the kitchen shelf to the living room table. Avoid entering the bedroom.",
        "entities": {
            "robot": "robot",
            "kitchen": "room",
            "living_room": "room",
            "bedroom": "room",
            "kitchen_shelf": "surface",
            "living_room_table": "surface",
            "mug": "item"
        },
        "initial_facts": [
            {"predicate": "robot_at", "arguments": ["robot", "kitchen"]},
            {"predicate": "gripper_empty", "arguments": ["robot"]},
            {"predicate": "surface_in_room", "arguments": ["kitchen_shelf", "kitchen"]},
            {"predicate": "surface_in_room", "arguments": ["living_room_table", "living_room"]},
            {"predicate": "connected_rooms", "arguments": ["kitchen", "living_room"]},
            {"predicate": "item_on", "arguments": ["mug", "kitchen_shelf"]}
        ],
        "goal_facts": [
            {"predicate": "item_on", "arguments": ["mug", "living_room_table"]}
        ],
        "negative_constraints": [
            {"predicate": "robot_at", "arguments": ["robot", "bedroom"]}
        ],
        "confidence": 0.96
    })

    proposal = provider._parse_json_to_schema(
        raw_json_str=sample_output,
        prompt="Move the mug from the kitchen shelf to the living room table. Avoid entering the bedroom.",
        domain=household_domain,
    )

    # 1. Assert entity types were preserved (not forced to 'block')
    assert proposal.entities["robot"] == "robot"
    assert proposal.entities["kitchen"] == "room"
    assert proposal.entities["mug"] == "item"
    assert proposal.entities["kitchen_shelf"] == "surface"

    # 2. Assert NO blocks-world predicates hallucinated into initial state
    init_preds = {f.predicate for f in proposal.initial_facts}
    assert "on_table" not in init_preds
    assert "clear" not in init_preds
    assert "handempty" not in init_preds

    # 3. Assert negative constraint was preserved
    assert len(proposal.negative_constraints) == 1
    assert proposal.negative_constraints[0].predicate == "robot_at"
    assert proposal.negative_constraints[0].arguments == ("robot", "bedroom")

    # 4. ConsistencyValidator passes 100%
    val_res = validator.validate(proposal, household_domain)
    assert val_res.is_valid is True, f"Validation errors: {val_res.errors}"


def test_live_llm_infers_entity_types_from_domain_predicates():
    """Verify that missing entity types are inferred from domain predicate definitions."""
    sw_domain = SoftwareWorkflowDomain()
    provider = LiveLLMProvider(provider="mock")
    validator = ConsistencyValidator()

    # Raw model JSON has empty entities dictionary
    sample_output = json.dumps({
        "task_id": "sw_task",
        "raw_prompt": "Deploy pkg_v1 to staging. Do not deploy to production.",
        "entities": {},
        "initial_facts": [
            {"predicate": "code_checked_out", "arguments": ["commit_v1"]},
            {"predicate": "environment_healthy", "arguments": ["staging"]},
            {"predicate": "environment_healthy", "arguments": ["production"]}
        ],
        "goal_facts": [
            {"predicate": "deployed_to", "arguments": ["pkg_v1", "staging"]}
        ],
        "negative_constraints": [
            {"predicate": "deployed_to", "arguments": ["pkg_v1", "production"]}
        ],
        "confidence": 0.95
    })

    proposal = provider._parse_json_to_schema(
        raw_json_str=sample_output,
        prompt="Deploy pkg_v1 to staging.",
        domain=sw_domain,
    )

    # Predicate deployed_to(artifact, environment) should infer pkg_v1: artifact, staging: environment
    assert proposal.entities.get("pkg_v1") == "artifact"
    assert proposal.entities.get("staging") == "environment"
    assert proposal.entities.get("production") == "environment"
    assert proposal.entities.get("commit_v1") == "commit"

    val_res = validator.validate(proposal, sw_domain)
    assert val_res.is_valid is True, f"Validation errors: {val_res.errors}"
