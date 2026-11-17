"""Unit tests for the Text-First NLP Domain Router and Clarification Engine."""

import pytest
from core.domains import registry
from interpretation.router import DomainRouter, RoutingOutcome, DomainRoutingResult


def test_router_selects_blocks_world():
    router = DomainRouter(registry=registry)
    res = router.route("Stack the red box on the blue box. Do not move the glass.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "blocks_world"
    assert "glass" in res.entities
    assert len(res.negative_constraints) >= 1


def test_router_selects_grid_world():
    router = DomainRouter(registry=registry)
    res = router.route("Move the agent from cell_a to cell_b.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "grid_world"


def test_router_selects_household():
    router = DomainRouter(registry=registry)
    res = router.route("Move the mug from the kitchen shelf to the living room table.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "household_robotics"


def test_router_selects_warehouse():
    router = DomainRouter(registry=registry)
    res = router.route("Move SKU_A from shelf_1 to dispatch dock.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "warehouse"


def test_router_selects_scheduling():
    router = DomainRouter(registry=registry)
    res = router.route("Schedule design_review before code_review on machine_1.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "scheduling"


def test_router_selects_software_workflow():
    router = DomainRouter(registry=registry)
    res = router.route("Build and test commit_v1, then deploy to staging environment.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "software_workflow"


def test_router_detects_unsupported_task():
    router = DomainRouter(registry=registry)
    res = router.route("Book a flight ticket from Paris to New York and reserve a hotel room.")
    assert res.outcome == RoutingOutcome.UNSUPPORTED
    assert res.selected_domain_id is None
    assert "unsupported" in res.explanation.lower()


def test_router_detects_ambiguity_needs_clarification():
    router = DomainRouter(registry=registry)
    # Ambiguous / missing initial location
    res = router.route("Deliver the package to the destination.")
    assert res.outcome == RoutingOutcome.NEEDS_CLARIFICATION
    assert res.clarification_question is not None
    assert len(res.clarification_question) > 5


def test_router_explicit_domain_override():
    router = DomainRouter(registry=registry)
    # Explicit domain override should be respected directly
    res = router.route("Move item to destination.", explicit_domain_id="warehouse")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert res.selected_domain_id == "warehouse"


def test_router_negative_constraint_preserves_semantics():
    router = DomainRouter(registry=registry)
    res = router.route("Move agent from cell_a to cell_b. Do not enter cell_c.")
    assert res.outcome == RoutingOutcome.SUPPORTED
    assert len(res.negative_constraints) >= 1
    # For grid world, the negative constraint should target location, not holding
    assert res.negative_constraints[0].predicate == "at"
    assert "cell_c" in res.negative_constraints[0].arguments


@pytest.mark.parametrize(
    "prompt",
    [
        "The red box is on the green box. Put the red box on the table. Do not touch the glass or move the glass.",
        "The red box is on the green box. Put the red box on the table. Do not touch or move the glass.",
    ],
)
def test_router_extracts_glass_constraint_from_coordinated_verbs(prompt):
    router = DomainRouter(registry=registry)
    res = router.route(prompt, explicit_domain_id="blocks_world")

    assert res.outcome == RoutingOutcome.SUPPORTED
    assert any(f.predicate == "holding" and f.arguments == ("glass",) for f in res.negative_constraints)
    assert all("or" not in f.arguments and "and" not in f.arguments for f in res.negative_constraints)
    assert any(c.entity == "glass" for c in res.typed_constraints)


def test_router_resolves_multiword_prohibited_block_name():
    router = DomainRouter(registry=registry)
    res = router.route(
        "Put the blue box on the table. Do not move the red box.",
        explicit_domain_id="blocks_world",
    )

    assert any(f.predicate == "holding" and f.arguments == ("red_box",) for f in res.negative_constraints)
    assert any(c.entity == "red_box" for c in res.typed_constraints)
