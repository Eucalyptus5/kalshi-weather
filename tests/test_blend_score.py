import ast
import inspect
from decimal import Decimal

import pytest

from bot.forecast import blend_score
from bot.forecast.blend import DISCOVERY, BlendWeights, freeze_digest
from bot.forecast.blend_score import blend_probability


ALPHA = "ecmwf_ifs025"
BETA = "icon_global"


def weights_of(mapping: dict[str, str], event_days: int = 12) -> BlendWeights:
    weights = {member: Decimal(value) for member, value in mapping.items()}
    members = tuple(sorted(weights))
    payload = {
        "members": list(members),
        "weights": {member: str(weights[member]) for member in members},
        "fitted_on_event_days": event_days,
        "fitted_on_split": DISCOVERY,
    }
    return BlendWeights(
        weights=weights,
        members=members,
        fitted_on_event_days=event_days,
        fitted_on_split=DISCOVERY,
        sha256=freeze_digest(payload),
    )


@pytest.fixture
def even_weights() -> BlendWeights:
    return weights_of({ALPHA: "0.5", BETA: "0.5"})


def test_weights_are_a_required_leading_positional():
    signature = inspect.signature(blend_probability, eval_str=True)
    parameters = list(signature.parameters.values())
    first = parameters[0]
    assert first.name == "weights"
    assert first.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert first.default is inspect.Parameter.empty
    assert first.annotation is BlendWeights
    assert parameters[1].name == "per_class"


def test_the_scorer_cannot_reach_the_fit():
    tree = ast.parse(inspect.getsource(blend_score))
    from_blend: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "bot.forecast.blend":
                from_blend.extend(alias.name for alias in node.names)
            if node.module == "bot.forecast":
                assert all(alias.name != "blend" for alias in node.names)
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("bot.forecast.blend") for alias in node.names)
    assert from_blend == ["BlendWeights"]


def test_an_even_blend_of_two_members(even_weights):
    result = blend_probability(even_weights, {ALPHA: Decimal("0.4"), BETA: Decimal("0.6")})
    assert isinstance(result, Decimal)
    assert result == Decimal("0.5")


def test_the_blend_is_order_independent(even_weights):
    forward = blend_probability(even_weights, {ALPHA: Decimal("0.31"), BETA: Decimal("0.77")})
    backward = blend_probability(even_weights, {BETA: Decimal("0.77"), ALPHA: Decimal("0.31")})
    assert forward == backward


def test_a_lopsided_blend(even_weights):
    weights = weights_of({ALPHA: "0.25", BETA: "0.75"})
    result = blend_probability(weights, {ALPHA: Decimal("0.20"), BETA: Decimal("0.60")})
    assert result == Decimal("0.5")


def test_a_missing_member_raises(even_weights):
    with pytest.raises(ValueError) as excinfo:
        blend_probability(even_weights, {ALPHA: Decimal("0.4")})
    assert BETA in str(excinfo.value)


def test_an_unfitted_member_raises(even_weights):
    with pytest.raises(ValueError) as excinfo:
        blend_probability(
            even_weights,
            {ALPHA: Decimal("0.4"), BETA: Decimal("0.6"), "hrrr": Decimal("0.5")},
        )
    assert "hrrr" in str(excinfo.value)
