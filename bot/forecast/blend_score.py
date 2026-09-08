from collections.abc import Mapping
from decimal import Decimal

from bot.forecast.blend import BlendWeights
from bot.lag.fee_floor import BAR_CONTEXT


def blend_probability(weights: BlendWeights, per_class: Mapping[str, Decimal]) -> Decimal:
    difference = set(per_class) ^ set(weights.members)
    if difference:
        raise ValueError(f"per_class members differ from the fitted members: {sorted(difference)}")
    total = Decimal(0)
    for member in weights.members:
        total = BAR_CONTEXT.add(
            total, BAR_CONTEXT.multiply(weights.weights[member], per_class[member])
        )
    return total
