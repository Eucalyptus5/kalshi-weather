from __future__ import annotations

from decimal import Decimal

from tests.fixtures.partial_fill_paper_trade_row import make_partial_fill_intent_and_trade


def test_overlay_uses_filled_not_attempted_contracts() -> None:
    intent, trade, book = make_partial_fill_intent_and_trade()
    cost_per_contract = Decimal("1") - book.yes_bid

    buggy = cost_per_contract * Decimal(intent.contracts)
    correct = cost_per_contract * Decimal(trade.contracts)

    assert buggy == Decimal("71.94")
    assert correct == Decimal("0.01")
    assert buggy - correct == Decimal("71.93")
