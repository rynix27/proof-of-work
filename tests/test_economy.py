"""Ledger: loan lifecycle, interest math, defaults, and the conservation invariant."""
import random

import pytest

from core.agents import fresh_roster
from core.economy import Ledger, LedgerError, compute_interest
from core.simconfig import SimulationConfig


def make(n=3, **cfg):
    agents = fresh_roster(n)
    return agents, Ledger(agents, SimulationConfig.defaults(**cfg))


def test_interest_is_ceil_in_integer_math():
    assert compute_interest(20, 1500) == 3      # 20 @ 15% -> 3.0, no float error
    assert compute_interest(5, 1500) == 1       # 0.75 -> ceil 1
    assert compute_interest(3, 1500) == 1       # 0.45 -> ceil 1
    assert compute_interest(20, 0) == 0


def test_loan_total_owed_and_balances_move():
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, round_num=1)
    assert (loan.interest, loan.total_owed, loan.outstanding) == (3, 23, 23)
    assert (a.balance, b.balance) == (80, 120)
    assert loan.due_round == 1 + L.cfg.loan_term_rounds


def test_full_repayment_with_interest_closes_loan_and_lender_profits():
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, 1)
    L.repay(loan, 23, 3)
    assert loan.status == "repaid" and loan.outstanding == 0
    assert (a.balance, b.balance) == (103, 97)
    assert L.reconcile()["ok"]


def test_partial_repayment_then_remainder():
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, 1)
    L.repay(loan, 10, 2)
    assert loan.status == "active" and loan.outstanding == 13 and loan.repaid == 10
    L.repay(loan, 13, 3)
    assert loan.status == "repaid"


def test_cannot_repay_a_closed_loan_twice():
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 5, 1)
    L.repay(loan, loan.total_owed, 2)
    before = (a.balance, b.balance)
    with pytest.raises(LedgerError) as e:
        L.repay(loan, 1, 3)
    assert e.value.code == "loan_not_active"
    assert (a.balance, b.balance) == before


def test_overpayment_and_unaffordable_repayment_rejected():
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, 1)
    with pytest.raises(LedgerError) as e:
        L.repay(loan, 24, 2)
    assert e.value.code == "over_repayment"
    b.balance = 2
    with pytest.raises(LedgerError) as e:
        L.repay(loan, 5, 2)
    assert e.value.code == "insufficient_funds"


def test_due_loan_collects_what_it_can_then_defaults_remainder():
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, 1)
    b.balance = 8                              # borrower can only pay part
    L.settle_due_loans(round_num=loan.due_round)
    assert loan.status == "defaulted" and loan.repaid == 8 and loan.written_off == 15
    assert b.balance == 0
    assert L.outstanding_debt() == 0


def test_loan_not_collected_before_due_round():
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, 1)
    L.settle_due_loans(round_num=loan.due_round - 1)
    assert loan.status == "active" and loan.repaid == 0


def test_bankruptcy_defaults_all_of_borrowers_loans_immediately():
    (a, b, c), L = make()
    l1, l2 = L.create_loan(a, c, 10, 1), L.create_loan(b, c, 5, 1)
    L.default_loans_of(c, round_num=2)
    assert l1.status == l2.status == "defaulted"
    assert l1.default_round == 2


def test_bankrupt_borrower_can_be_revived_and_old_debt_stays_written_off():
    (a, b, _), L = make()
    old = L.create_loan(a, b, 10, 1)
    b.balance = 0
    b.go_bankrupt()
    L.default_loans_of(b, 2)
    new = L.create_loan(a, b, 15, 3)
    assert not b.bankrupt and b.balance == 15
    assert old.status == "defaulted" and new.status == "active"


@pytest.mark.parametrize("amount,code", [
    (0, "bad_amount"), (-5, "bad_amount"), (2.5, "bad_amount"), ("9", "bad_amount"),
    (float("nan"), "bad_amount"), (True, "bad_amount"), (51, "over_max_loan"), (100, "over_max_loan"),
])
def test_invalid_loan_amounts_rejected_without_side_effects(amount, code):
    (a, b, _), L = make()
    with pytest.raises(LedgerError) as e:
        L.create_loan(a, b, amount, 1)
    assert e.value.code == code
    assert (a.balance, b.balance) == (100, 100) and L.loans == []


def test_lender_must_keep_reserve_and_cannot_lend_to_self_or_when_bankrupt():
    (a, b, _), L = make(max_loan_amount=500)
    with pytest.raises(LedgerError) as e:
        L.create_loan(a, b, 100, 1)                # would leave 0 < reserve 1
    assert e.value.code == "lender_reserve"
    with pytest.raises(LedgerError) as e:
        L.create_loan(a, a, 5, 1)
    assert e.value.code == "self_loan"
    a.go_bankrupt()
    with pytest.raises(LedgerError) as e:
        L.create_loan(a, b, 5, 1)
    assert e.value.code == "lender_bankrupt"


def test_borrower_active_loan_cap():
    (a, b, _), L = make(max_active_loans=2)
    L.create_loan(a, b, 5, 1)
    L.create_loan(a, b, 5, 1)
    with pytest.raises(LedgerError) as e:
        L.create_loan(a, b, 5, 1)
    assert e.value.code == "too_many_loans"


def test_charge_cannot_overdraw_and_rejects_bad_costs():
    (a, *_), L = make()
    a.balance = 3
    with pytest.raises(LedgerError):
        L.charge_compute(a, 12)
    with pytest.raises(LedgerError):
        L.charge_compute(a, -1)
    with pytest.raises(LedgerError):
        L.charge_compute(a, 1.5)
    assert a.balance == 3


@pytest.mark.parametrize("seed", range(25))
def test_conservation_invariant_holds_under_random_operations(seed):
    """sum(balances) == start + minted - burned after ANY sequence of operations."""
    rng = random.Random(seed)
    agents, L = make(5)
    for rnd in range(1, 80):
        x, y = rng.sample(agents, 2)
        op = rng.choice(["loan", "repay", "charge", "reward", "settle", "default"])
        try:
            if op == "loan":
                L.create_loan(x, y, rng.randint(1, 60), rnd)
            elif op == "repay" and L.active_loans():
                loan = rng.choice(L.active_loans())
                L.repay(loan, rng.randint(1, loan.outstanding), rnd)
            elif op == "charge":
                L.charge_compute(x, rng.choice([1, 12]))
            elif op == "reward":
                L.reward(x, rng.choice([8, 20, 50]))
            elif op == "settle":
                L.settle_due_loans(rnd)
            elif op == "default":
                L.default_loans_of(x, rnd)
        except LedgerError:
            pass  # rejected transactions must leave the books untouched
        r = L.reconcile()
        assert r["ok"], (seed, rnd, op, r)
        assert all(a.balance >= 0 for a in agents)
