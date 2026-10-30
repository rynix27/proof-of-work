"""Phase 4: transaction journal, idempotency, atomic rollback, reconciliation, reset."""
import random

import pytest

import core.orchestrator as orch
from core import events
from core.agents import fresh_roster
from core.economy import Ledger, LedgerError
from core.simconfig import SimulationConfig


def make(n=3, **cfg):
    agents = fresh_roster(n)
    return agents, Ledger(agents, SimulationConfig.defaults(**cfg))


def snapshot(agents, L):
    return ([(a.balance, a.bankrupt, a.compute_spent, a.rewards_earned) for a in agents],
            len(L.journal), len(L.loans), L.minted, L.burned, set(L._refs))


def test_every_transaction_gets_a_unique_increasing_id_and_is_journaled(isolated_data_dir):
    (a, b, _), L = make()
    t1 = L.charge_compute(a, 1, 1, ref="c1")
    t2 = L.reward(a, 8, 1, ref="b1")
    loan = L.create_loan(a, b, 20, 1, request_key="k1")
    L.repay(loan, 5, 2)
    ids = [t.id for t in L.journal]
    assert ids == ["t1", "t2", "t3", "t4"] and (t1, t2, loan.txn_id) == ("t1", "t2", "t3")
    assert [t.kind for t in L.journal] == ["compute", "bounty", "loan_principal", "loan_repayment"]
    assert loan.history == [{"round": 2, "amount": 5, "txn_id": "t4"}]


def test_balances_rederived_from_the_journal_match_live_balances():
    agents, L = make(4)
    rng = random.Random(1)
    for rnd in range(1, 60):
        x, y = rng.sample(agents, 2)
        try:
            op = rng.choice(["loan", "repay", "charge", "reward", "settle", "default"])
            if op == "loan":
                L.create_loan(x, y, rng.randint(1, 40), rnd)
            elif op == "repay" and L.active_loans():
                ln = rng.choice(L.active_loans()); L.repay(ln, rng.randint(1, ln.outstanding), rnd)
            elif op == "charge":
                L.charge_compute(x, rng.choice([1, 12]), rnd)
            elif op == "reward":
                L.reward(x, 8, rnd)
            elif op == "settle":
                L.settle_due_loans(rnd)
            else:
                L.default_loans_of(x, rnd)
        except LedgerError:
            pass
    r = L.reconcile()
    assert r["ok"] and r["journal_ok"] and r["txn_ids_unique"]
    assert L.replay_balances() == {a.name: a.balance for a in agents}


def test_reconcile_detects_an_unjournaled_balance_change():
    (a, b, _), L = make()
    L.create_loan(a, b, 10, 1)
    a.balance += 5                      # tampering that bypasses the ledger
    r = L.reconcile()
    assert r["ok"] is False and r["journal_ok"] is False


# ---------------------------------------------------------------- idempotency
def test_duplicate_bounty_for_the_same_task_is_refused(isolated_data_dir):
    (a, *_), L = make()
    L.reward(a, 8, 1, ref="bounty:1:7:A")
    before = snapshot([a], L)
    with pytest.raises(LedgerError) as e:
        L.reward(a, 8, 1, ref="bounty:1:7:A")
    assert e.value.code == "duplicate_transaction" and snapshot([a], L) == before


def test_duplicate_compute_charge_is_refused():
    (a, *_), L = make()
    L.charge_compute(a, 1, 1, ref="solve:1:7:A")
    with pytest.raises(LedgerError):
        L.charge_compute(a, 1, 1, ref="solve:1:7:A")
    assert a.balance == 99


def test_repeated_loan_request_creates_exactly_one_loan(isolated_data_dir):
    (a, b, _), L = make()
    L.create_loan(a, b, 10, 1, request_key="lend:1:A")
    with pytest.raises(LedgerError) as e:
        L.create_loan(a, b, 10, 1, request_key="lend:1:A")
    assert e.value.code == "duplicate_transaction"
    assert len(L.loans) == 1 and (a.balance, b.balance) == (90, 110)


def test_a_rejected_loan_does_not_burn_its_request_key():
    """A request that fails validation must remain retryable with the same key."""
    (a, b, _), L = make()
    with pytest.raises(LedgerError):
        L.create_loan(a, b, 999, 1, request_key="k")        # over max
    L.create_loan(a, b, 10, 1, request_key="k")
    assert len(L.loans) == 1


def test_repayment_and_default_cannot_apply_twice(isolated_data_dir):
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 10, 1)
    L.settle_due_loans(loan.due_round)
    j = len(L.journal)
    L.settle_due_loans(loan.due_round)                      # same round settled again
    L.settle_due_loans(loan.due_round + 5)
    assert len(L.journal) == j and loan.status == "repaid"
    with pytest.raises(LedgerError):
        L.default_loan(loan, 9, "x")
    with pytest.raises(LedgerError):
        L.repay(loan, 1, 9)


# ----------------------------------------------------------------- atomicity
def test_failure_after_mutation_rolls_back_every_part_of_a_loan(isolated_data_dir, monkeypatch):
    (a, b, _), L = make()
    before = snapshot([a, b], L)

    def boom(*args, **kw):
        raise RuntimeError("event sink exploded mid-transaction")
    with monkeypatch.context() as m:               # NOT monkeypatch.undo(): that would also undo the
        m.setattr(events, "emit", boom)            # autouse fixture that keeps tests out of the real data/
        with pytest.raises(RuntimeError):
            L.create_loan(a, b, 20, 1, request_key="k")
    assert snapshot([a, b], L) == before and L._next_loan_id == 0
    L.create_loan(a, b, 20, 1, request_key="k")             # key was released by the rollback
    assert len(L.loans) == 1 and L.reconcile()["ok"]


def test_failure_mid_repayment_restores_loan_and_balances(isolated_data_dir, monkeypatch):
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, 1)
    before = (a.balance, b.balance, loan.outstanding, loan.repaid, list(loan.history), loan.status, len(L.journal))
    monkeypatch.setattr(events, "emit", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        L.repay(loan, 23, 2)
    assert (a.balance, b.balance, loan.outstanding, loan.repaid, loan.history, loan.status, len(L.journal)) == before
    assert L.reconcile()["ok"]


def test_failure_mid_default_restores_loan(isolated_data_dir, monkeypatch):
    (a, b, _), L = make()
    loan = L.create_loan(a, b, 20, 1)
    monkeypatch.setattr(events, "emit", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        L.default_loan(loan, 3, "test")
    assert loan.status == "active" and loan.outstanding == 23 and loan.written_off == 0


@pytest.mark.parametrize("op", [
    lambda L, a, b: L.create_loan(a, b, 0, 1), lambda L, a, b: L.create_loan(a, a, 5, 1),
    lambda L, a, b: L.create_loan(a, b, 10**6, 1), lambda L, a, b: L.charge_compute(a, 10**6),
    lambda L, a, b: L.charge_compute(a, -1), lambda L, a, b: L.reward(a, -5),
    lambda L, a, b: L.reward(a, float("nan")), lambda L, a, b: L.create_loan(a, b, "5", 1),
])
def test_every_rejected_operation_leaves_the_books_untouched(op):
    (a, b, _), L = make()
    before = snapshot([a, b], L)
    with pytest.raises(LedgerError):
        op(L, a, b)
    assert snapshot([a, b], L) == before


# ---------------------------------------------------------- debt & bankruptcy
def test_debt_and_lending_accessors():
    (a, b, c), L = make()
    L.create_loan(a, b, 20, 1)
    L.create_loan(a, c, 10, 1)
    assert L.debt_of(b) == 23 and L.lent_out(a) == 35 and L.loans_issued(a) == 2 and L.debt_of(a) == 0
    L.default_loans_of(b, 2)
    assert L.debt_of(b) == 0 and L.lent_out(a) == 12


def test_bankruptcy_writes_off_debt_without_moving_balances():
    (a, b, _), L = make()
    L.create_loan(a, b, 20, 1)
    b.balance = 0
    b.go_bankrupt()
    L.default_loans_of(b, 2)
    wo = [t for t in L.journal if t.kind == "loan_writeoff"]
    assert len(wo) == 1 and wo[0].amount == 23 and wo[0].effect == "writeoff"


def test_bankrupt_agents_cannot_bid_or_lend_but_can_be_revived_by_a_loan(isolated_data_dir, monkeypatch):
    events.reset_log()
    agents = fresh_roster()
    agents[0].balance = 0
    agents[0].go_bankrupt()
    seen = []
    monkeypatch.setattr(orch, "decide_action", lambda a, t, r: seen.append(a.name) or {"action": "skip"})
    orch.run_round(agents, 1)
    assert agents[0].name not in seen                        # bankrupt agents are not even asked
    (a, b, *_), L = make()
    b.balance = 0; b.go_bankrupt()
    L.create_loan(a, b, 5, 1)
    assert not b.bankrupt and b.balance == 5


# ---------------------------------------------------------------------- reset
def test_reset_gives_a_fresh_ledger_with_restarted_transaction_ids(isolated_data_dir):
    def final_recon():
        events.reset_log()
        orch.run_simulation(max_rounds=25, round_delay=0, verbose=False, seed=2)
        return [e for e in events.read_all() if e["type"] == "simulation_ended"][0]["reconciliation"]
    r1, r2 = final_recon(), final_recon()
    assert r1 == r2 and r1["ok"] and r1["transactions"] > 0           # identical, independent of the first run
    ids = [e["txn_id"] for e in events.read_all() if e["type"] == "loan_made"]
    assert ids and all(i.startswith("t") for i in ids)


def test_events_reference_every_journal_transaction_exactly_once(isolated_data_dir):
    events.reset_log()
    orch.run_simulation(max_rounds=40, round_delay=0, verbose=False, seed=4)
    evs = events.read_all()
    ended = [e for e in evs if e["type"] == "simulation_ended"][0]
    ids = [e[k] for e in evs for k in ("txn_id", "cost_txn", "reward_txn") if e.get(k)]
    assert len(ids) == len(set(ids)) == ended["reconciliation"]["transactions"]
