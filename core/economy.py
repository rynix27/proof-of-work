"""
Ledger: the ONLY place balances change. Every transaction goes through a
method here, validates its inputs, updates balances/loans atomically (single
thread owns a Ledger; see orchestrator) and emits its own event, so the event
log and the balances cannot disagree.

ACCOUNTING MODEL (all amounts are integer "compute-seconds")
  balance        spendable compute an agent holds.
  compute cost   simulated cost of a solve: NANO_COST or ULTRA_COST. Burned
                 (leaves the economy). Decisions (bid/skip/lend/borrow) are free.
  reward         the task bounty, minted into the winner's balance when graded
                 correct. Wrong answers earn nothing and lose only compute.
  conservation   sum(balances) == start_total + minted - burned, always.
                 Loans, repayments and interest only MOVE balance between agents;
                 defaults write off a claim but never create or destroy balance.
                 `reconcile()` checks this invariant.

LOANS (simple interest, single period, integer math)
  interest   = ceil(principal * rate_bp / 10000)        20 @ 15% -> 3
  total_owed = principal + interest                     -> 23
  A loan falls due `loan_term_rounds` after creation. At the due round the
  borrower repays min(balance, outstanding). Whatever is still unpaid is
  DEFAULTED (written off; the lender absorbs the loss). Voluntary partial
  repayment via repay() is supported. Terminal states (repaid, defaulted) are
  immutable, which prevents double repayment.
  Bankruptcy: when an agent goes bankrupt every active loan it owes is
  defaulted immediately. A bankrupt agent may be revived by a new loan
  (bail-out); its old defaulted debts stay written off.

TRANSACTION JOURNAL
  Every movement is appended to `journal` as a Txn with a unique, strictly
  increasing id (t1, t2, ...): compute burns, bounty mints, loan principal,
  repayments and write-offs. `replay_balances()` re-derives every balance from
  the journal alone, and `reconcile()` requires it to match the live balances
  (an independent check of the live arithmetic).

IDEMPOTENCY / ATOMICITY
  * `ref` / `request_key` make a transaction single-use: presenting the same
    reference twice raises LedgerError("duplicate_transaction") and changes
    nothing (no double bounty, no double charge, no duplicate loan).
  * Every mutating method validates first and then mutates inside `_atomic`,
    which snapshots the touched state and restores it if anything raises, so a
    failed transaction can never leave a half-applied change. The idempotency
    key is claimed INSIDE the snapshot, so a rolled-back attempt releases it.
  * Balances are integers and can never go negative: an overdraft raises.

FINAL SCORE
  An agent's final score is its final balance (0 if bankrupt). Outstanding debt
  is not subtracted from a solvent agent's balance; it is reported separately.
"""
import contextlib
from dataclasses import dataclass, asdict, field

from core import events
from core.simconfig import SimulationConfig


class LedgerError(ValueError):
    """A transaction was rejected. `.code` is a short machine-readable reason."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


@dataclass
class Loan:
    id: int
    lender: str
    borrower: str
    principal: int
    interest_rate: float
    interest: int
    total_owed: int
    outstanding: int
    repaid: int
    created_round: int
    due_round: int
    status: str = "active"          # active | repaid | defaulted
    default_round: int | None = None
    written_off: int = 0
    initiated_by: str = "lender"
    history: list = field(default_factory=list)   # repayment history: {round, amount, txn_id}
    txn_id: str | None = None                      # the principal-transfer transaction

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Txn:
    """One journal entry. effect: transfer (src->dst), mint (-> dst), burn (src ->),
    writeoff (bookkeeping only, moves no balance)."""
    id: str
    kind: str          # compute | bounty | loan_principal | loan_repayment | loan_writeoff
    effect: str
    round: int
    src: str | None
    dst: str | None
    amount: int
    loan_id: int | None = None
    ref: str | None = None


def compute_interest(principal: int, rate_bp: int) -> int:
    """ceil(principal * rate_bp / 10000) using integers only."""
    return -(-principal * rate_bp // 10000)


class Ledger:
    def __init__(self, agents: list, cfg: SimulationConfig):
        self.agents = agents
        self.cfg = cfg
        self.by_name = {a.name: a for a in agents}
        self.loans: list[Loan] = []
        self._next_loan_id = 0
        self.start_total = sum(a.balance for a in agents)
        self.minted = 0
        self.burned = 0
        self.journal: list[Txn] = []
        self._start_balances = {a.name: a.balance for a in agents}
        self._refs: set[tuple[str, str]] = set()      # (namespace, ref) already used

    # ------------------------------------------------------------ plumbing
    def _txn(self, kind, effect, round_num, src, dst, amount, loan_id=None, ref=None) -> str:
        t = Txn(f"t{len(self.journal) + 1}", kind, effect, round_num, src, dst, amount, loan_id, ref)
        self.journal.append(t)
        return t.id

    def _claim(self, namespace: str, ref: str | None):
        """Mark a reference single-use. Raises if it was already used."""
        if ref is None:
            return
        if (namespace, ref) in self._refs:
            raise LedgerError("duplicate_transaction", f"{namespace} reference {ref!r} was already applied")
        self._refs.add((namespace, ref))

    @contextlib.contextmanager
    def _atomic(self, *agents, loans=()):
        """Snapshot, run, and restore on ANY exception: all-or-nothing."""
        snap = [(a, a.balance, a.bankrupt, a.compute_spent, a.rewards_earned) for a in agents]
        loan_snap = [(l, l.status, l.outstanding, l.repaid, l.written_off, l.default_round, list(l.history))
                     for l in loans]
        jlen, nloans, minted, burned, refs, nid = (len(self.journal), len(self.loans), self.minted,
                                                    self.burned, set(self._refs), self._next_loan_id)
        try:
            yield
        except BaseException:
            for a, bal, bk, cs, re_ in snap:
                a.balance, a.bankrupt, a.compute_spent, a.rewards_earned = bal, bk, cs, re_
            for l, st, out, rep, wo, dr, hist in loan_snap:
                l.status, l.outstanding, l.repaid, l.written_off, l.default_round, l.history = st, out, rep, wo, dr, hist
            del self.journal[jlen:]
            del self.loans[nloans:]
            self.minted, self.burned, self._refs, self._next_loan_id = minted, burned, refs, nid
            raise

    # ------------------------------------------------------------ compute
    def charge_compute(self, agent, cost: int, round_num: int = 0, ref: str | None = None) -> str:
        """Debit simulated compute (burned). Returns the transaction id.
        Never overdraws; `ref` makes the charge single-use."""
        self._check_int(cost, "cost", lo=0)
        if cost > agent.balance:
            raise LedgerError("insufficient_funds", f"{agent.name} has {agent.balance}, needs {cost}")
        with self._atomic(agent):
            self._claim("compute", ref)          # inside the snapshot: a rollback releases the key
            agent.balance -= cost
            agent.compute_spent += cost
            self.burned += cost
            return self._txn("compute", "burn", round_num, agent.name, None, cost, ref=ref)

    def reward(self, agent, amount: int, round_num: int = 0, ref: str | None = None) -> str:
        """Mint a task bounty into `agent`. `ref` makes it single-use (no double bounty)."""
        self._check_int(amount, "reward", lo=0)
        with self._atomic(agent):
            self._claim("bounty", ref)
            agent.balance += amount
            agent.rewards_earned += amount
            self.minted += amount
            return self._txn("bounty", "mint", round_num, None, agent.name, amount, ref=ref)

    # ------------------------------------------------------------- loans
    def active_loans(self, borrower=None) -> list[Loan]:
        return [l for l in self.loans if l.status == "active"
                and (borrower is None or l.borrower == borrower.name)]

    def validate_loan(self, lender, borrower, amount) -> None:
        self._check_int(amount, "amount", lo=1)
        if lender is borrower:
            raise LedgerError("self_loan", "cannot lend to self")
        if lender.bankrupt:
            raise LedgerError("lender_bankrupt", "bankrupt agents cannot lend")
        if amount > self.cfg.max_loan_amount:
            raise LedgerError("over_max_loan", f"max loan is {self.cfg.max_loan_amount}")
        if lender.balance - amount < self.cfg.lender_reserve:
            raise LedgerError("lender_reserve",
                              f"lender must keep at least {self.cfg.lender_reserve} after lending")
        if len(self.active_loans(borrower)) >= self.cfg.max_active_loans:
            raise LedgerError("too_many_loans", "borrower is at the active-loan limit")

    def create_loan(self, lender, borrower, amount: int, round_num: int,
                    initiated_by: str = "lender", request_key: str | None = None) -> Loan:
        self.validate_loan(lender, borrower, amount)
        interest = compute_interest(amount, self.cfg.rate_bp)
        with self._atomic(lender, borrower):
            self._claim("loan", request_key)
            self._next_loan_id += 1
            loan = Loan(
                id=self._next_loan_id, lender=lender.name, borrower=borrower.name,
                principal=amount, interest_rate=self.cfg.interest_rate, interest=interest,
                total_owed=amount + interest, outstanding=amount + interest, repaid=0,
                created_round=round_num, due_round=round_num + self.cfg.loan_term_rounds,
                initiated_by=initiated_by,
            )
            lender.balance -= amount
            borrower.balance += amount
            revived = borrower.bankrupt
            if revived:
                borrower.bankrupt = False
            loan.txn_id = self._txn("loan_principal", "transfer", round_num, lender.name,
                                    borrower.name, amount, loan_id=loan.id, ref=request_key)
            self.loans.append(loan)
            events.emit("loan_made", round=round_num, loan_id=loan.id, txn_id=loan.txn_id,
                        lender=lender.name, borrower=borrower.name, amount=amount, principal=amount,
                        interest_rate=loan.interest_rate, interest=interest,
                        total_owed=loan.total_owed, due_round=loan.due_round,
                        initiated_by=initiated_by, revived=revived,
                        lender_balance=lender.balance, borrower_balance=borrower.balance)
            return loan

    def repay(self, loan: Loan, amount: int, round_num: int) -> int:
        """Repay `amount` of `loan` (partial allowed). Returns amount paid."""
        if loan.status != "active":
            raise LedgerError("loan_not_active", f"loan {loan.id} is {loan.status}")
        self._check_int(amount, "amount", lo=1)
        borrower, lender = self.by_name[loan.borrower], self.by_name[loan.lender]
        if amount > loan.outstanding:
            raise LedgerError("over_repayment", "amount exceeds outstanding balance")
        if amount > borrower.balance:
            raise LedgerError("insufficient_funds", "borrower cannot cover this repayment")
        with self._atomic(borrower, lender, loans=(loan,)):
            borrower.balance -= amount
            lender.balance += amount
            loan.outstanding -= amount
            loan.repaid += amount
            if loan.outstanding == 0:
                loan.status = "repaid"
            txn = self._txn("loan_repayment", "transfer", round_num, borrower.name, lender.name,
                            amount, loan_id=loan.id)
            loan.history.append({"round": round_num, "amount": amount, "txn_id": txn})
            events.emit("loan_repayment", round=round_num, loan_id=loan.id, txn_id=txn,
                        lender=loan.lender, borrower=loan.borrower, amount=amount,
                        outstanding=loan.outstanding, status=loan.status,
                        lender_balance=lender.balance, borrower_balance=borrower.balance)
        return amount

    def default_loan(self, loan: Loan, round_num: int, reason: str):
        if loan.status != "active":
            raise LedgerError("loan_not_active", f"loan {loan.id} is {loan.status}")
        with self._atomic(loans=(loan,)):
            loan.status = "defaulted"
            loan.default_round = round_num
            loan.written_off = loan.outstanding
            loan.outstanding = 0
            txn = self._txn("loan_writeoff", "writeoff", round_num, loan.borrower, loan.lender,
                            loan.written_off, loan_id=loan.id)
            events.emit("loan_default", round=round_num, loan_id=loan.id, txn_id=txn,
                        lender=loan.lender, borrower=loan.borrower, written_off=loan.written_off,
                        repaid=loan.repaid, reason=reason)

    def settle_due_loans(self, round_num: int):
        """Collect what can be collected on loans due this round; default the rest."""
        for loan in sorted(self.active_loans(), key=lambda l: l.id):
            if loan.due_round > round_num:
                continue
            borrower = self.by_name[loan.borrower]
            pay = min(borrower.balance, loan.outstanding)
            if pay > 0:
                self.repay(loan, pay, round_num)
            if loan.status == "active":
                self.default_loan(loan, round_num, "insufficient_funds_at_due")

    def default_loans_of(self, borrower, round_num: int):
        for loan in self.active_loans(borrower):
            self.default_loan(loan, round_num, "borrower_bankrupt")

    # -------------------------------------------------------- reporting
    def outstanding_debt(self) -> int:
        return sum(l.outstanding for l in self.loans if l.status == "active")

    def loan_summary(self) -> dict:
        settled = [l for l in self.loans if l.status in ("repaid", "defaulted")]
        owed = sum(l.total_owed for l in settled)
        return {
            "loans_made": len(self.loans),
            "loan_volume": sum(l.principal for l in self.loans),
            "outstanding_debt": self.outstanding_debt(),
            "loans_repaid": sum(1 for l in self.loans if l.status == "repaid"),
            "loans_defaulted": sum(1 for l in self.loans if l.status == "defaulted"),
            "written_off": sum(l.written_off for l in self.loans),
            "repayment_rate": round(sum(l.repaid for l in settled) / owed, 3) if owed else None,
        }

    def debt_of(self, agent) -> int:
        """Outstanding amount `agent` currently owes (active loans)."""
        return sum(l.outstanding for l in self.loans if l.status == "active" and l.borrower == agent.name)

    def lent_out(self, agent) -> int:
        """Outstanding amount owed TO `agent` (active loans)."""
        return sum(l.outstanding for l in self.loans if l.status == "active" and l.lender == agent.name)

    def loans_issued(self, agent) -> int:
        return sum(1 for l in self.loans if l.lender == agent.name)

    def replay_balances(self) -> dict:
        """Re-derive every balance from the journal alone."""
        bal = dict(self._start_balances)
        for t in self.journal:
            if t.effect == "writeoff":
                continue
            if t.src is not None:
                bal[t.src] -= t.amount
            if t.dst is not None:
                bal[t.dst] += t.amount
        return bal

    def reconcile(self) -> dict:
        """Two independent checks: (1) the conservation law
        sum(balances) == start + minted - burned, and (2) balances re-derived from
        the transaction journal equal the live balances."""
        expected = self.start_total + self.minted - self.burned
        actual = sum(a.balance for a in self.agents)
        journal_ok = self.replay_balances() == {a.name: a.balance for a in self.agents}
        ids = [t.id for t in self.journal]
        unique_ok = len(ids) == len(set(ids))
        return {"expected_total": expected, "actual_total": actual,
                "journal_ok": journal_ok, "txn_ids_unique": unique_ok, "transactions": len(self.journal),
                "ok": expected == actual and journal_ok and unique_ok,
                "minted": self.minted, "burned": self.burned}

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _check_int(value, name: str, lo: int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise LedgerError(f"bad_{name}", f"{name} must be an integer")
        if value < lo:
            raise LedgerError(f"bad_{name}", f"{name} must be >= {lo}")
