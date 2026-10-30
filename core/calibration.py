"""
Calibration: does a stated confidence of 0.8 mean the answer is right ~80% of the time?

We never ASSUME the model's self-reported confidence is calibrated; we measure it from
graded outcomes. Two probabilities are evaluated against the same results:
  self_reported   the bid confidence the model stated
  policy_estimate the agent's own EV estimate (history blended with confidence)

Metrics (standard definitions)
  Brier score  mean((p - outcome)^2)           0 = perfect, 0.25 = always saying 0.5
  ECE          sum_b (n_b / N) * |acc_b - conf_b|    expected calibration error over bins
  reliability  per-bin (mean predicted, observed accuracy, n)

With few graded results these numbers are noise, so `reliable` is False below
MIN_RELIABLE_N and the dashboard says so instead of presenting a tidy-looking number.
"""
MIN_RELIABLE_N = 30
BINS = 5


def _bin(p: float) -> int:
    return min(int(p * BINS), BINS - 1)


def _evaluate(pairs: list[tuple[float, int]]) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None, "bins": [], "reliable": False,
                "note": "no graded results yet"}
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    bins = [[0, 0.0, 0] for _ in range(BINS)]          # n, sum_p, sum_y
    for p, y in pairs:
        b = bins[_bin(p)]
        b[0] += 1; b[1] += p; b[2] += y
    ece = sum(b[0] / n * abs(b[2] / b[0] - b[1] / b[0]) for b in bins if b[0])
    out_bins = [{"lo": i / BINS, "hi": (i + 1) / BINS, "n": b[0],
                 "mean_predicted": round(b[1] / b[0], 3) if b[0] else None,
                 "observed_accuracy": round(b[2] / b[0], 3) if b[0] else None}
                for i, b in enumerate(bins)]
    reliable = n >= MIN_RELIABLE_N
    return {"n": n, "brier": round(brier, 4), "ece": round(ece, 4), "bins": out_bins, "reliable": reliable,
            "note": "" if reliable else f"only {n} graded results (< {MIN_RELIABLE_N}); treat as indicative"}


def calibration_report(all_events: list) -> dict:
    """Build the report from task_resolved events. Ungraded (exhibition) results and events
    lacking numeric predictions are excluded, never imputed."""
    conf, est = [], []
    for e in all_events:
        if not isinstance(e, dict) or e.get("type") != "task_resolved" or e.get("exhibition"):
            continue
        y = e.get("correct")
        if not isinstance(y, bool):
            continue
        for key, bucket in (("bid_confidence", conf), ("p_est", est)):
            v = e.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0:
                bucket.append((float(v), int(y)))
    return {"self_reported": _evaluate(conf), "policy_estimate": _evaluate(est)}
