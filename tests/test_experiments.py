"""Statistics helpers and the experiment framework: correctness, honesty, reproducibility, isolation."""
import json
import math

import pytest

from core import config
from core.simconfig import ConfigError
from core.stats import MIN_N_FOR_CLAIMS, describe, paired_difference, t_critical_95, wilson_interval
from training import experiment as ex


# ------------------------------------------------------------- statistics
def test_describe_matches_a_hand_calculation():
    d = describe([1, 2, 3, 4, 5])
    assert (d["n"], d["mean"], d["median"], d["min"], d["max"]) == (5, 3.0, 3, 1, 5)
    sd = math.sqrt(2.5)
    assert d["std"] == pytest.approx(sd, abs=1e-4)
    half = 2.776 * sd / math.sqrt(5)                    # t(0.975, df=4) = 2.776
    assert d["ci95"] == [pytest.approx(3 - half, abs=1e-3), pytest.approx(3 + half, abs=1e-3)]


def test_describe_never_invents_spread_from_one_observation_or_garbage():
    d = describe([7])
    assert d["n"] == 1 and d["std"] is None and d["ci95"] is None
    assert describe([]) is None
    assert describe([None, "x", True, float("nan")]) is None
    assert describe([1, None, 3, float("nan"), True])["n"] == 2


def test_t_table_known_values_and_monotone():
    assert t_critical_95(1) == 12.706 and t_critical_95(4) == 2.776 and t_critical_95(30) == 2.042
    assert t_critical_95(500) == 1.96
    vals = [t_critical_95(d) for d in range(1, 31)]
    assert vals == sorted(vals, reverse=True)
    with pytest.raises(ValueError):
        t_critical_95(0)


def test_paired_difference_uses_pairs_and_flags_detectability_correctly():
    r = paired_difference([10, 12, 11, 13, 12, 14], [5, 6, 5, 7, 6, 6])
    assert r["detectable"] is True and r["direction"] == "higher" and r["mean_diff"] == pytest.approx(6.1667, abs=1e-3)
    assert r["ci95"][0] > 0
    noisy = paired_difference([1, 5, 2, 8, 3, 9], [4, 4, 4, 4, 4, 4])
    assert noisy["detectable"] is False and noisy["direction"] is None


def test_small_samples_are_never_called_detectable_even_with_a_huge_difference():
    r = paired_difference([100, 101, 102, 103], [0, 0, 0, 0])
    assert r["n"] == 4 < MIN_N_FOR_CLAIMS and r["detectable"] is False


def test_paired_difference_skips_unusable_pairs():
    r = paired_difference([1, None, 3, "x"], [0, 5, 0, 1])
    assert r["n"] == 2 and r["mean_diff"] == 2.0
    assert paired_difference([], []) is None


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.237, abs=1e-3), pytest.approx(0.763, abs=1e-3))
    assert wilson_interval(0, 0) is None
    assert wilson_interval(10, 10)[1] == 1.0 and wilson_interval(0, 10)[0] == 0.0


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("param,raw,expected", [
    ("interest_rate", "0.15", 0.15), ("interest_rate", "1", 1), ("max_rounds", "40", 40),
    ("strategies", "conservative", "conservative"), ("strategies", "a+b", ("a", "b")),
    ("bounties", "8:20:50", [8, 20, 50]), ("task_weights", "1:1:0", [1.0, 1.0, 0.0]),
])
def test_parse_value(param, raw, expected):
    assert ex.parse_value(param, raw) == expected


# ------------------------------------------------------------- the framework
def sweep(tmp_path, *extra, trials=2, rounds=8, seed=50, name="o"):
    return ex.main(["--param", "interest_rate", "--values", "0,0.3", "--trials", str(trials), "--rounds", str(rounds),
                    "--seed", str(seed), "--out", str(tmp_path / name), *extra])


def test_sweep_is_reproducible_and_writes_all_three_artifacts(tmp_path, isolated_data_dir):
    s1, s2 = sweep(tmp_path, name="a"), sweep(tmp_path, name="b")
    assert s1["conditions"] == s2["conditions"] and s1["comparisons"] == s2["comparisons"]
    for f in ("runs.csv", "summary.json", "analysis.md"):
        assert (tmp_path / "a" / f).exists()
    assert json.loads((tmp_path / "a" / "summary.json").read_text())["control"] == "0.0"


def test_conditions_share_the_same_seeds_so_comparisons_are_paired(tmp_path, isolated_data_dir):
    sweep(tmp_path, trials=3, name="p")
    import csv
    rows = list(csv.DictReader(open(tmp_path / "p" / "runs.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["value"], []).append(r["seed"])
    assert by["0.0"] == by["0.3"] == ["50", "51", "52"]


def test_every_trial_keeps_the_books_balanced(tmp_path, isolated_data_dir):
    s = sweep(tmp_path, trials=3)
    assert all(c["ledger_all_ok"] for c in s["conditions"].values())


def test_few_trials_never_produce_a_claimed_difference(tmp_path, isolated_data_dir):
    s = ex.main(["--param", "strategies", "--values", "baseline,aggressive", "--trials", "3", "--rounds", "20",
                 "--seed", "1", "--out", str(tmp_path / "f")])
    assert all(not (d and d["detectable"]) for comp in s["comparisons"].values() for d in comp.values())
    txt = (tmp_path / "f" / "analysis.md").read_text()
    assert "fewer than 5" in txt and "Absence of a detectable difference is not evidence" in txt


def test_analysis_labels_only_directional_metrics_as_good_or_bad():
    summary = {"param": "p", "mode": "mock", "trials": 6, "rounds": 10, "base_seed": 1, "control": "c",
               "warning": "w", "comparisons": {"x": {
                   "economic_profit": {"n": 6, "mean_diff": 5.0, "ci95": [1.0, 9.0], "detectable": True, "direction": "higher"},
                   "escalations": {"n": 6, "mean_diff": -2.0, "ci95": [-3.0, -1.0], "detectable": True, "direction": "lower"},
                   "compute_spent": {"n": 6, "mean_diff": -4.0, "ci95": [-5.0, -3.0], "detectable": True, "direction": "lower"},
                   "accuracy": {"n": 6, "mean_diff": 0.1, "ci95": [-0.1, 0.3], "detectable": False, "direction": None}}}}
    md = ex.analysis_markdown(summary)
    assert "**economic_profit** is higher" in md and "favourable" in md
    assert "**compute_spent** is lower" in md.replace("\n", " ") and md.count("unfavourable") == 0
    escalation_line = [l for l in md.splitlines() if "escalations" in l][0]
    assert "descriptive" in escalation_line and "favourable" not in escalation_line.replace("descriptive", "")
    assert "**accuracy**" not in md                                 # its interval includes 0: not reported


def test_no_detectable_difference_is_stated_plainly():
    summary = {"param": "p", "mode": "mock", "trials": 10, "rounds": 10, "base_seed": 1, "control": "c", "warning": "w",
               "comparisons": {"x": {"accuracy": {"n": 10, "mean_diff": 0.0, "ci95": [-1, 1], "detectable": False, "direction": None}}}}
    assert "No metric showed a detectable difference" in ex.analysis_markdown(summary)


def test_experiments_write_only_inside_their_output_directory(tmp_path, isolated_data_dir):
    from core import events, model_client
    events.reset_log()
    events.emit("live-marker")
    live = {"log": events.LOG_PATH, "board": events.LEADERBOARD_PATH, "replays": events.REPLAY_DIR,
            "calls": model_client.TRAINING_LOG_PATH}
    live_log_before = live["log"].read_bytes()
    sweep(tmp_path, name="iso", rounds=12)
    assert live["log"].read_bytes() == live_log_before              # the live event log is untouched
    assert not live["board"].exists()                               # no live leaderboard rows
    assert not live["replays"].exists() or not list(live["replays"].glob("*"))
    assert not live["calls"].exists()
    out = tmp_path / "iso"
    assert (out / "events.jsonl").exists() and (out / "leaderboard.jsonl").exists()      # all inside --out


def test_invalid_input_is_rejected_before_any_trial_runs(tmp_path):
    for bad in (["--param", "interest_rate", "--values", "5"], ["--param", "num_agents", "--values", "99"],
                ["--param", "strategies", "--values", "nope"], ["--param", "bounties", "--values", "1:2"]):
        with pytest.raises(ConfigError):
            ex.main([*bad, "--trials", "1", "--rounds", "2", "--out", str(tmp_path / "x")])
    assert not (tmp_path / "x" / "runs.csv").exists()
    with pytest.raises(SystemExit):
        ex.main(["--param", "interest_rate", "--values", "0,0.1", "--control", "0.9", "--trials", "1",
                 "--rounds", "2", "--out", str(tmp_path / "y")])


def test_live_mode_requires_explicit_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    with pytest.raises(SystemExit) as e:
        ex.main(["--tournament", "--trials", "1", "--out", str(tmp_path)])
    assert "confirm-live" in str(e.value)


def test_tournament_rotates_seats_so_no_strategy_keeps_a_fixed_position(tmp_path, isolated_data_dir):
    s = ex.main(["--tournament", "--trials", "5", "--rounds", "10", "--seed", "3", "--out", str(tmp_path / "t")])
    import csv
    seatings = [r["value"].split("+") for r in csv.DictReader(open(tmp_path / "t" / "runs.csv"))]
    for seat in range(5):
        assert {row[seat] for row in seatings} == set(seatings[0]), seat       # every strategy sits in every seat
    assert s["kind"] == "tournament" and s["ledger_all_ok"] is True
    assert set(s["strategies"]) == {"conservative", "aggressive", "cost_efficient", "accuracy_focused", "lender"}
    assert all(t["economic_profit"]["n"] == 5 for t in s["strategies"].values())    # one agent per trial each


def test_tournament_is_reproducible(tmp_path, isolated_data_dir):
    a = ex.main(["--tournament", "--trials", "3", "--rounds", "12", "--seed", "9", "--out", str(tmp_path / "a")])
    b = ex.main(["--tournament", "--trials", "3", "--rounds", "12", "--seed", "9", "--out", str(tmp_path / "b")])
    assert a["strategies"] == b["strategies"] and a["comparisons"] == b["comparisons"]


def test_tournament_few_trials_claims_no_ranking(tmp_path, isolated_data_dir):
    ex.main(["--tournament", "--trials", "2", "--rounds", "10", "--seed", "3", "--out", str(tmp_path / "t")])
    assert "no ranking is claimed" in (tmp_path / "t" / "analysis.md").read_text()
