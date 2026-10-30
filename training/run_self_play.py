"""
Batch-runs many short simulations back to back, purely to accumulate
self-play history in data/training/calls.jsonl for the dataset builder.
Each run resets the economy (fresh agents, fresh balances) but every real
model call still gets appended to the persistent training log.

This is meant to run against the REAL Token Factory API (fake-mode calls
are never logged, since there's nothing genuine to learn from). Set
USE_FAKE_MODEL=0 in .env before running this.

Usage:
    python -m training.run_self_play --runs 20 --rounds-per-run 15
"""
import argparse

from core import config
from core.orchestrator import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="How many simulation runs to do.")
    parser.add_argument("--rounds-per-run", type=int, default=15,
                        help="Max rounds per run (kept short so you get many varied runs "
                             "rather than one long one).")
    parser.add_argument("--delay", type=float, default=0,
                        help="Seconds between rounds. 0 for fastest possible data collection.")
    args = parser.parse_args()

    if config.USE_FAKE_MODEL:
        print("USE_FAKE_MODEL=1 in your .env -- fake-mode calls are never logged as training "
              "data (there's nothing real to learn from). Set USE_FAKE_MODEL=0 and add your "
              "real NEBIUS_API_KEY before running this.")
        return

    for i in range(1, args.runs + 1):
        print(f"\n=== self-play run {i}/{args.runs} ===")
        run_simulation(max_rounds=args.rounds_per_run, round_delay=args.delay, verbose=True)

    print(f"\nDone. Ran {args.runs} runs. Now build a dataset with:")
    print("  python -m training.build_dataset")


if __name__ == "__main__":
    main()
