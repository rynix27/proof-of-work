"""
Launches a fine-tuning job on Nebius Token Factory from a dataset built by
build_dataset.py.

WHAT IS VERIFIED vs NOT (see docs/ENGINEERING_AUDIT.md)
  Verified against docs.tokenfactory.nebius.com: Token Factory exposes
  client.files.create(purpose="fine-tune") and client.fine_tuning.jobs.create /
  retrieve through the OpenAI-compatible API at NEBIUS_BASE_URL.
  NOT verified (no credentials in the build environment):
    * that the Nemotron models you use are in Nebius's list of models supported
      for fine-tuning (check docs.tokenfactory.nebius.com/post-training/models),
    * the exact job/hyperparameter fields,
    * that a finished job's `fine_tuned_model` is directly servable. Nebius
      documents a separate DEPLOY step for fine-tuned LoRA models
      (docs.tokenfactory.nebius.com/fine-tuning/deploy-custom-model). Only after
      deployment should the served model name go in FINE_TUNED_NANO_MODEL.
  Nothing in this repository claims a fine-tune has been run or has helped.

The network boundary is isolated in `launch()` / `build_job_request()` so
everything else (dataset validation, request construction) is unit tested.

Usage:
    python -m training.finetune --dataset data/training/datasets/nano_all.jsonl --dry-run
    python -m training.finetune --dataset ... --base-model <supported model id>
"""
import argparse
import json
import sys
import time
from pathlib import Path

from core import config


class DatasetError(ValueError):
    pass


def validate_dataset(path: Path) -> int:
    """Local sanity checks so a malformed dataset fails fast, before a paid job.
    Raises DatasetError; returns the number of examples."""
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    count = 0
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise DatasetError(f"Line {line_num}: invalid JSON")
            msgs = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(msgs, list) or not msgs:
                raise DatasetError(f"Line {line_num}: missing or empty 'messages'")
            for m in msgs:
                if (not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                        or not isinstance(m.get("content"), str) or not m["content"].strip()):
                    raise DatasetError(f"Line {line_num}: each message needs a valid role and non-empty text")
            if msgs[-1]["role"] != "assistant":
                raise DatasetError(f"Line {line_num}: last message must be the assistant answer to imitate")
            count += 1
    if count == 0:
        raise DatasetError("Dataset is empty -- run more self-play and rebuild it.")
    return count


def build_job_request(training_file_id: str, base_model: str, suffix: str,
                      validation_file_id: str | None = None) -> dict:
    req = {"training_file": training_file_id, "model": base_model, "suffix": suffix}
    if validation_file_id:
        req["validation_file"] = validation_file_id
    return req


def launch(dataset_path: Path, base_model: str, suffix: str, dry_run: bool,
           validation_path: Path | None = None, timeout_s: float = 6 * 3600, client=None):
    count = validate_dataset(dataset_path)
    print(f"Dataset OK: {count} examples in {dataset_path}")
    if count < 10:
        print(f"WARNING: only {count} examples; results will be unreliable.")
    if validation_path:
        print(f"Validation set OK: {validate_dataset(validation_path)} examples")

    if dry_run:
        print("--dry-run: not contacting Nebius.")
        print("Would create job:", build_job_request("<uploaded-file-id>", base_model, suffix))
        return None

    if client is None:
        if not config.NEBIUS_API_KEY:
            raise SystemExit("NEBIUS_API_KEY is not set in .env -- can't launch a real job.")
        from openai import OpenAI
        client = OpenAI(api_key=config.NEBIUS_API_KEY, base_url=config.NEBIUS_BASE_URL)

    print(f"Uploading {dataset_path} ...")
    with open(dataset_path, "rb") as f:
        training = client.files.create(file=f, purpose="fine-tune")
    val_id = None
    if validation_path:
        with open(validation_path, "rb") as f:
            val_id = client.files.create(file=f, purpose="fine-tune").id

    job = client.fine_tuning.jobs.create(**build_job_request(training.id, base_model, suffix, val_id))
    print(f"Job created: id={job.id} status={job.status}")
    print("Polling (Ctrl+C is safe; the job keeps running; use --poll-only <job_id>).")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  status: {job.status}")
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(20)
    else:
        print("Timed out waiting; the job may still be running. Re-run with --poll-only.")
        return job

    _report(job)
    return job


def _report(job):
    if job.status != "succeeded":
        print(f"Job ended with status: {job.status}. Check the Nebius dashboard for details.")
        return
    print()
    print(f"Job succeeded. fine_tuned_model={getattr(job, 'fine_tuned_model', None)!r}")
    print("NEXT: Nebius requires deploying the fine-tuned adapter before it can be served "
          "(docs.tokenfactory.nebius.com/fine-tuning/deploy-custom-model). Once deployed, put the "
          "served model name in .env as FINE_TUNED_NANO_MODEL, then run training/evaluate.py "
          "against the same held-out tasks to measure whether it helped.")


def poll_only(job_id: str, client=None):
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=config.NEBIUS_API_KEY, base_url=config.NEBIUS_BASE_URL)
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"status: {job.status}")
    _report(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--base-model", default=None, help="defaults to NANO_MODEL")
    parser.add_argument("--suffix", default="proof-of-work")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-only", metavar="JOB_ID")
    args = parser.parse_args()
    try:
        if args.poll_only:
            poll_only(args.poll_only)
        else:
            if not args.dataset:
                parser.error("--dataset is required unless using --poll-only")
            launch(args.dataset, args.base_model or config.NANO_MODEL, args.suffix, args.dry_run,
                   validation_path=args.validation)
    except DatasetError as e:
        print(e)
        sys.exit(1)
