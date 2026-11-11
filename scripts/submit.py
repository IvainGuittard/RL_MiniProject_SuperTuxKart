#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlarena-sdk==4.6.0", "python-dotenv"]
# ///
"""Submits the stk_actor/ folder on ML-Arena and follows its evaluation.

Run it with:

    uv run scripts/submit.py

The API key is created on your ML-Arena profile page (Profile → API Keys) and
read from the MLARENA_API_KEY variable, which can be set in the .env file of
the base directory (MLARENA_API_KEY=mlk_...); it is asked if not set.

A deployed submission cannot be modified, so each call creates a new
submission, whose id is saved in .mlarena_submission: when the previous one
could not be deployed (e.g. its files were rejected), it is updated instead
(use --new to always create a new one). The submission is named after the
player_name of pystk_actor.py (or --name).
"""

import argparse
import ast
import getpass
import os
import sys
from pathlib import Path

import mlarena
from dotenv import load_dotenv

from submission_files import AGENT_DIR, ROOT, CheckError, collect_files

CHALLENGE_ID = 169
#: The runtime is the one that provides pystk2-gymnasium
RUNTIME_PACKAGE = "pystk2-gymnasium"
#: Where the submission id is saved
SUBMISSION_FILE = ROOT / ".mlarena_submission"
#: Names that must be changed
DEFAULT_NAMES = {"", "Example"}


def find_runtime(client) -> dict:
    """Returns the runtime that provides pystk2-gymnasium"""
    options = client.runtime_options(CHALLENGE_ID)
    rows = options.to_dict("records") if hasattr(options, "to_dict") else options
    for runtime in rows:
        if any(p["name"] == RUNTIME_PACKAGE for p in runtime.get("packages") or []):
            return runtime
    raise CheckError(f"no runtime provides {RUNTIME_PACKAGE} on ML-Arena")


def player_name(folder: Path) -> str:
    """Reads ``player_name`` from pystk_actor.py (without importing it)"""
    tree = ast.parse((folder / "pystk_actor.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "player_name" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise CheckError(
        "player_name must be set to a string in pystk_actor.py (or use --name)"
    )


def submission_name(args) -> str:
    """The submission name: --name, or the player name"""
    name = args.name or player_name(args.folder)
    if name in DEFAULT_NAMES:
        raise CheckError(f"change player_name in pystk_actor.py (it is still {name!r})")
    if not 1 <= len(name) <= 40 or "\n" in name or set(name) & {"<", ">"}:
        raise CheckError(
            f"invalid submission name {name!r}: 1 to 40 characters on one line,"
            " without < or >"
        )
    return name


def upload(client, submission_id: int, files: list[Path]):
    """Uploads the files, removing the ones that are not part of the agent"""
    names = {path.name for path in files}
    for path in files:
        print(f"Uploading {path.name}")
        client.upload_submission_file(CHALLENGE_ID, submission_id, str(path))

    listing = client.list_submission_files(CHALLENGE_ID, submission_id)
    for info in listing["files"]:
        if info["name"] not in names and info["role"] != "report":
            print(f"Removing {info['name']}")
            client.delete_submission_file(CHALLENGE_ID, submission_id, info["name"])


def submit(args):
    files = collect_files(args.folder)
    name = submission_name(args)

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("MLARENA_API_KEY") or getpass.getpass(
        "ML-Arena API key (Profile → API Keys): "
    )
    client = mlarena.connect(api_key=api_key)

    submission_id = args.submission_id
    if submission_id is None and not args.new and SUBMISSION_FILE.is_file():
        submission_id = int(SUBMISSION_FILE.read_text())

    status = None
    if submission_id is not None:
        status = client.submission(submission_id)
        if not status["is_uploadable"]:
            # A deployed (active) submission cannot be modified
            print(
                f"Submission {submission_id} is {status['status']} and cannot be"
                " updated: creating a new one"
            )
            submission_id, status = None, None

    if submission_id is None:
        created = client.create_submission(CHALLENGE_ID, name)
        submission_id = created["submission_id"]
        SUBMISSION_FILE.write_text(f"{submission_id}\n")
        print(
            f"Created submission {submission_id} ({name}, saved in "
            f"{SUBMISSION_FILE.name})"
        )
    else:
        print(f"Updating submission {submission_id}")
        if status["submission_name"] != name:
            print(
                f"Renaming the submission from {status['submission_name']!r}"
                f" to {name!r}"
            )
            client.update_submission(submission_id, submission_name=name)

    runtime = find_runtime(client)
    current_runtime = (status or {}).get("agent_runtime") or {}
    if current_runtime.get("id") != runtime["id"]:
        print(f"Using runtime {runtime['display_name']}")
        client.set_agent_runtime(submission_id, runtime["id"])

    upload(client, submission_id, files)

    status = client.submission(submission_id)
    if not status["is_deployable"]:
        raise CheckError(f"the files were rejected: {status['last_status_message']}")
    client.deploy_submission(CHALLENGE_ID, submission_id)
    print(f"Deployed: https://ml-arena.com/viewchallenge/{CHALLENGE_ID}")

    if not args.no_wait:
        for line in client.tail_logs(CHALLENGE_ID, submission_id):
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "folder", nargs="?", type=Path, default=AGENT_DIR, help="Agent folder"
    )
    parser.add_argument("--submission-id", type=int, help="Submission to update")
    parser.add_argument("--new", action="store_true", help="Create a new submission")
    parser.add_argument(
        "--name", help="Submission name (default: player_name of pystk_actor.py)"
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Do not follow the evaluation"
    )
    args = parser.parse_args()

    try:
        submit(args)
    except (CheckError, mlarena.MLArenaError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
