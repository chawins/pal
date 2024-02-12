"""Gather best suffixes from a run."""

import argparse
import json
from pathlib import Path


def main(args):
    exp_name = args.exp_name
    model, attack, params = exp_name.split("_", maxsplit=2)
    path = Path(f"results/{model}/{attack}/{params}")
    assert path.exists()

    suffixes = [""] * 50
    for result_path in path.glob("*.jsonl"):
        with result_path.open("r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f]
        # Best loss is on the last line
        assert "best_loss" in results[-1], "Run may be unfinished."
        best_suffix = results[-1]["suffix"]
        idx = int(result_path.name.split("_")[-1].replace(".jsonl", ""))
        suffixes[idx] = best_suffix

    if not all(suffixes):
        print("WARNING: Some suffixes are empty.")
    with Path(exp_name + ".jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(suffixes))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_name", type=str)
    _args = argparser.parse_args()
    main(_args)
