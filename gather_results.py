"""Print results to table."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from tabulate import tabulate

# MODEL = "Llama-2-7b-chat-hf"
MODEL = "vicuna-7b-v1.3"
# MODEL = "mistral-7b-instruct-v0.1"
ATTACKS = ["gcg", "ral"]

# Specify keys/params to reduce over
# SEEDS = (0, 10, 20)
# SEEDS = (0,)
SEEDS = None
# BEHAVIOR_IDS = tuple(range(12))
BEHAVIOR_IDS = tuple(range(50))


class ResultTable:
    def __init__(self):
        self.headers = ["attack", "exp_name", "seed", "behavior_id", "data"]
        self.table = []

    def add_row(self, row):
        """Add experiment data to table."""
        assert len(row) == len(self.headers)
        self.table.append(row)

    def _get_run_id(self, row):
        run_id = []
        for i, header in enumerate(self.headers):
            if header in ("seed", "data"):
                continue
            run_id.append(row[i])
        return tuple(run_id)

    def _reduce_by_key(self, table, headers, key, vals):
        col_ids = [i for i, h in enumerate(headers) if h != key]
        reduce_id = headers.index(key)
        temp_table = {}

        for row in table:
            run_id = tuple(row[i] for i in col_ids)
            if run_id not in temp_table:
                temp_table[run_id] = {}
            temp_table[run_id][row[reduce_id]] = row[-1]

        combined_table = []
        for run_id, exp_dict in temp_table.items():
            if any(v not in exp_dict for v in vals):
                continue
            # Aggregate data across specified vals
            combined_data = {}
            for val in vals:
                for k, v in exp_dict[val].items():
                    # v can be list if multiple keys are being reduced
                    if k not in combined_data:
                        combined_data[k] = v if isinstance(v, list) else [v]
                    elif isinstance(v, list):
                        combined_data[k].extend(v)
                    else:
                        combined_data[k].append(v)
            combined_table.append(
                [
                    *run_id[:reduce_id],
                    vals[0] if len(vals) == 1 else vals,
                    *run_id[reduce_id:],
                    combined_data,
                ]
            )
        return combined_table

    def print(
        self,
        average_keys: dict[str, tuple[Any]] | None = None,
        display_keys: list[str] | None = None,
    ) -> None:
        """Print this table.

        Args:
            average_keys: Dict of key to reduce over and list of values to reduce
                over, e.g. {"seed": (0, 10, 20), "behavior_id": (0, 1, 2)}
            display_keys: List of data keys to display in table. Headers (attack
                name, exp id, etc.) are displayed by default. If None, display
                all data keys.
        """
        table = self.table
        headers = self.headers[:-1]  # Remove "data"
        if average_keys is not None:
            for key, vals in average_keys.items():
                table = self._reduce_by_key(table, headers, key, vals)
                # headers.remove(key)

        if display_keys is None:
            display_keys = self.headers
        skip_id = self.headers.index("behavior_id")

        if not table:
            print("No data")
            return

        # Turn data into multi-columns
        for i, row in enumerate(table):
            # Iterate over data (success, loss, best_loss, steps, etc.)
            mean_data = []
            new_keys = []
            for k, data in row[-1].items():
                if k not in display_keys:
                    continue
                if k == "success":
                    # We count success if it happens at least once
                    mean_data.append(np.mean([d.max() for d in data]))
                    new_keys.append(k)
                else:
                    # We just pick the last loss
                    _data = [d[-1] for d in data]
                    mean_data.extend(
                        [
                            np.mean(_data),
                            np.exp(np.mean(np.log(_data))),
                            np.median(_data),
                        ]
                    )
                    new_keys.extend([f"{k}_mean", f"{k}_geo", f"{k}_med"])
            table[i] = [*row[:skip_id], *row[skip_id + 1 : -1], *mean_data]
            # Convert tuple to string without comma
            table[i] = list(
                map(
                    lambda x: f"({' '.join(map(str, x))})"
                    if isinstance(x, tuple)
                    else x,
                    table[i],
                )
            )
        table.sort()
        headers = headers[:skip_id] + headers[skip_id + 1 :]
        headers += new_keys

        # Add comma
        for i, row in enumerate(table):
            table[i] = [f"{d}," for d in row]

        print(tabulate(table, headers=headers))


def main():
    # Gather jsonl results
    result_table = ResultTable()
    for attack in ATTACKS:
        log_path = Path("./results") / MODEL / attack
        for log_file in log_path.glob("**/*.jsonl"):
            assert "Toxicity" in log_file.name

            # Skip on old/wrong format
            if "lenNone" in str(log_file):
                continue

            behavior_id = int(log_file.stem.split("_")[1])
            exp_name = str(log_file.parent.name)
            total_steps = [s for s in exp_name.split("_") if "step" in s][0]
            total_steps = int(total_steps.replace("step", ""))
            with log_file.open("r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            # Skip unfinished sample
            if data[-1]["step"] != total_steps:
                continue

            data = {
                "success": [not d["passed"] for d in data],
                "loss": [d["loss"] for d in data[:-1]],
                "best_loss": [d["best_loss"] for d in data],
                "steps": [d["step"] for d in data],
            }
            for k, v in data.items():
                if isinstance(v, list):
                    data[k] = np.array(v, dtype=np.float32)
            exp_tokens = []
            for t in exp_name.split("_"):
                if "seed" in t:
                    seed = int(t.replace("seed", ""))
                else:
                    exp_tokens.append(t)
            exp_name = "_".join(exp_tokens)
            result_table.add_row([attack, exp_name, seed, behavior_id, data])

    # Print final table
    average_keys = {"behavior_id": BEHAVIOR_IDS}
    if SEEDS:
        average_keys["seed"] = SEEDS
    result_table.print(average_keys, display_keys=["best_loss", "success"])


if __name__ == "__main__":
    main()
