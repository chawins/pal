"""Submit slurm jobs for running attacks."""

import itertools
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module

from ml_collections import ConfigDict

from src import attacks
from src.utils.argparser import auto_config

SCENARIO = "AdvBenchAll"  # Toxicity, ToxicityAll, AdvBench, AdvBenchAll
WANDB_NAME = "chawins"  # NOTE: Not actually used
# Splits data (50 samples for Toxicity) to run attacks in parallel
NUM_GPUS = 1
HOURS = 4
MINS = 0
SAMPLE_IDS = tuple(range(50))
NUM_DATA_SHARDS = len(SAMPLE_IDS)


@dataclass
class FakeArgs:
    """Fake args for running attacks."""

    seed: int
    custom_name: str
    log_dir: str
    model: str
    behaviors: str | None = None
    verbose: bool = False
    temperature: float = 0.0
    init_suffix_path: str = ""


def main():
    """Submit slurm jobs for transfer attacks evaluation."""
    num_jobs = 0
    hyp = {
        "model": ["llama-2@~/data/models/Llama-2-7b-chat-hf"],
        "add_space": [True],  # TODO: orig
        "seed": [20],
        "batch_size": [128],
        "mini_batch_size": [128],
        "init_suffix_len": [20],
        "num_steps": [2000],
        "log_freq": [1],
        "early_stop": [False],
        "attack": ["pal"],  # "gcg", "ral", "pal"
        "fixed_params": [True],
        "skip_mode": ["visited"],  # visited, seen, none # TODO: orig
        "loss_func": ["ce-one"],  # TODO: orig
        "cw_margin": [1e-3],
        "seq_len": [50],
        "max_queries": [25000],
        "sample_mode": ["rnd"],  # TODO: orig
        "custom_name": ["init"],
        "init_suffix_path": [
            f"data/init_suffix/{SCENARIO}/vicuna-7b-v1.5-16k_gcg_len20.jsonl",
        ],
        # ============================== proxy ============================== #
        "finetune": [False],
        # "proxy_loss_func": ["ce-all"],
        "proxy_tune_bs": [32],
        "peft": ["noembed"],  # "none", "noembed", "lora"
        "proxy_dtype": ["bfloat16"],
        "proxy_device": [(0,)],  # EDIT: GPT, Cohere
        "proxy_model": [
            (
                "vicuna_v1.1@~/data/models/vicuna-7b-v1.5-16k",
            )
        ],
        "log_target_loss": [True],
        "tune_on_past_samples": [False],
        "proxy_tune_period": [1],
        "num_target_query_per_step": [32],
    }

    # Shard behaviors to run in parallel
    behaviors = []
    for i in range(NUM_DATA_SHARDS):
        ids = [
            SAMPLE_IDS[idx]
            for idx in range(i, len(SAMPLE_IDS), NUM_DATA_SHARDS)
        ]
        behaviors.append(ids)
    hyp["behaviors"] = behaviors

    keys, values = zip(*hyp.items())
    hyp = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for kwargs in hyp:
        print(kwargs)
        run_once(**kwargs)
        num_jobs += 1
    print(f"All {num_jobs} jobs submitted successfully!")


def run_once(
    init_suffix_len: int = 20,
    attack: str = "gcg",
    system_message: str | None = "llama_default",
    custom_name: str = "",
    **kwargs,
):
    """Submit a single slurm job."""
    if "llama" in kwargs.get("proxy_model", [""])[0]:
        kwargs["proxy_system_message"] = ("llama_default",)
    elif "vicuna" in kwargs.get("proxy_model", [""])[0]:
        kwargs["proxy_system_message"] = ("vicuna_default",)
    if "seq_len" in kwargs:
        kwargs["tune_seq_len"] = kwargs["seq_len"]

    adv_suffix_init = " ".join(["!"] * init_suffix_len)

    # Dynamically import desired attack config
    config_mod = import_module(f"configs.{attack}")
    config: ConfigDict = config_mod.get_config()
    config_dict = {
        "init_suffix_len": init_suffix_len,
        "adv_suffix_init": adv_suffix_init,
    }
    args_dict = {"log_dir": "./results/"}
    # Populate config and args with kwargs
    for key, val in kwargs.items():
        if key in config:
            config_dict[key] = val
            default_val = getattr(config, key)
            assert default_val is None or isinstance(val, type(default_val)), (
                val,
                type(default_val),
            )
        elif key in FakeArgs.__annotations__:
            args_dict[key] = val
        else:
            print("Ignoring key:", key)
    config.update(**config_dict)

    # Get string command line args
    key_val_pairs = []
    for key, val in config.iteritems():
        if key not in config_dict:
            continue
        if isinstance(val, (str, tuple)):
            val = f'"{val}"'
        key_val_pairs.append(f"    --config.{key}={val} \\")
    config_command = "\n".join(key_val_pairs)
    args_kv_pairs = []
    for key, val in args_dict.items():
        if isinstance(val, bool):
            args_kv_pairs.append(f"    --{key} \\")
        else:
            if isinstance(val, str):
                val = f'"{val}"'
            elif isinstance(val, list):
                val = " ".join([str(v) for v in val])
            args_kv_pairs.append(f"    --{key} {val} \\")
    if system_message:
        args_kv_pairs.append(f"    --system_message {system_message} \\")
    if custom_name:
        args_kv_pairs.append(f"    --custom_name {custom_name} \\")
    arg_command = "\n".join(args_kv_pairs)

    # Set up attack to get name
    args = FakeArgs(custom_name=custom_name, **args_dict)
    args, config = auto_config(args, config)
    attack: attacks.base.BaseAttack = attacks.setup_attacker(
        config,
        wrapped_model=None,
        tokenizer=None,
        suffix_manager=None,
        eval_fn=None,
        not_allowed_tokens=None,
        lazy_init=True,
        build_messages_fn=None,
    )
    output_name = f"{args_dict['model'].split('/')[-1]}_{str(attack)}"
    if custom_name:
        output_name += f"_{custom_name}"
    if kwargs.get("behaviors"):
        output_name += f"_shard{int(kwargs['behaviors'][0]):02d}"

    base_script = f"""#!/bin/bash
#SBATCH --job-name=prompt-inject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={max(8, 8 * NUM_GPUS)}
#SBATCH --gpus={NUM_GPUS}
#SBATCH --time={HOURS}:{MINS}:00
#SBATCH --output %j-{output_name}.log

source $HOME/.bashrc
source activate prompt-inject

echo #SCRIPT_PATH#

export WANDB_MODE=disabled

python -u main.py \\
    --config="./configs/{attack.name}.py" \\
{config_command}
    -- \\
    --scenario {SCENARIO} \\
{arg_command}
"""
    # --num_api_processes {kwargs['num_target_query_per_step']} \\
    script = base_script
    hash_id = f"{hash(script) % 100000:05d}"
    script_path = f"_tmp/{output_name}.sh".replace("#HASH#", hash_id)
    script = script.replace("#SCRIPT_PATH#", script_path)
    script = script.replace("#HASH#", hash_id)

    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script)
    print(f"Submitting job at {script_path}")
    output = subprocess.run(["sbatch", script_path], check=True)
    if output.returncode != 0:
        print("Failed to submit job!")
        sys.exit(1)


if __name__ == "__main__":
    main()
