"""Parse command line arguments."""

import argparse
from pathlib import Path

from ml_collections import ConfigDict
from rules import scenarios

from src import models


def parse_args(argv) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM attacks.")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-2@~/data/models/Llama-2-7b-chat-hf",
        help="template_name@checkpoint_path",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./results/",
        help="Base directory for logging results (defaults to ./results/).",
    )
    parser.add_argument("--justask_file", type=str, default="data/justask.yaml")
    parser.add_argument("--target_file", type=str, default="data/targets.yaml")
    parser.add_argument(
        "--scenario",
        type=str,
        default="Toxicity",
        choices=scenarios.SCENARIOS.keys(),
        help="Scenario to evaluate (defaults to 'Toxicity').",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Names or indices of behaviors to evaluate in the scenario "
            "(defaults to None = all)."
        ),
    )
    parser.add_argument(
        "--use_system_instructions",
        action="store_true",
        default=False,
        help="If True, present instructions as a system message.",
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default=None,
        choices=models.SYSTEM_MESSAGES.keys(),
        help=(
            "System message to model, if not using --system-instructions. "
            "Defaults to None = no system message."
        ),
    )
    parser.add_argument(
        "--disable_eval",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If True, set logger to debug mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=20, help="Random seed (default: 20)."
    )
    parser.add_argument(
        "--custom_name",
        type=str,
        default="",
        help="Custom experiment name to append to log dir.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Target model generation temperature (default: 0.0).",
    )
    parser.add_argument(
        "--init_suffix_path",
        type=str,
        default="",
        help=(
            "Path to load init suffixes from. WARNING: This overwrites "
            "adv_suffix_init in attack config."
        ),
    )
    parser.add_argument(
        "--num_api_processes",
        type=int,
        default=8,
        help="Number of processes to spawn to query an LLM API in parallel.",
    )
    args = parser.parse_args(argv[1:])
    return args


def auto_config(
    args: argparse.Namespace, atk_config: ConfigDict
) -> tuple[argparse.Namespace, ConfigDict]:
    """Set up log dir and auto-generate experiment parameters.

    Args:
        args: Command line arguments.
        atk_config: Attack config.

    Returns:
        args: Modified command line arguments.
        atk_config: Modified attack config.
    """
    args.log_dir = str(
        Path(args.log_dir or "results") / args.model.split("/")[-1]
    )
    with atk_config.unlocked():
        atk_config.seed = args.seed
        atk_config.log_dir = args.log_dir
        atk_config.custom_name = args.custom_name
        atk_config.sample_name = ""
    return args, atk_config
