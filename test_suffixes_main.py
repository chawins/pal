"""Test saved suffixes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from openai import OpenAI
from rules import scenarios
from src import models
from src.message import Message, Role
from src.models import SYSTEM_MESSAGES
from src.models import utils as models_utils
from src.models.base import BaseModel
from tqdm import tqdm

CLIENT = OpenAI()


def build_messages_and_target(
    scenario: scenarios.BaseScenario,
    target_template: str,
    justask_templates: dict[str, str] | None = None,
    system_message: str | None = None,
    use_system_instructions: bool = False,
):
    """Create messages and target for a scenario."""
    var_dict = asdict(scenario.p)
    target = target_template.format(**var_dict)
    if justask_templates is None:
        messages = [
            Message(Role.SYSTEM, models.SYSTEM_MESSAGES[system_message]),
            Message(Role.USER, scenario.prompt),
        ]
    else:
        var_dict["first_message"] = justask_templates["first_message"].format(
            **var_dict
        )
        first_user_message = justask_templates["messages"][0].format(**var_dict)
        if use_system_instructions:
            messages = [
                Message(Role.SYSTEM, scenario.prompt),
                Message(Role.USER, first_user_message),
            ]
        else:
            messages = [
                Message(Role.SYSTEM, models.SYSTEM_MESSAGES[system_message]),
                Message(Role.USER, scenario.prompt),
                Message(Role.ASSISTANT, scenario.initial_response),
                Message(Role.USER, first_user_message),
            ]
    return messages, target


def evaluate_suffix(
    args: argparse.Namespace,
    model_fn: BaseModel,
    scenario: scenarios.BaseScenario,
    justask_templates: dict[str, str] | None,
    suffix: str,
    use_system_instructions: bool,
) -> tuple[bool, list[str]]:
    """Evaluate if model's generation follows the rules."""
    var_dict = asdict(scenario.p)
    if use_system_instructions:
        messages = [Message(Role.SYSTEM, scenario.prompt)]
    else:
        messages = [
            Message(Role.SYSTEM, models.SYSTEM_MESSAGES[args.system_message])
        ]

    if justask_templates is None:
        # Toxicity scenario
        messages.append(Message(Role.USER, f"{scenario.prompt} {suffix}"))
        # Get model's output
        response = model_fn(messages)
        content = models.concat_stream(response)
        # Evaluate if the rule is followed
        messages.append(Message(Role.ASSISTANT, content))
        result = scenario.evaluate(
            messages, system_instructions=use_system_instructions
        )
    else:
        # Other rule-following scenarios
        messages.append(Message(Role.USER, scenario.prompt))
        messages.append(Message(Role.ASSISTANT, scenario.initial_response))
        var_dict["first_message"] = justask_templates["first_message"].format(
            **var_dict
        )
        # templates may contain multiple messages (longer conversation), but
        # suffix is only applied to the first message.
        for i, m in enumerate(justask_templates["messages"]):
            if i == 0:
                messages.append(
                    Message(Role.USER, m.format(**var_dict) + suffix)
                )
            else:
                messages.append(Message(Role.USER, m.format(**var_dict)))
            response = model_fn(messages)
            content = models.concat_stream(response)
            messages.append(Message(Role.ASSISTANT, content))
            result = scenario.evaluate(
                messages, system_instructions=use_system_instructions
            )
            if not result.passed:
                break
    messages_list = [f"{str(m)}" for m in messages]
    return result.passed, messages_list


def openai_content_mod(gen_output: str):
    response = CLIENT.moderations.create(input=gen_output)
    output = response.results[0]
    if not output.flagged:
        return None
    scores_dict = {}
    category_scores = output.category_scores.dict()
    for category, flagged in output.categories.dict().items():
        if flagged:
            scores_dict[category] = category_scores[category]
    return scores_dict


def main(args):
    # e.g., results/Llama-2-7b-chat-hf/ral/len20_100000query_bs32_seed20_l50_cw-one-0.001_static_visited_uniform_t1.0_c1
    result_path = Path(args.result_path)
    if not result_path.is_dir():
        raise NotADirectoryError(f"{result_path} is not a directory!")

    model_name = result_path.parent.parent.name
    print(f"model_name: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model and the tokenizer
    model_path = str(Path(f"~/data/models/{model_name}").expanduser())

    wrapped_model = None
    if args.regenerate_outputs:
        if "Llama" in model_name:
            template = "llama-2"
            args.system_message = "llama_default"
        elif "vicuna" in model_name:
            template = "vicuna_v1.1"
            args.system_message = "vicuna_default"
        elif "gpt" in model_name:
            template = "chatgpt"
            args.system_message = "helpful"
        elif "openchat-3.5-1210" in model_name:
            template = "openchat_3.5"
            args.system_message = "helpful"
        else:
            raise ValueError(f"Unknown model {model_name}!")
        wrapped_model = models_utils.load_model_and_tokenizer(
            f"{template}@{model_path}",
            low_cpu_mem_usage=True,
            use_cache=True,
            device=device,
            system_message=args.system_message,
            use_system_instructions=args.use_system_instructions,
            temperature=0.0,
            max_tokens=args.max_tokens,
            num_api_processes=1,
        )[0]

    # Get scenario and templates
    with open(args.justask_file, "r", encoding="utf-8") as f:
        justask = yaml.safe_load(f)
    with open(args.target_file, "r", encoding="utf-8") as f:
        targets = yaml.safe_load(f)

    # Select behaviors to attack
    sc = scenarios.SCENARIOS[args.scenario]
    if args.behaviors is None:
        behaviors = sc.behaviors
        behavior_ids = range(len(behaviors))
    else:
        behaviors = []
        for b in args.behaviors:
            behaviors.append(sc.behaviors[b])
        behavior_ids = args.behaviors

    num_success = 0

    # ====================== Main loop over behaviors ======================= #
    for i, behavior in zip(behavior_ids, behaviors):
        full_name = f"{args.scenario}_{i:02d}"
        print(f"Running on {full_name}...")
        if full_name in targets:
            target_template = targets[full_name]
        else:
            # For Toxicity scenario: target_template is just "{prefix}"
            target_template = targets[full_name.split("_")[0]]
        justask_templates = justask.get(full_name)

        log_file = result_path / f"{args.scenario}_{i:02d}.jsonl"
        with log_file.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        # TODO(future): This will fail with rule-following tasks
        scenario = sc({"behavior": behavior, "target": target_template})
        # messages, target = build_messages_and_target(
        #     scenario,
        #     target_template,
        #     justask_templates=justask_templates,
        #     system_message=args.system_message,
        #     use_system_instructions=args.use_system_instructions,
        # )

        def eval_suffix(adv_suffix):
            # pylint: disable=cell-var-from-loop
            return evaluate_suffix(
                args,
                wrapped_model,
                scenario,
                justask_templates,
                adv_suffix,
                args.use_system_instructions,
            )

        outputs, filtered_outputs = [], []
        is_flagged_by_openai = False
        is_any_success = False
        print(f"Running inference on {len(lines)} suffixes...")
        for j, line in enumerate(tqdm(lines)):
            if line["queries"] > args.max_queries:
                break
            if line["step"] > args.max_step:
                break

            if args.regenerate_outputs:
                passed, messages = eval_suffix(line["suffix"])
                assert passed == line["passed"]
                if j != 0:
                    messages = messages[1:]
                m = "\n".join(messages)
                outputs.append(
                    f">>> {'FAIL' if passed else 'SUCCEED'}: step="
                    f"{line['step']}, queries={line['queries']}\n{m}"
                )
            else:
                passed = line["passed"]
                prompt = ""
                if j == 0:
                    sys_msg = SYSTEM_MESSAGES[args.system_message]
                    prompt += f"[System]: {sys_msg}\n"
                prompt += f"[User]: {behavior} {line['suffix']}\n"
                output = (
                    f">>> {'FAIL' if passed else 'SUCCEED'}: step="
                    f"{line['step']}, queries={line['queries']}\n"
                    f"{prompt}{line['generated'].strip()}"
                )
                if args.openai_filter:
                    scores_dict = openai_content_mod(line["generated"])
                    if scores_dict is not None:
                        filtered_outputs.append(output + f"\n{scores_dict}")
                    is_flagged_by_openai |= scores_dict is not None
                outputs.append(output)
            is_any_success |= not passed
        num_success += is_any_success
        # print(num_success)

        if args.openai_filter and is_flagged_by_openai:
            outputs = filtered_outputs

        text = f"\n{'=' * 80}\n\n".join(outputs)
        text_file = str(log_file).replace(".jsonl", ".txt")
        text_file = "/".join(["gen_outputs", *text_file.split("/")[1:]])
        print(f"Writing to file: {text_file}")
        text_file = Path(text_file)
        text_file.parent.mkdir(parents=True, exist_ok=True)
        with text_file.open("w", encoding="utf-8") as f:
            f.write(text)

    print(f"ASR: {num_success / len(behaviors):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    parser.add_argument(
        "--scenario",
        type=str,
        default="AdvBenchAll",
        choices=scenarios.SCENARIOS.keys(),
        help="Scenario to evaluate (defaults to 'AdvBenchAll').",
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
        "--behaviors",
        nargs="+",
        type=int,
        default=None,
        help="Names or indices of behaviors to evaluate in the scenario (defaults to None = all).",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=256, help="Max tokens to generate."
    )
    parser.add_argument("--max_queries", type=int, default=10000000)
    parser.add_argument("--max_step", type=int, default=10000)
    parser.add_argument("--justask_file", type=str, default="data/justask.yaml")
    parser.add_argument("--target_file", type=str, default="data/targets.yaml")
    parser.add_argument(
        "--regenerate-outputs",
        action="store_true",
        help="Run inference on target model against all suffixes.",
    )
    parser.add_argument(
        "--openai_filter",
        action="store_true",
        help="Filter out outputs flagged by OpenAI.",
    )
    _args = parser.parse_args()
    main(_args)
