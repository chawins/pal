"""Run attack on LLM."""

import argparse
import gc
import json
import logging
import pprint
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
import yaml
from absl import app
from ml_collections import config_flags

from rules import scenarios
from src import attacks, models
from src.message import Message, Role
from src.models import utils as models_utils
from src.models.base import BaseModel
from src.servers.openai_server import kill_servers
from src.utils import argparser
from src.utils.log import setup_logger
from src.utils.suffix import build_prompt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()
_CONFIG = config_flags.DEFINE_config_file("config")

logger = logging.getLogger(__name__)

MAX_TOKENS = 150


def _process_init_suffix(
    atk_config, tokenizer, init_suffix: str, pad_str: str = " !"
) -> tuple[str, int]:
    """Process loaded init suffix."""

    def encode(text: str):
        return tokenizer(text, add_special_tokens=False).input_ids

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    suffix_ids = encode(init_suffix)
    init_suffix_len = len(suffix_ids)
    _init_len = atk_config.init_suffix_len
    if _init_len in (-1, init_suffix_len):
        return init_suffix, init_suffix_len

    logger.info(
        "Init suffix len (%d) does not match atk_config.init_suffix_len (%d).",
        init_suffix_len,
        _init_len,
    )
    num_trials = 0
    while _init_len != init_suffix_len:
        if num_trials >= _init_len:
            raise ValueError(
                "Cannot get init suffix to have the same length as atk_config."
                f"init_suffix_len ({_init_len}) even after {num_trials} trials!"
            )
        logger.info(
            "[trial %3d] BEFORE: init_suffix_len=%d, init_suffix=%s",
            num_trials + 1,
            init_suffix_len,
            init_suffix,
        )
        if _init_len > init_suffix_len:
            logger.info(
                '              Pad init suffix with "%s" to match '
                "init_suffix_len.",
                pad_str,
            )
            init_suffix += " !"
        else:
            logger.info(
                "              Truncate init suffix to match init_suffix_len."
            )
            suffix_ids = suffix_ids[:_init_len]
            init_suffix = decode(suffix_ids)
        suffix_ids = encode(init_suffix)
        init_suffix_len = len(suffix_ids)
        num_trials += 1
        logger.info(
            "            AFTER: init_suffix_len=%d, init_suffix=%s",
            init_suffix_len,
            init_suffix,
        )
    logger.info("New init_suffix: %s", init_suffix)
    return init_suffix, init_suffix_len


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


# def generate_batch(
#     args: argparse.Namespace,
#     model_fn: TransformersModel,
#     scenario: scenarios.BaseScenario,
#     justask_templates: dict[str, str] | None,
#     suffixes: list[str],
#     use_system_instructions: bool,
# ):
#     """Generate a batch of responses from the model."""
#     if use_system_instructions:
#         sys_message = Message(Role.SYSTEM, scenario.prompt)
#     else:
#         sys_message = Message(
#             Role.SYSTEM, models.SYSTEM_MESSAGES[args.system_message]
#         )

#     messages = [[sys_message]] * len(suffixes)
#     for i, suffix in enumerate(suffixes):
#         if justask_templates is None:
#             # Toxicity scenario
#             messages[i].append(
#                 Message(Role.USER, scenario.prompt + " " + suffix)
#             )
#         else:
#             raise NotImplementedError(
#                 "Not implemented for rule-following scenarios!"
#             )

#     # Get model's output
#     # TODO(future): add option to get token ids
#     response = model_fn(messages)
#     return response


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
    messages = "\n".join(messages_list)
    conv = f"Evaluated messages:\n{'-' * 80}\n{messages}\n{'-' * 80}"
    logger.debug(conv)
    return result.passed, messages_list


def main(argv):
    """Run attack LLM."""
    args = argparser.parse_args(argv)
    args, atk_config = argparser.auto_config(args, _CONFIG.value)
    setup_logger(args.verbose)
    logger.info("\n%s\n%s", "-" * 80, pprint.pformat(vars(args)))
    logger.info("\n%s\n%s", pprint.pformat(atk_config), "-" * 80)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # To reduce randomness, we set the following flags
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8 (before running the script)
    # torch.use_deterministic_algorithms(True)

    template_name, checkpoint_path = args.model.split("@")
    logger.info("Loading %s from %s...", template_name, checkpoint_path)
    loaded = models_utils.load_model_and_tokenizer(
        args.model,
        low_cpu_mem_usage=True,
        use_cache=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        system_message=args.system_message,
        use_system_instructions=args.use_system_instructions,
        temperature=args.temperature,
        max_tokens=max(
            MAX_TOKENS, atk_config.seq_len, atk_config.get("tune_seq_len", 0)
        ),
        num_api_processes=args.num_api_processes,
    )
    wrapped_model, tokenizer, suffix_manager = loaded
    logger.info("Finished loading model.")

    if args.init_suffix_path:
        # Load init suffix from path
        logger.info("Loading init suffix from %s...", args.init_suffix_path)
        with open(args.init_suffix_path, "r", encoding="utf-8") as file:
            init_suffixes = [json.loads(line) for line in file][0]
        assert len(init_suffixes) == len(
            scenarios.SCENARIOS[args.scenario].behaviors
        ), "Number of init suffixes does not match number of behaviors!"
    else:
        # Use init suffix from config
        init_suffixes = atk_config.adv_suffix_init

    # Get scenario and templates
    with open(args.justask_file, "r", encoding="utf-8") as f:
        justask = yaml.safe_load(f)
    with open(args.target_file, "r", encoding="utf-8") as f:
        targets = yaml.safe_load(f)

    not_allowed_tokens: torch.Tensor | None = (
        None
        if atk_config.allow_non_ascii
        else models_utils.get_nonascii_toks(tokenizer)
    )

    # Select behaviors to attack
    all_behaviors = scenarios.SCENARIOS[args.scenario].behaviors
    if args.behaviors is not None:
        try:
            # If behaviors are given as integer indices
            behaviors = [all_behaviors[int(i)] for i in args.behaviors]
        except ValueError:
            # If behaviors are given as strings
            behaviors = args.behaviors
    else:
        behaviors = all_behaviors

    # ====================== Main loop over behaviors ======================= #
    for i, behavior in enumerate(behaviors):
        logger.info("Behavior %d/%d: %s", i + 1, len(behaviors), behavior)

        _config = atk_config.copy_and_resolve_references()
        behavior_idx = all_behaviors.index(behavior)
        if isinstance(init_suffixes, str):
            init_suffix = init_suffixes
        else:
            init_suffix = init_suffixes[behavior_idx]

        # Process init suffix to have the desired length
        init_suffix, init_suffix_len = _process_init_suffix(
            _config, tokenizer, init_suffix
        )
        _config.init_suffix_len = init_suffix_len
        _config.adv_suffix_init = init_suffix

        full_name = f"{args.scenario}_{behavior_idx:02d}"
        _config.sample_name = full_name
        if full_name in targets:
            target_template = targets[full_name]
        else:
            # For Toxicity scenario: target_template is just "{prefix}"
            target_template = targets[full_name.split("_")[0]]
        justask_templates = justask.get(full_name)

        # TODO(future): This will fail with rule-following tasks
        scenario = scenarios.SCENARIOS[args.scenario](
            {"behavior": behavior, "target": target_template}
        )
        messages, target = build_messages_and_target(
            scenario,
            target_template,
            justask_templates=justask_templates,
            system_message=args.system_message,
            use_system_instructions=args.use_system_instructions,
        )
        m = "\n".join([f"{str(m)}" for m in messages])
        conv = f"Input messages:\n{'-' * 80}\n{m}\nTarget: {target}\n{'-' * 80}"
        logger.info(conv)
        # Show exact string input
        prompt = build_prompt(messages, template_name)
        logger.debug("Exact string input:\n%s", prompt)

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

        def build_messages_fn(system_message):
            # pylint: disable=cell-var-from-loop
            return build_messages_and_target(
                scenario,
                target_template,
                justask_templates=justask_templates,
                system_message=system_message,
                use_system_instructions=args.use_system_instructions,
            )

        # Set up attacker
        logger.info("Setting up attacker...")
        attack: attacks.base.BaseAttack = attacks.setup_attacker(
            _config,
            wrapped_model=wrapped_model,
            tokenizer=tokenizer,
            suffix_manager=suffix_manager,
            eval_fn=eval_suffix,
            not_allowed_tokens=not_allowed_tokens,
            use_system_instructions=args.use_system_instructions,
            build_messages_fn=build_messages_fn,
        )
        # Run the attack on one behavior
        adv_results = attack.run(messages, target)
        # Generate full output of the best adv suffix
        wrapped_model.max_tokens = 512
        passed, messages = eval_suffix(adv_results.best_suffix)
        m = "\n".join([f"{str(m)}" for m in messages])
        final_log = f"""Final result:\n{'-' * 80}
Target: {target}
{m}
best_loss={adv_results.best_loss:.4f}
passed={passed}
{'-' * 80}"""
        logger.info(final_log)
        logger.info("Finished behavior %d/%d", i + 1, len(behaviors))
        logger.info("=" * 80)
        attack.log(
            log_dict={
                "best_loss": adv_results.best_loss,
                "passed": passed,
                "suffix": adv_results.best_suffix,
                "generated": messages[-1],  # last message
                "finished": True,
            },
        )
        attack.cleanup()
        del attack
        gc.collect()
    kill_servers()
    logger.info("Attack finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    app.run(main)
