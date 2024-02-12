"""Attacks module."""

from ml_collections import ConfigDict

from src.attacks.base import BaseAttack
from src.attacks.gcg import GCGAttack
from src.attacks.pal import PalAttack
from src.attacks.ral import RalAttack

_ATTACKS_DICT = {"gcg": GCGAttack, "ral": RalAttack, "pal": PalAttack}


def setup_attacker(atk_config: ConfigDict, **kwargs) -> BaseAttack:
    """Set up the attacker.

    Args:
        atk_config: A dictionary containing the configuration for the attacker.

    Raises:
        ValueError: If the attack is not supported.

    Returns:
        An attacker object.
    """
    attack_name = atk_config.name
    if attack_name not in _ATTACKS_DICT:
        raise ValueError(f"Attack {attack_name} not supported")
    return _ATTACKS_DICT[attack_name](atk_config, **kwargs)
