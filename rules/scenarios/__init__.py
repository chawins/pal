from typing import Dict

from rules.scenarios.advbench import AdvBench, AdvBenchAll, AdvBenchString
from rules.scenarios.scenario import BaseScenario
from rules.scenarios.toxicity import Toxicity, ToxicityAll

SCENARIO_CLASSES = [
    Toxicity,
    ToxicityAll,
    AdvBench,
    AdvBenchAll,
    AdvBenchString,
]

SCENARIOS: Dict[str, BaseScenario] = {s.__name__: s for s in SCENARIO_CLASSES}
