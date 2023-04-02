from types import ModuleType
from typing import DefaultDict, Dict, List, Union

from torch import Tensor

from modules.extra_networks import ExtraNetworkParams
from modules.prompt_parser import MulticondLearnedConditioning, ScheduledPromptConditioning
from modules.sd_samplers_compvis import VanillaStableDiffusionSampler
from modules.sd_samplers_kdiffusion import KDiffusionSampler

Sampler = Union[KDiffusionSampler, VanillaStableDiffusionSampler]
Cond = MulticondLearnedConditioning
Uncond = List[List[ScheduledPromptConditioning]]
ExtraNetworkData = DefaultDict[str, List[ExtraNetworkParams]]

# 'c_crossattn': Tensor    # prompt cond
# 'c_concat':    Tensor    # latent mask
CondDict = Dict[str, Tensor]

__all__ = [
    "ModuleType",
    "Sampler",
    "Cond",
    "Uncond",
    "ExtraNetworkData",
    "CondDict",
    "MulticondLearnedConditioning",
    "ExtraNetworkParams",
    "ScheduledPromptConditioning",
]
