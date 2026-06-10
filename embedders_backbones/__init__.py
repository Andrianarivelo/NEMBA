"""embedders_backbones - standalone self-supervised sequence encoders for any
population time series (spikes, pose, segmentation masks, fiber photometry, ...).

    from embedders_backbones import Embedder, list_backbones
    Embedder("dtc", n_states=8).fit_spikes(spike_times).run("out")
"""

from .core import Embedder
from .backbones import list_backbones, label, make_model, BACKBONES
from . import data, analysis, viz, backbones

__all__ = ["Embedder", "list_backbones", "label", "make_model", "BACKBONES",
           "data", "analysis", "viz", "backbones"]
__version__ = "0.1.0"
