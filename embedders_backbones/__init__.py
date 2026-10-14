"""embedders_backbones (NEMBA) - self-supervised sequence encoders for any
population time series (spikes, pose, segmentation masks, fiber photometry, ...),
with time-series-aware decoding and visualisation. CPU-first.

    from embedders_backbones import Embedder, list_backbones
    emb = Embedder("tcn", speed="fast").fit_spikes(spike_times)
    emb.decode({"x": x, "speed": v})
    emb.run("out", behavior={"x": x, "speed": v})
"""

from .core import Embedder
from .compare import compare_embedders, Comparison
from .backbones import list_backbones, label, make_model, BACKBONES
from .training import SPEED_PRESETS, OBJECTIVES
from . import data, analysis, backbones, compare, decoding, training, synthetic


def __getattr__(name):
    # viz pulls in matplotlib: import it lazily (keeps `import` fast for CLI / workers)
    if name == "viz":
        from . import viz
        return viz
    raise AttributeError(name)


__all__ = ["Embedder", "compare_embedders", "Comparison", "list_backbones", "label",
           "make_model", "BACKBONES", "SPEED_PRESETS", "OBJECTIVES", "data", "analysis",
           "viz", "backbones", "compare", "decoding", "training", "synthetic"]
__version__ = "0.2.0"
