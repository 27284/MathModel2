from dataclasses import dataclass,asdict


@dataclass(frozen=True)
class Config:
    seed: int = 20260925
    fs: int = 256
    folds: int = 5
    pre: float = .2
    post: float = .8
    padding: float = 1.
    clip_level: float = 999.9
    lowpass: float = 30.
    filter_order: int = 4
    min_weight: float = .05
    huber_c: float = 1.345
    source_rho: float = .6
    synaptic_tau: float = .02
    bootstrap: int = 1000

    def to_dict(self):return asdict(self)
