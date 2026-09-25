"""V3 小型时间形变组；神经子模型不含 persistent step。"""
import numpy as np
from scipy.signal import lfilter, sosfiltfilt
from .data import recenter

WARP_GROUPS = ('base', 'shift40', 'shift80', 'stretch', 'shift_stretch')


def shift_basis(basis, t, latency):
    return stretch_basis(basis, t, 1., latency)


def stretch_basis(basis, t, stretch, latency=0.):
    if stretch <= 0:
        raise ValueError('stretch must be positive')
    basis = np.asarray(basis)
    result = [np.interp((t-latency)/stretch, t, row, left=0., right=row[-1])
              for row in basis.reshape(-1, len(t))]
    return np.array(result).reshape(basis.shape)


class BasisBank:
    def __init__(self, source):
        self.source, self.t, self.cache = source, source.t, {}

    def get(self, duration, family='wc', warp='base', lp750=False):
        key = (round(float(duration), 10), family, warp, lp750)
        if key in self.cache:
            return self.cache[key]
        if warp not in WARP_GROUPS:
            raise ValueError(warp)
        bank, t = self.source, self.t
        if family == 'temporal':
            # Ordinary smooth temporal basis, fixed without EEG or label estimates.
            common = np.array([np.exp(-.5*((t-c)/.10)**2) for c in (.08, .25, .45, .7)])
            contrast = np.array([np.exp(-.5*((t-c)/.06)**2) for c in (.15, .30, .45)])
            result = (recenter(common, t), recenter(contrast, t))
        elif family == 'wc':
            q = bank.sim.sources(bank.theta, [-1], [duration], [1], states=True)[0, :, 2]
            sources = (q.mean(axis=0), (q[0]-q[1])/2)
            result = []
            for component, source in enumerate(sources):
                group = [source]
                # Different families: common q/LP80/LP250, contrast q/LP80.
                taus = [.08, .25] if component == 0 else [.08]
                if lp750:
                    taus.append(.75)
                for tau in taus:
                    a = np.exp(-1/(bank.cfg.fs*tau))
                    group.append(lfilter([1-a], [1, -a], source))
                base = np.array(group)
                variants = [base]
                shifts = (-.04, .04) if warp == 'shift40' else (-.08, .08) if warp == 'shift80' else (-.08, -.04, .04, .08) if warp == 'shift_stretch' else ()
                variants.extend(shift_basis(base, bank.pt, d) for d in shifts)
                if warp in ('stretch', 'shift_stretch'):
                    variants.extend(stretch_basis(base, bank.pt, s) for s in (.7, 1.4))
                values = sosfiltfilt(bank.sim.sos, np.concatenate(variants), axis=-1)[..., bank.core]
                result.append(recenter(values, t))
        else:
            raise ValueError(family)
        # Scaling depends only on fixed simulated bases and known duration, never EEG.
        self.cache[key] = tuple(b / np.maximum(np.sqrt(np.mean(b[:, t >= 0]**2, axis=1, keepdims=True)), 1e-12) for b in result)
        return self.cache[key]


def build_common_basis(bank, duration, **kwargs):
    return bank.get(duration, **kwargs)[0]


def build_contrast_basis(bank, duration, **kwargs):
    return bank.get(duration, **kwargs)[1]
