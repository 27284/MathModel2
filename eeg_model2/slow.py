"""低自由度绝对实验时间趋势；仅用训练 Trial 的刺激前原始样本校准。"""
from dataclasses import dataclass, asdict
import numpy as np
from .data import recenter

SLOW_SETTINGS = ('none', 'linear', 'quadratic', 'natural_spline_df4')


def slow_design(time, setting, center, scale, knots):
    x = (np.asarray(time) - center) / scale
    if setting == 'none':
        return np.zeros((*x.shape, 0))
    columns = [np.ones_like(x), x]
    if setting == 'quadratic':
        columns.append(x*x)
    elif setting == 'natural_spline_df4':
        # Restricted cubic spline: 4 knots -> intercept + linear + 2 nonlinear df.
        k = np.asarray(knots)
        def d(a):
            return (np.maximum(x-a, 0)**3 - np.maximum(x-k[-1], 0)**3)/(k[-1]-a)
        columns.extend(d(a)-d(k[-2]) for a in k[:-2])
    elif setting != 'linear':
        raise ValueError(f'未知慢趋势: {setting}')
    return np.stack(columns, axis=-1)


@dataclass
class SlowFit:
    setting: str
    center: float
    scale: float
    knots: list
    coef: np.ndarray

    def predict(self, onsets, t, baseline=True):
        design = slow_design(np.asarray(onsets)[:, None] + t, self.setting,
                             self.center, self.scale, self.knots)
        result = (design @ self.coef).transpose(0, 2, 1)
        return recenter(result, t) if baseline else result

    def to_dict(self):
        result = asdict(self)
        result['coef'] = self.coef.tolist()
        return result


def fit_slow(train_epochs, onsets, setting, t):
    """onsets 单位秒。传入的数组必须已限定为训练样本；不读取测试范围或节点。"""
    if setting not in SLOW_SETTINGS:
        raise ValueError(setting)
    epochs, onsets, t = np.asarray(train_epochs), np.asarray(onsets), np.asarray(t)
    pre = (t >= -1.) & (t < -.2)
    if not pre.any() or not len(onsets):
        raise ValueError('慢趋势需训练刺激前 -1 至 -0.2 秒原始样本')
    times = (onsets[:, None] + t[pre]).ravel()
    center, scale = float(times.mean()), max(float(times.std()), 1.)
    knots = np.quantile((times-center)/scale, [0., 1/3, 2/3, 1.]).tolist()
    x = slow_design(times, setting, center, scale, knots)
    y = epochs[..., pre].transpose(0, 2, 1).reshape(-1, epochs.shape[1])
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return SlowFit(setting, center, scale, knots, coef)


def apply_slow(model, epochs, onsets, t):
    """固定训练模型，只减去与原轻处理相同基线校正后的慢趋势。"""
    component = model.predict(onsets, t)
    return np.asarray(epochs) - component, component
