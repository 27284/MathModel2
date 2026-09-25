"""读取原始通道；固定处理在每个不相交的扩展 Trial 内独立进行。"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

CHANNELS = ('Fz', 'F3', 'F4')


def recenter(x, t):
    return x - x[..., t < 0].mean(axis=-1, keepdims=True)


def onsets_and_ends(cue):
    starts = np.flatnonzero((cue != 0) & np.r_[True, cue[1:] != cue[:-1]])
    ends = np.array([s + np.flatnonzero(np.r_[cue[s:] != cue[s], True])[0]
                     for s in starts], dtype=int)
    return starts, ends


def prestim_metrics(raw, padded_t):
    """输入 (..., channel, time)；四项指标不读取刺激后数据。"""
    base = raw[..., (padded_t >= -51/256) & (padded_t < 0)]
    long = raw[..., (padded_t >= -1.) & (padded_t < 0)]
    mask = (padded_t >= -1.) & (padded_t < -.2)
    t = padded_t[mask] - padded_t[mask].mean()
    q1 = 1.4826 * np.median(np.abs(base - np.median(base, axis=-1, keepdims=True)), axis=-1)
    h = base.shape[-1] // 2
    q2 = np.abs(base[..., h:].mean(axis=-1) - base[..., :h].mean(axis=-1))
    q3 = np.abs(np.diff(long, axis=-1)).max(axis=-1)
    q4 = np.abs(np.einsum('...t,t->...', raw[..., mask], t) / (t @ t))
    return np.stack([q1, q2, q3, q4], axis=-1)


@dataclass
class Record:
    name: str
    task: int
    labels: np.ndarray
    duration: np.ndarray
    folds: np.ndarray
    onsets: np.ndarray
    valid: np.ndarray
    reasons: list
    t: np.ndarray
    padded_t: np.ndarray
    core: np.ndarray
    raw_epochs: np.ndarray
    padded: np.ndarray
    light: np.ndarray
    ecg: np.ndarray
    metrics: np.ndarray
    prestim: np.ndarray
    background: list
    background_ecg: list
    audit: dict


def read_record(path, cfg):
    path = Path(path)
    mat = loadmat(path)
    d = np.asarray(mat['data'], dtype=float)
    fs = float(np.asarray(mat['SampleRate']).squeeze())
    names = [str(np.asarray(v).squeeze()) for v in mat['DataLabel'].ravel()]
    if d.ndim != 2 or d.shape[0] != 10 or fs != cfg.fs or names[:3] != list(CHANNELS):
        raise ValueError(f'{path.name}: 不支持的数据形状、采样率或通道名')
    # 不读取机器处理列作为模型输入。
    if not np.isfinite(d[[0, 1, 2, 6, 7, 9]]).all():
        raise ValueError(f'{path.name}: 原始信号/事件/时间戳包含非有限值')
    if not np.allclose(np.diff(d[9]), 1/fs, atol=1e-8, rtol=0):
        raise ValueError(f'{path.name}: 时间戳不连续')
    if not set(np.unique(d[7])).issubset({-1, 0, 1}):
        raise ValueError(f'{path.name}: 未定义的提示代码')
    on, end = onsets_and_ends(d[7])
    n = len(on)
    if n < cfg.folds * 4:
        raise ValueError('每个时间块的试次数过少')
    offsets = np.arange(-round(cfg.pre*fs), round(cfg.post*fs)+1)
    pad = np.arange(offsets[0]-round(cfg.padding*fs), offsets[-1]+round(cfg.padding*fs)+1)
    core = np.searchsorted(pad, offsets)
    t, pt = offsets/fs, pad/fs
    # 原始 Trial 上定义块，剔除坏 Trial 后也不改变时间块。
    folds = np.minimum(cfg.folds-1, np.arange(n)*cfg.folds//n)
    raw = np.zeros((n, 3, len(pad)))
    eeg = np.zeros_like(raw)
    ecg = np.zeros((n, len(pad)))
    valid = np.ones(n, bool)
    reasons = [''] * n
    lp = butter(cfg.filter_order, cfg.lowpass, fs=fs, output='sos')
    ep = butter(2, [1, 25], btype='bandpass', fs=fs, output='sos')
    for i, s in enumerate(on):
        ix = s + pad
        if ix[0] < 0 or ix[-1] >= d.shape[1]:
            valid[i], reasons[i] = False, 'padded_window_out_of_bounds'
            continue
        raw[i] = d[:3, ix]
        if np.any(np.abs(raw[i]) >= cfg.clip_level):
            valid[i], reasons[i] = False, 'raw_clipping_in_padded_window'
        eeg[i] = sosfiltfilt(lp, raw[i], axis=-1)
        ecg[i] = sosfiltfilt(ep, d[6, ix])
    # 当前实验间隔>8 s，扩展窗仅3 s；未来更密集数据必须先隔离，不能静默泄漏。
    if n > 1 and np.any(np.diff(on) <= pad[-1]-pad[0]):
        raise ValueError('扩展Trial有重叠，请增加训练/测试隔离后再使用本程序')
    light = recenter(eeg[..., core], t)
    metrics = prestim_metrics(raw, pt)
    # 负对照只使用原始刺激前样本；不用零相位滤波、低维重建后的刺激前值。
    pre_raw = raw[..., (pt >= -.2) & (pt < 0)]
    h = pre_raw.shape[-1]//2
    means = pre_raw.mean(axis=-1)
    prestim = np.column_stack([means[:, 0], means[:, 1:].mean(axis=1),
                              means[:, 2]-means[:, 1],
                              pre_raw[:, 0, h:].mean(axis=-1)-pre_raw[:, 0, :h].mean(axis=-1)])
    # 背景校准在每块独立滤波，且剔除所有扩展Trial及边缘1 s。
    boundaries = np.r_[0, [(on[np.flatnonzero(folds == k)[0]-1]+on[np.flatnonzero(folds == k)[0]])//2 for k in range(1,cfg.folds)], d.shape[1]]
    bg, bg_ecg = [], []
    slow_filter = butter(2, [.5, 8], btype='bandpass', fs=fs, output='sos')
    occupied = np.zeros(d.shape[1], bool)
    for s in on:
        occupied[max(0,s+pad[0]):min(d.shape[1],s+pad[-1]+1)] = True
    from scipy.ndimage import maximum_filter1d
    clipped = maximum_filter1d(np.any(np.abs(d[:3]) >= cfg.clip_level, axis=0), size=257)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        slow = sosfiltfilt(slow_filter, d[:3, a:b], axis=-1)
        ok = ~occupied[a:b] & ~clipped[a:b]
        ok[:cfg.fs] = False
        ok[-cfg.fs:] = False
        bg.append(slow[:, ok])
        bg_ecg.append(d[6, a:b].copy())
    task = int(re.search(r'Task-(\d)', path.stem).group(1))
    audit = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 samples=d.shape[1], fs=fs, channel_names=names, units='原始文件单位（未确认为微伏）',
                 trials=n, valid=int(valid.sum()), left=int((d[7,on]<0).sum()),
                 right=int((d[7,on]>0).sum()), cue_duration_min=float(((end-on)/fs).min()),
                 cue_duration_max=float(((end-on)/fs).max()), action_codes=np.unique(d[8]).tolist(),
                 action_used_as_label=False, min_onset_interval=float(np.diff(on).min()/fs),
                 machine_filtered_channels_used=False)
    return Record(path.stem, task, d[7,on].astype(int), (end-on)/fs, folds, on,
                  valid, reasons, t, pt, core, raw, eeg, light, ecg, metrics,
                  prestim, bg, bg_ecg, audit)


def load_records(directory, cfg):
    files = [Path(directory)/f'VisualCog{s}_Task-{task}.mat' for s in 'AB' for task in (1, 2)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'缺少原始文件：{path}')
    return [read_record(p, cfg) for p in files]
