"""沿用既有轻处理和刺激前QC；不读取或修改第一问处理结果。"""
from dataclasses import dataclass
import numpy as np
from .data import recenter


def quality_weights(metrics, train, valid, cfg):
    z = metrics[train & valid]
    if len(z) < 10:
        raise ValueError('训练质量标定有效Trial不足10个')
    center = np.median(z, axis=0)
    scale = np.maximum(1.4826*np.median(np.abs(z-center), axis=0),
                       (np.percentile(z, 75, axis=0)-np.percentile(z, 25, axis=0))/1.349)
    score = np.maximum(0., (metrics-center)/np.maximum(scale, 1e-12)) / np.array([3.,3.,4.,3.])
    w = np.clip(1/np.maximum(1., np.sum(score**2, axis=-1)), cfg.min_weight, 1.)
    w[~valid] = 0.
    return w, center, scale


def weighted_median(x, w):
    order = np.argsort(x, axis=0)
    sx = np.take_along_axis(x, order, axis=0)
    sw = w[order]
    at = np.argmax(np.cumsum(sw, axis=0) >= w.sum()/2, axis=0)
    return np.take_along_axis(sx, at[None], axis=0)[0]


def huber_erp(x, weights, t, c=1.345):
    """逐通道QC×Huber；固定加权MAD尺度，必要时二分求得分根。"""
    out = np.zeros(x.shape[1:])
    for ch in range(x.shape[1]):
        keep = weights[:, ch] > 0
        z, w = x[keep, ch], weights[keep, ch]
        if not len(w):
            raise ValueError('条件内没有有效ERP试次')
        mu = weighted_median(z, w)
        scale = 1.4826*weighted_median(np.abs(z-mu), w)
        positive = scale[scale > 0]
        floor = max(1e-12, 1e-6*np.median(positive)) if len(positive) else 1e-12
        scale = np.maximum(scale, floor)
        for _ in range(100):
            h = np.minimum(1., c/np.maximum(np.abs((z-mu)/scale), 1e-300))
            combined = w[:, None]*h
            updated = np.sum(combined*z, axis=0)/combined.sum(axis=0)
            converged = np.abs(updated-mu) <= 1e-8*(1+np.abs(mu))
            mu = updated
            if converged.all():
                break
        if not converged.all():
            mask = ~converged
            zz, ss = z[:, mask], scale[mask]
            lo, hi = zz.min(axis=0), zz.max(axis=0)
            for _ in range(80):
                mid = (lo+hi)/2
                score = np.sum(w[:, None]*np.clip((zz-mid)/ss, -c, c), axis=0)
                lo = np.where(score >= 0, mid, lo)
                hi = np.where(score <= 0, mid, hi)
            mu[mask] = (lo+hi)/2
        out[ch] = mu
    return recenter(out, t)


@dataclass
class Prepared:
    weights: np.ndarray
    train: np.ndarray
    audit: dict


def prepare_fold(record,blocks,cfg):
    train=np.isin(record.folds,list(blocks))&record.valid
    w,center,scale=quality_weights(record.metrics,train,record.valid,cfg)
    audit=dict(record=record.name,training_blocks=sorted(blocks),
               training_trials=np.flatnonzero(train).tolist(),qc_center=center.tolist(),qc_scale=scale.tolist())
    return Prepared(w,train,audit)



