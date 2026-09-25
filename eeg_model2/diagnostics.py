"""训练内混杂回归；仅作敏感性分析，不改写Record.light或主分类输入。"""
import numpy as np
from .data import recenter
from .features import time_features

RIDGE = 1.0


def residualize(features, covariates, train):
    """训练均值标准化协变量、固定岭回归；所有试次减去训练所得预测偏差。"""
    c = np.asarray(covariates, float)
    center = c[train].mean(axis=0)
    scale = np.maximum(c[train].std(axis=0), 1e-8)
    z = (c - center) / scale
    feature_mean = features[train].mean(axis=0)
    matrix = z[train].T @ z[train]
    coef = np.linalg.solve(matrix + RIDGE * np.eye(c.shape[1]),
                           z[train].T @ (features[train] - feature_mean))
    adjusted = features - z @ coef
    return adjusted, dict(center=center.tolist(), scale=scale.tolist(),
                          coefficients=coef.tolist(), ridge=RIDGE,
                          training_trials=np.flatnonzero(train).tolist())


def ecg_adjust(record, train):
    """用训练试次的心电及±约40/80ms滞后项回归脑电，测试脑电不参与估计。"""
    lags = np.array([-20, -10, 0, 10, 20])
    design = np.stack([record.ecg[:, record.core + lag] for lag in lags], axis=-1)
    design -= design[:, record.t < 0].mean(axis=1, keepdims=True)
    scale = np.maximum(np.sqrt(np.mean(design[train] ** 2, axis=(0, 1))), 1e-8)
    design = design / scale
    x = design[train].reshape(-1, len(lags))
    y = record.light[train].transpose(0, 2, 1).reshape(-1, 3)
    coef = np.linalg.solve(x.T @ x / len(x) + .01 * np.eye(len(lags)), x.T @ y / len(x))
    contribution = (design @ coef).transpose(0, 2, 1)
    adjusted = recenter(record.light - contribution, record.t)
    return adjusted, dict(lags_samples=lags.tolist(), scale=scale.tolist(),
                          coefficients=coef.tolist(), ridge=.01,
                          training_trials=np.flatnonzero(train).tolist(),
                          note='辅助ECG为既有1–25Hz信号；不等于完整心源/眼源伪影剔除')


def diagnostic_features(record, train):
    raw = time_features(record.light, record.t)
    qc = record.metrics.reshape(len(record.labels), -1)
    order = np.arange(len(record.labels), dtype=float)[:, None]
    adjusted_signal, ecg_audit = ecg_adjust(record, train)
    ecg_features = time_features(adjusted_signal, record.t)
    quality, qc_audit = residualize(raw, qc, train)
    ordinal, order_audit = residualize(raw, order, train)
    combined, combined_audit = residualize(ecg_features, np.column_stack([qc, order]), train)
    return dict(ecg_adjusted=ecg_features, qc_adjusted=quality, order_adjusted=ordinal,
                all_adjusted=combined, prestim_only=record.prestim.copy()), dict(
                    record=record.name, ecg=ecg_audit, qc=qc_audit,
                    order=order_audit, combined=combined_audit)


def trend_metrics(records, cells):
    """全数据条件ERP的描述性线性趋势，不作为来源判据或留出预测。"""
    rows = []
    for j, record in enumerate(records):
        for sign, direction in enumerate((-1, 1)):
            for window, lo, hi in [('0_800', 0., .801), ('250_500', .25, .5)]:
                mask = (record.t >= lo) & (record.t <= hi)
                t = record.t[mask]
                design = np.column_stack([np.ones(len(t)), t - t.mean()])
                for ch, name in enumerate(('Fz', 'F3', 'F4')):
                    y = cells[0][j, sign, ch, mask]
                    coef = np.linalg.lstsq(design, y, rcond=None)[0]
                    residual = y - design @ coef
                    sst = np.sum((y - y.mean()) ** 2)
                    rows.append(dict(record=record.name, direction=direction, channel=name,
                                     window=window, slope=float(coef[1]),
                                     trend_r2=float(1 - np.sum(residual ** 2) / max(sst, 1e-12)),
                                     scope='descriptive_full_data_erp'))
    return rows
