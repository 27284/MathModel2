"""固定逐记录分类主方案；对照不参与外层择优。"""
import numpy as np
from scipy.special import expit
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

PRIMARY_SETTING = ('temporal24', 'lr', 0.1)
SETTINGS = (PRIMARY_SETTING, ('temporal24', 'lda', 0.))
PROTOCOLS = {
    'main': dict(scope='record', family='temporal24', estimator='lr', weighted=False),
    'record_lda': dict(scope='record', family='temporal24', estimator='lda', weighted=False),
    'pooled': dict(scope='task', family='temporal24', estimator='lr', weighted=False),
    'qc_weighted': dict(scope='record', family='temporal24', estimator='lr', weighted=True),
    'mechanistic': dict(scope='record', family='contrast25', estimator='lr', weighted=False),
    **{name: dict(scope='record', family=name, estimator='lr', weighted=False)
       for name in ('ecg_adjusted', 'qc_adjusted', 'order_adjusted', 'all_adjusted', 'prestim_only')},
}


def time_features(x, t):
    """Fz的8个窗、F3的8个窗、F4的8个窗；100–500ms，左闭右开。"""
    edges = np.linspace(.1, .5, 9)
    return np.stack([x[..., (t >= a) & (t < b)].mean(axis=-1)
                     for a, b in zip(edges[:-1], edges[1:])], axis=-1).reshape(len(x), 24)


def build_features(records, prepared, labels, bank=None, model=None, cells=None):
    """只为mechanistic对照构建监督匹配分；主方案直接使用原幅值24维。"""
    variants = {'temporal24': [], 'contrast25': []}
    transforms = []
    for j, (r, p, y) in enumerate(zip(records, prepared, labels)):
        x = time_features(r.light, r.t)
        variants['temporal24'].append(x)
        transform = dict(record=r.name, task=r.task,
                         training_trials=np.flatnonzero(p.train).tolist())
        if model is not None:
            train, truth = x[p.train], y[p.train]
            residual = train.copy()
            for sign in (-1, 1):
                residual[truth == sign] -= train[truth == sign].mean(axis=0)
            covariance = LedoitWolf().fit(residual).covariance_
            covariance += max(np.trace(covariance) / 24, 1e-12) * 1e-6 * np.eye(24)
            duration = float(np.median(r.duration[p.train]))
            template = time_features(model.predict(bank, j, [-1, 1], [duration] * 2), r.t)
            m, d = template.mean(axis=0), (template[1] - template[0]) / 2
            w = np.linalg.solve(covariance, d)
            norm = float(np.sqrt(max(d @ w, 1e-12)))
            w /= norm
            variants['contrast25'].append(np.column_stack([x, (x - m) @ w]))
            transform.update(template_common=m.tolist(), matched_weight=w.tolist(),
                             mahalanobis_contrast=norm)
        transforms.append(transform)
    return variants, transforms


def apply_transform(epochs, t, transform, family):
    x = time_features(epochs, t)
    if family == 'temporal24':
        return x
    if family != 'contrast25':
        raise ValueError('预测接口仅支持主方案或机理对照的脑电特征')
    score = (x - np.asarray(transform['template_common'])) @ np.asarray(transform['matched_weight'])
    return np.column_stack([x, score])


def fit_classifier(x, y, w=None, setting=PRIMARY_SETTING, weighted=False):
    family, kind, c = setting
    weight = None if not weighted else np.asarray(w, float) / np.mean(w)
    scaler = StandardScaler().fit(x, sample_weight=weight)
    xx = scaler.transform(x)
    if kind == 'lr':
        clf = LogisticRegression(C=c, class_weight='balanced', max_iter=2000).fit(xx, y, sample_weight=weight)
    elif kind == 'lda':
        if weighted:
            raise ValueError('LDA对照不接受QC样本权重')
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[.5, .5]).fit(xx, y)
    else:
        raise ValueError('未知分类器')
    return dict(family=family, estimator=kind, C=c, qc_weighted=weighted,
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coefficients=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]))


def classify(model, x):
    xx = (x - np.asarray(model['mean'])) / np.asarray(model['scale'])
    probability = expit(xx @ np.asarray(model['coefficients']) + model['intercept'])
    return np.where(probability >= .5, 1, -1), probability
