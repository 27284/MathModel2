"""固定分类主方案、嵌套波形验证、公平基线与训练内敏感性对照。"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from .preprocess import prepare_fold
from .response import targets, fit_response, validation_loss, SETTINGS, FEATURE_SETTING
from .features import (time_features, build_features, fit_classifier, classify,
                       PRIMARY_SETTING, PROTOCOLS)
from .diagnostics import diagnostic_features

WAVE_MODELS = ('response', 'legacy_response', 'common_only', 'train_erp')
WINDOWS = (('0_800', 0., .801), ('0_500', 0., .5),
           ('250_500', .25, .5), ('600_800', .6, .801))


def write_json(path, data):
    def default(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding='utf-8')


def score_arrays(actual, predicted):
    error = actual - predicted
    sse = float(np.sum(error ** 2))
    centered = actual - actual.mean(axis=(0, 2), keepdims=True)
    sst = float(np.sum(centered ** 2))
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                r2=1 - sse / max(sst, 1e-12), sse=sse, sst=sst)


def erp_scores(actual, predicted, t, **metadata):
    rows = []
    for component in ('joint', 'common', 'contrast'):
        a, p = actual, predicted
        if component == 'common': a, p = a.mean(axis=0, keepdims=True), p.mean(axis=0, keepdims=True)
        if component == 'contrast': a, p = (a[1:2] - a[:1]) / 2, (p[1:2] - p[:1]) / 2
        for window, lo, hi in WINDOWS:
            mask = (t >= lo) & (t <= hi)
            rows.append(dict(**metadata, component=component, window=window,
                             **score_arrays(a[..., mask], p[..., mask])))
    return rows


class Engine:
    def __init__(self, records, bank, cfg):
        self.records, self.bank, self.cfg = records, bank, cfg
        self.prepared, self.learned, self.valid_targets = {}, {}, {}
        # 主分类特征完全不依赖标签，置换时可复用；标准化和分类系数每次重训。
        self.raw_features = [time_features(r.light, r.t) for r in records]

    def prepare(self, blocks):
        key = tuple(sorted(blocks))
        if key not in self.prepared:
            self.prepared[key] = [prepare_fold(r, blocks, self.cfg) for r in self.records]
        return self.prepared[key]

    def learn(self, blocks, labels):
        key = tuple(sorted(blocks))
        if key not in self.learned:
            p = self.prepare(blocks)
            self.learned[key] = (p, targets(self.records, p, labels, self.cfg))
        return self.learned[key]

    def select_response(self, blocks, labels):
        inner = []
        for b in sorted(blocks):
            p, cells = self.learn(blocks - {b}, labels)
            key = (tuple(sorted(blocks - {b})), b)
            if key not in self.valid_targets:
                self.valid_targets[key] = targets(self.records, p, labels, self.cfg, {b})
            inner.append((cells, self.valid_targets[key]))
        scores = []
        for setting in SETTINGS:
            losses = [validation_loss(self.records, self.bank,
                       fit_response(self.records, self.bank, train, setting), train, valid)
                      for train, valid in inner]
            scores.append(dict(setting=setting, inner_loss=float(np.mean(losses))))
        return min(scores, key=lambda row: row['inner_loss'])['setting'], scores

    def classify_fold(self, labels, fold, protocol, features, prepared, variant):
        groups = [[j] for j in range(len(self.records))] if protocol['scope'] == 'record' else [
            [j for j, r in enumerate(self.records) if r.task == task] for task in (1, 2)]
        rows, audits = [], []
        for group in groups:
            xt, yt, wt, xv, truth, indices, normalization = [], [], [], [], [], [], []
            for j in group:
                r, p, x, y = self.records[j], prepared[j], features[j], labels[j]
                test = (r.folds == fold) & r.valid
                if protocol['scope'] == 'task':
                    # 池化对照保留训练内逐记录归一化，避免仅靠量纲差异制造劣势。
                    center = x[p.train].mean(axis=0)
                    scale = np.maximum(x[p.train].std(axis=0), 1e-6)
                    x = (x - center) / scale
                    normalization.append(dict(record=r.name, center=center.tolist(), scale=scale.tolist()))
                xt.append(x[p.train]); yt.append(y[p.train]); wt.append(p.weights[p.train].mean(axis=1))
                xv.append(x[test]); truth.extend(y[test])
                indices.extend((j, int(i)) for i in np.flatnonzero(test))
            setting = (protocol['family'], protocol['estimator'], .1 if protocol['estimator'] == 'lr' else 0.)
            model = fit_classifier(np.concatenate(xt), np.concatenate(yt), np.concatenate(wt),
                                   setting, protocol['weighted'])
            guess, probability = classify(model, np.concatenate(xv))
            for (j, i), y, pred, prob in zip(indices, truth, guess, probability):
                r = self.records[j]
                rows.append(dict(record=r.name, task=r.task, trial=i + 1, fold=fold, variant=variant,
                                 truth=int(y), prediction=int(pred), right_probability=float(prob)))
            audits.append(dict(fold=fold, variant=variant, scope=protocol['scope'],
                               records=[self.records[j].name for j in group],
                               training_blocks=sorted(set(range(self.cfg.folds)) - {fold}),
                               training_trials={self.records[j].name: np.flatnonzero(prepared[j].train).tolist() for j in group},
                               record_normalization=normalization, classifier=model,
                               selection='fixed_before_evaluation'))
        return rows, audits

    def run(self, labels, out=None, full=True):
        self.learned.clear(); self.valid_targets.clear()
        rows, classifiers, audits, diagnostics, parameters, erp_rows = [], [], [], [], [], []
        archive_erp = {'time_s': self.bank.t} if full else {}
        waves = {name: [np.full_like(r.light, np.nan) for r in self.records] for name in WAVE_MODELS} if full else {}
        for fold in range(self.cfg.folds):
            blocks = set(range(self.cfg.folds)) - {fold}
            p = self.prepare(blocks)
            variants = {'temporal24': self.raw_features}
            if full:
                _, cells = self.learn(blocks, labels)
                feature_model = fit_response(self.records, self.bank, cells, FEATURE_SETTING)
                variants, transforms = build_features(self.records, p, labels, self.bank, feature_model, cells)
                for j, (r, prep) in enumerate(zip(self.records, p)):
                    d, audit = diagnostic_features(r, prep.train)
                    diagnostics.append(dict(fold=fold, **audit))
                    for name, x in d.items(): variants.setdefault(name, []).append(x)
                    audits.append(dict(fold=fold, **prep.audit, matching_transform=transforms[j]))
            wanted = PROTOCOLS if full else {'main': PROTOCOLS['main']}
            for variant, protocol in wanted.items():
                rr, aa = self.classify_fold(labels, fold, protocol, variants[protocol['family']], p, variant)
                rows.extend(rr); classifiers.extend(aa)
            if not full: continue
            setting, scores = self.select_response(blocks, labels)
            model = fit_response(self.records, self.bank, cells, setting)
            legacy_setting = min((s for s in scores if s['setting'][2] == 'legacy'), key=lambda s: s['inner_loss'])['setting']
            legacy = fit_response(self.records, self.bank, cells, legacy_setting)
            parameters.append(dict(fold=fold, training_blocks=sorted(blocks), candidates=scores,
                                   main=model.to_dict(), legacy=legacy.to_dict()))
            test_cells = targets(self.records, p, labels, self.cfg, {fold})
            for j, (r, y) in enumerate(zip(self.records, labels)):
                ids = (r.folds == fold) & r.valid
                waves['response'][j][ids] = model.predict(self.bank, j, y[ids], r.duration[ids])
                waves['legacy_response'][j][ids] = legacy.predict(self.bank, j, y[ids], r.duration[ids])
                waves['common_only'][j][ids] = model.predict(self.bank, j, y[ids], r.duration[ids], True)
                waves['train_erp'][j][ids] = cells[0][j, (y[ids] > 0).astype(int)]
                for split, target in [('train', cells), ('heldout', test_cells)]:
                    duration = [target[2][j]] * 2
                    predictions = dict(response=model.predict(self.bank, j, [-1, 1], duration),
                                       legacy_response=legacy.predict(self.bank, j, [-1, 1], duration),
                                       common_only=model.predict(self.bank, j, [-1, 1], duration, True),
                                       train_erp=cells[0][j])
                    for name, prediction in predictions.items():
                        actual = target[0][j]
                        erp_rows.extend(erp_scores(actual, prediction, r.t, fold=fold, record=r.name,
                                                  task=r.task, split=split, model=name))
                        for sign in (0, 1):
                            prefix = f'{fold}__{r.name}__{split}__{name}__{sign}'
                            archive_erp[prefix + '__actual'] = actual[sign]
                            archive_erp[prefix + '__predicted'] = prediction[sign]
            logging.info('外层时间块 %d/5：固定分类、混杂对照及波形验证完成', fold + 1)
        pred = pd.DataFrame(rows)
        if out:
            pred.to_csv(out / 'oof_predictions.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(erp_rows).to_csv(out / 'erp_validation.csv', index=False, encoding='utf-8-sig')
            write_json(out / 'classifier_audit.json', classifiers)
            write_json(out / 'response_parameters.json', parameters)
            write_json(out / 'training_audit.json', audits)
            write_json(out / 'confound_audit.json', diagnostics)
            archive = {'time_s': self.bank.t}
            for j, r in enumerate(self.records):
                archive[r.name + '__valid'] = r.valid
                archive[r.name + '__actual'] = r.light
                for name in waves: archive[r.name + '__' + name] = waves[name][j]
            np.savez_compressed(out / 'heldout_waveforms.npz', **archive)
            np.savez_compressed(out / 'heldout_erp.npz', **archive_erp)
        return pred, waves

    def final_model(self, labels):
        self.learned.clear(); self.valid_targets.clear()
        blocks = set(range(self.cfg.folds))
        p, cells = self.learn(blocks, labels)
        setting, scores = self.select_response(blocks, labels)
        response = fit_response(self.records, self.bank, cells, setting)
        classifiers, transforms = [], []
        for r, prep, y, x in zip(self.records, p, labels, self.raw_features):
            classifiers.append(dict(record=r.name, task=r.task,
                **fit_classifier(x[prep.train], y[prep.train], setting=PRIMARY_SETTING)))
            transforms.append(dict(record=r.name, task=r.task,
                                   training_trials=np.flatnonzero(prep.train).tolist()))
        result = dict(schema_version=2, config=self.cfg.to_dict(), time_s=self.bank.t.tolist(),
                      records=[r.name for r in self.records], response=response.to_dict(),
                      response_candidates=scores, transforms=transforms, classifiers=classifiers,
                      source_audit=self.bank.audit(), classification_protocol=PROTOCOLS['main'],
                      note='逐记录固定LR；全数据模型不用于留出计分。无解剖空间前向模型。')
        return result, response, cells


def restricted_permutation(records, labels, rng):
    result = []
    for r, y in zip(records, labels):
        shuffled = y.copy()
        for block in np.unique(r.folds):
            ids = np.flatnonzero(r.valid & (r.folds == block))
            shuffled[ids] = rng.permutation(shuffled[ids])
        result.append(shuffled)
    return result


def groups(frame):
    yield 'all', 'all', 0, frame
    for task in (1, 2): yield f'task{task}', 'task', task, frame[frame.task == task]
    for name, rows in frame.groupby('record', sort=True): yield name, 'record', int(rows.task.iloc[0]), rows


def permutation_test(engine, labels, observed, count, out):
    path = out / 'permutation_scores.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    observed = observed[observed.variant == 'main']
    expected = {name for name, _, _, _ in groups(observed)}
    done = {i for i in range(count)
            if {r['group'] for r in rows if int(r['permutation']) == i} == expected
            and sum(int(r['permutation']) == i for r in rows) == len(expected)}
    for index in range(count):
        if index in done: continue
        rows = [row for row in rows if int(row['permutation']) != index]
        shuffled = restricted_permutation(engine.records, labels, np.random.default_rng(engine.cfg.seed + 10000 + index))
        pred, _ = engine.run(shuffled, full=False)
        for name, scope, task, s in groups(pred):
            rows.append(dict(permutation=index, group=name, scope=scope, task=task,
                             ba=float(balanced_accuracy_score(s.truth, s.prediction))))
        temporary = path.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(temporary, index=False, encoding='utf-8-sig')
        temporary.replace(path)
        if (index + 1) % 20 == 0 or index + 1 == count:
            logging.info('主分类完整置换 %d/%d', index + 1, count)
    if not path.exists():
        pd.DataFrame(columns=['permutation', 'group', 'scope', 'task', 'ba']).to_csv(path, index=False, encoding='utf-8-sig')
    result = []
    for name, scope, task, s in groups(observed):
        obs = float(balanced_accuracy_score(s.truth, s.prediction))
        null = np.array([r['ba'] for r in rows if r['group'] == name and int(r['permutation']) < count])
        result.append(dict(group=name, scope=scope, task=task, ba=obs, permutations=len(null),
                           p_value=float((1 + (null >= obs).sum()) / (1 + len(null))) if len(null) else None))
    write_json(out / 'permutation_test.json', result)
    return result


def classification_metrics(pred, cfg):
    rows = []
    rng = np.random.default_rng(cfg.seed)
    for variant, selection in pred.groupby('variant', sort=False):
        for name, scope, task, s in groups(selection):
            cm = confusion_matrix(s.truth, s.prediction, labels=[-1, 1])
            boot = np.zeros((cfg.bootstrap, 2, 2))
            for _, record in s.groupby('record', sort=True):
                blocks = np.array([confusion_matrix(b.truth, b.prediction, labels=[-1, 1]) for _, b in record.groupby('fold')])
                boot += blocks[rng.integers(0, len(blocks), (cfg.bootstrap, len(blocks)))].sum(axis=1)
            denominator = boot.sum(axis=-1)
            keep = (denominator > 0).all(axis=1)
            ba = (boot[keep].diagonal(axis1=-2, axis2=-1) / denominator[keep]).mean(axis=-1)
            lo, hi = np.quantile(ba, [.025, .975])
            rows.append(dict(group=name, scope=scope, task=task, variant=variant, n=len(s),
                             ba=balanced_accuracy_score(s.truth, s.prediction),
                             accuracy=float((s.truth == s.prediction).mean()),
                             conditional_low=lo, conditional_high=hi,
                             LL=int(cm[0, 0]), LR=int(cm[0, 1]), RL=int(cm[1, 0]), RR=int(cm[1, 1])))
    return pd.DataFrame(rows)


def waveform_metrics(records, waves):
    rows = []
    for j, record in enumerate(records):
        for name, arrays in waves.items():
            for window, lo, hi in WINDOWS:
                mask = (record.t >= lo) & (record.t <= hi)
                actual = record.light[record.valid][..., mask]
                prediction = arrays[j][record.valid][..., mask]
                rows.append(dict(record=record.name, task=record.task, model=name, window=window,
                                 **score_arrays(actual, prediction)))
    return pd.DataFrame(rows)
