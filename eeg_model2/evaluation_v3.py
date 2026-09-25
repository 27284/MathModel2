"""Nested blocked V3 response evaluation, with frozen classification."""
from dataclasses import asdict, replace
import logging
import numpy as np
import pandas as pd
from .basis import BasisBank, WARP_GROUPS
from .slow import fit_slow, apply_slow, SLOW_SETTINGS
from .preprocess import huber_erp
from .response import targets, fit_response, FEATURE_SETTING
from .response_v3 import Setting, fit_response_v3
from .evaluation import Engine, erp_scores, write_json, WINDOWS
from .nuisance import nuisance_features
from .features import fit_classifier, classify, time_features, PRIMARY_SETTING, PROTOCOLS
from .evaluation import restricted_permutation, groups
from sklearn.metrics import balanced_accuracy_score

LAMBDAS = (.1, .01, .001)
POOLS = (0., .1, 1.)


class V3Engine(Engine):
    def __init__(self, records, bank, cfg):
        super().__init__(records, bank, cfg)
        self.basis = BasisBank(bank)
        self.slow_cache, self.cell_cache = {}, {}

    def slow_models(self, blocks, setting):
        key = (tuple(sorted(blocks)), setting)
        if key not in self.slow_cache:
            self.slow_cache[key] = [fit_slow(r.raw_epochs[p.train], r.onsets[p.train]/self.cfg.fs,
                                                   setting, r.padded_t)
                                    for r, p in zip(self.records, self.prepare(blocks))]
        return self.slow_cache[key]

    def cells(self, blocks, slow, evaluation=None):
        key = (tuple(sorted(blocks)), slow, tuple(sorted(evaluation)) if evaluation is not None else None)
        if key in self.cell_cache:
            return self.cell_cache[key]
        prepared = self.prepare(blocks)
        models = self.slow_models(blocks, slow)
        actual, residual, components, sizes, durations = [], [], [], [], []
        for r, p, model in zip(self.records, prepared, models):
            selection = p.train if evaluation is None else r.valid & np.isin(r.folds, list(evaluation))
            clean, trend = apply_slow(model, r.light, r.onsets/self.cfg.fs, r.t)
            a, c, s, n = [], [], [], []
            for sign in (-1, 1):
                ids = selection & (r.labels == sign)
                if ids.sum()<2:
                    raise ValueError('条件内有效 Trial 不足')
                a.append(huber_erp(r.light[ids], p.weights[ids], r.t, self.cfg.huber_c))
                c.append(huber_erp(clean[ids], p.weights[ids], r.t, self.cfg.huber_c))
                w = p.weights[ids]
                s.append(np.sum(trend[ids]*w[..., None], axis=0)/w.sum(axis=0)[:, None])
                w = w.mean(axis=1); n.append(w.sum()**2/(w@w))
            actual.append(a); residual.append(c); components.append(s); sizes.append(n)
            # Both train and heldout predictions use training median cue duration.
            durations.append(float(np.median(r.duration[p.train])))
        result = dict(actual=np.array(actual), residual=np.array(residual), slow=np.array(components),
                      neff=np.array(sizes), durations=np.array(durations))
        self.cell_cache[key] = result
        return result

    def fit(self, blocks, setting):
        cells = self.cells(blocks, setting.slow)
        return fit_response_v3(self.records, self.basis, cells['residual'], cells['neff'],
                               cells['durations'], setting)

    def prediction(self, model, cells, j):
        neural = model.neural(self.basis, j, cells['durations'][j])
        return cells['slow'][j] + np.stack([neural[0]-neural[1], neural[0]+neural[1]])

    def select(self, blocks, candidates):
        scores = []
        for setting in candidates:
            losses, secondary = [], []
            for held in sorted(blocks):
                train = blocks-{held}
                model = self.fit(train, setting)
                tc = self.cells(train, setting.slow)
                vc = self.cells(train, setting.slow, {held})
                for j in range(len(self.records)):
                    actual, prediction = vc['actual'][j], self.prediction(model, vc, j)
                    scale = np.maximum(np.mean(tc['actual'][j][..., self.bank.t>=0]**2, axis=(0, 2)), 1.)
                    error = prediction-actual
                    contrast = (error[1]-error[0])/2
                    for target, mask in ((losses, (self.bank.t>=.25)&(self.bank.t<=.5)),
                                         (secondary, self.bank.t>=0)):
                        target.append(float(np.mean(error[..., mask]**2/scale[None, :, None]) +
                                            np.mean(contrast[:, mask]**2/scale[:, None])))
            scores.append(dict(setting=asdict(setting), inner_loss=float(np.mean(losses)),
                               inner_full_loss=float(np.mean(secondary))))
        selected = min(range(len(scores)), key=lambda i: (scores[i]['inner_loss'], scores[i]['inner_full_loss']))
        return candidates[selected], scores

    def stages(self, blocks):
        chosen, audit = {}, {}
        # Small, predeclared staged search, not an unrestricted factorial bank.
        for name, family in (('M1', 'temporal'), ('M2', 'wc')):
            candidates = [Setting(slow=s, family=family, lambda_common=l, lambda_contrast=10*l)
                          for s in SLOW_SETTINGS for l in LAMBDAS]
            chosen[name], audit[name] = self.select(blocks, candidates)
        candidates = [replace(chosen['M2'], warp=w) for w in WARP_GROUPS]
        chosen['M3'], audit['M3'] = self.select(blocks, candidates)
        candidates = [replace(chosen['M3'], lambda_common=c, lambda_contrast=d)
                      for c in LAMBDAS for d in LAMBDAS]
        chosen['M4'], audit['M4'] = self.select(blocks, candidates)
        # Pooling is a separate experiment; main ladder stays at pool=0.
        chosen['pool_selected'], audit['pool_selected'] = self.select(
            blocks, [replace(chosen['M4'], pool=p) for p in POOLS])
        chosen['M5'], audit['M5'] = self.select(blocks, [replace(chosen['M4'], forward='symmetric_forward',
                                                     lambda_forward=l) for l in LAMBDAS])
        chosen['geometry_prior_forward'], audit['geometry_prior_forward'] = self.select(
            blocks, [replace(chosen['M4'], forward='geometry_prior_forward', lambda_forward=l) for l in (.1, 1., 10.)])
        chosen.update({f'pool_{p:g}': replace(chosen['M4'], pool=p) for p in POOLS})
        chosen['LP750'] = replace(chosen['M4'], lp750=True)
        return chosen, audit

    def run_responses(self, out):
        rows, parameters, decomposition, forward_rows = [], [], [], []
        archive = dict(time_s=self.bank.t)
        labels = [r.labels for r in self.records]
        for fold in range(self.cfg.folds):
            blocks = set(range(self.cfg.folds))-{fold}
            logging.info('V3 outer block %d/%d: nested selection', fold+1, self.cfg.folds)
            settings, selection = self.stages(blocks)
            models = {name: self.fit(blocks, setting) for name, setting in settings.items()}
            base_train, base_test = self.cells(blocks, 'none'), self.cells(blocks, 'none', {fold})
            legacy_cells = targets(self.records, self.prepare(blocks), labels, self.cfg)
            legacy = fit_response(self.records, self.bank, legacy_cells, FEATURE_SETTING)
            parameters.append(dict(fold=fold, training_blocks=sorted(blocks),
                inner_training_blocks=[sorted(blocks-{b}) for b in sorted(blocks)],
                selection=selection, models={n: m.to_dict() for n, m in models.items()},
                slow_models={s: [m.to_dict() for m in self.slow_models(blocks, s)] for s in SLOW_SETTINGS},
                training_trials={r.name: np.flatnonzero(p.train).tolist() for r, p in zip(self.records, self.prepare(blocks))}))
            for j, r in enumerate(self.records):
                predictions = dict(M0=base_train['actual'][j],
                                   legacy_response=legacy.predict(self.bank, j, [-1, 1], [legacy_cells[2][j]]*2))
                for name, model in models.items():
                    test = self.cells(blocks, model.setting.slow, {fold})
                    predictions[name] = self.prediction(model, test, j)
                    if name in ('M4', 'M5'):
                        prefix = f'{fold}__{r.name}__{name}'
                        archive[prefix+'__actual'] = test['actual'][j]
                        archive[prefix+'__predicted'] = predictions[name]
                        archive[prefix+'__slow'] = test['slow'][j]
                        if name == 'M4':
                            for sign in range(2):
                                for ch, channel in enumerate(('Fz', 'F3', 'F4')):
                                    for k, t in enumerate(r.t):
                                        slow = test['slow'][j, sign, ch, k]
                                        observed = test['actual'][j, sign, ch, k]
                                        decomposition.append(dict(fold=fold, record=r.name, direction=(-1,1)[sign],
                                            channel=channel, time_s=t, observed=observed, slow=slow,
                                            residual_erp=observed-slow,
                                            neural_prediction=predictions[name][sign,ch,k]-slow))
                    if model.forward_matrices:
                        g = model.forward_matrices[j]
                        for ch, channel in enumerate(('Fz', 'F3', 'F4')):
                            forward_rows.append(dict(fold=fold, record=r.name, model=name, channel=channel,
                                g_left=g[ch,0], g_right=g[ch,1], lateralization=g[ch,1]-g[ch,0],
                                lambda_forward=model.setting.lambda_forward))
                # Prioritized P1 check: add slow to exactly the fixed legacy response.
                slow_setting = settings['M2'].slow
                tc, vc = self.cells(blocks, slow_setting), self.cells(blocks, slow_setting, {fold})
                old_slow = fit_response(self.records, self.bank, (tc['residual'], tc['neff'], tc['durations']), FEATURE_SETTING)
                predictions['slow_legacy_response'] = old_slow.predict(self.bank,j,[-1,1],[tc['durations'][j]]*2)+vc['slow'][j]
                for name, prediction in predictions.items():
                    scores = erp_scores(base_test['actual'][j], prediction, r.t,
                                        fold=fold, record=r.name, task=r.task, model=name, split='heldout')
                    for row in scores:
                        row['n_values'] = int(((r.t>=dict((w,(lo,hi)) for w,lo,hi in WINDOWS)[row['window']][0]) &
                                              (r.t<=dict((w,(lo,hi)) for w,lo,hi in WINDOWS)[row['window']][1])).sum())*3*(2 if row['component']=='joint' else 1)
                    rows.extend(scores)
            logging.info('V3 outer block %d: all response ablations complete', fold+1)
        frame = pd.DataFrame(rows)
        frame.to_csv(out/'component_metrics_v3.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(decomposition).to_csv(out/'slow_decomposition_v3.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(forward_rows).to_csv(out/'forward_parameters_v3.csv', index=False, encoding='utf-8-sig')
        write_json(out/'response_parameters_v3.json', parameters)
        np.savez_compressed(out/'heldout_erp_v3.npz', **archive)
        summarize_v3(frame, out)
        return frame

    def run_classification(self, out):
        labels = [r.labels for r in self.records]
        primary, _ = self.run(labels, full=False)
        rows, audits, status, classifier_audits = primary.to_dict('records'), [], [], []
        for fold in range(self.cfg.folds):
            blocks = set(range(self.cfg.folds))-{fold}
            for j, (r, p) in enumerate(zip(self.records, self.prepare(blocks))):
                features, audit = nuisance_features(r, p.train, self.cfg.fs)
                audits.append(dict(record=r.name, fold=fold, training_blocks=sorted(blocks), **audit))
                test = r.valid & (r.folds==fold)
                for name in ('raw', 'ecg_adjusted', 'qc_adjusted', 'order_adjusted', 'all_adjusted'):
                    status.append(dict(record=r.name, fold=fold, variant=name,
                        status='computed' if name in features else 'not_run_channel7_not_verified'))
                    if name not in features:
                        continue
                    model = fit_classifier(features[name][p.train], r.labels[p.train], setting=PRIMARY_SETTING)
                    guess, prob = classify(model, features[name][test])
                    classifier_audits.append(dict(record=r.name, fold=fold, variant=name,
                        training_trials=np.flatnonzero(p.train).tolist(), classifier=model))
                    for i, y, pr in zip(np.flatnonzero(test), guess, prob):
                        rows.append(dict(record=r.name, task=r.task, trial=i+1, fold=fold, variant=name,
                                         truth=int(r.labels[i]), prediction=int(y), right_probability=float(pr)))
        pred = pd.DataFrame(rows)
        pred.to_csv(out/'oof_predictions.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(status).to_csv(out/'nuisance_status_v3.csv', index=False, encoding='utf-8-sig')
        write_json(out/'confound_audit_v3.json', audits)
        write_json(out/'classifier_audit_v3.json', classifier_audits)
        # Reuse the frozen schema-2 classifier contract for predict.py.
        classifiers = [dict(record=r.name, task=r.task,
                            **fit_classifier(x[r.valid], r.labels[r.valid], setting=PRIMARY_SETTING))
                       for r, x in zip(self.records, self.raw_features)]
        write_json(out/'final_model.json', dict(schema_version=2, response_version=3,
            records=[r.name for r in self.records], time_s=self.bank.t, classifiers=classifiers,
            transforms=[dict(record=r.name, task=r.task) for r in self.records],
            classification_protocol=PROTOCOLS['main'],
            note='Frozen classifier only; response models are outer-fold models in response_parameters_v3.json.'))
        return pred

    def ecg_permutation_test(self, observed, count, out):
        """Independent seeds; label-free nuisance fits cached, LR retrained per permutation."""
        observed = observed[observed.variant=='ecg_adjusted']
        expected_count = sum(int(r.valid.sum()) for r in self.records)
        if len(observed)!=expected_count:
            result = dict(status='not_run_incomplete_ecg_gate_coverage', permutations=0)
            write_json(out/'ecg_permutation_test.json', result)
            return result
        entries = []
        for fold in range(self.cfg.folds):
            blocks = set(range(self.cfg.folds))-{fold}
            for j, (r, p) in enumerate(zip(self.records, self.prepare(blocks))):
                features, _ = nuisance_features(r, p.train, self.cfg.fs)
                entries.append((fold, j, p.train, r.valid&(r.folds==fold), features['ecg_adjusted']))
        path = out/'ecg_permutation_scores.csv'
        rows = pd.read_csv(path).to_dict('records') if path.exists() else []
        expected = {name for name,_,_,_ in groups(observed)}
        done = {i for i in range(count) if {r['group'] for r in rows if int(r['permutation'])==i}==expected
                and sum(int(r['permutation'])==i for r in rows)==len(expected)}
        for index in range(count):
            if index in done:continue
            rows = [r for r in rows if int(r['permutation'])!=index]
            labels = restricted_permutation(self.records, [r.labels for r in self.records],
                                            np.random.default_rng(self.cfg.seed+20000+index))
            predictions = []
            for fold,j,train,test,x in entries:
                r,y = self.records[j],labels[j]
                model = fit_classifier(x[train],y[train],setting=PRIMARY_SETTING)
                guess,_ = classify(model,x[test])
                predictions.extend(dict(record=r.name,task=r.task,fold=fold,truth=int(t),prediction=int(p))
                                   for t,p in zip(y[test],guess))
            for name,scope,task,s in groups(pd.DataFrame(predictions)):
                rows.append(dict(permutation=index,group=name,scope=scope,task=task,
                                 ba=float(balanced_accuracy_score(s.truth,s.prediction))))
            tmp=path.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(path)
            if (index+1)%20==0 or index+1==count:
                logging.info('ECG sensitivity independent permutation %d/%d',index+1,count)
        if not path.exists():
            pd.DataFrame(columns=['permutation','group','scope','task','ba']).to_csv(path,index=False,encoding='utf-8-sig')
        result=[]
        for name,scope,task,s in groups(observed):
            ba=float(balanced_accuracy_score(s.truth,s.prediction))
            null=np.array([r['ba'] for r in rows if r['group']==name and int(r['permutation'])<count])
            result.append(dict(group=name,scope=scope,task=task,ba=ba,permutations=len(null),
                p_value=float((1+(null>=ba).sum())/(1+len(null))) if len(null) else None,
                status='sensitivity_only',seed_offset=20000))
        write_json(out/'ecg_permutation_test.json',result)
        return result


def summarize_v3(frame, out):
    keys = ['fold', 'record', 'component', 'window']
    baseline = frame[frame.model=='M0'][keys+['r2','sse']].rename(columns={'r2':'template_r2','sse':'template_sse'})
    paired = frame.merge(baseline, on=keys, validate='many_to_one')
    paired['delta_r2'] = paired.r2-paired.template_r2
    paired['delta_sse'] = paired.template_sse-paired.sse
    paired.to_csv(out/'foldwise_delta_sse_v3.csv', index=False, encoding='utf-8-sig')
    aggregated = []
    selections = [('pooled','all',frame)] + [('record',n,s) for n,s in frame.groupby('record')] + [('fold',str(n),s) for n,s in frame.groupby('fold')]
    for scope, group, values in selections:
        for (model, component, window), s in values.groupby(['model','component','window'], sort=False):
            aggregated.append(dict(scope=scope, group=group, model=model, component=component, window=window,
                r2=1-s.sse.sum()/max(s.sst.sum(),1e-12), rmse=np.sqrt(s.sse.sum()/s.n_values.sum()),
                sse=s.sse.sum(), sst=s.sst.sum(), n_values=int(s.n_values.sum()), record_folds=len(s)))
    summary = pd.DataFrame(aggregated)
    k = ['scope','group','component','window']
    base = summary[summary.model=='M0'][k+['r2','sse']].rename(columns={'r2':'template_r2','sse':'template_sse'})
    summary = summary.merge(base, on=k, validate='many_to_one')
    summary['delta_r2'] = summary.r2-summary.template_r2
    summary['delta_sse'] = summary.template_sse-summary.sse
    summary.to_csv(out/'ablation_v3.csv', index=False, encoding='utf-8-sig')
    summary[summary.model.isin(['M4','M5','geometry_prior_forward'])].to_csv(out/'forward_metrics_v3.csv', index=False, encoding='utf-8-sig')
    summary[summary.model.str.startswith('pool_')].to_csv(out/'pooling_metrics_v3.csv', index=False, encoding='utf-8-sig')
