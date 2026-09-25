import unittest
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from eeg_model2.config import Config
from eeg_model2.data import load_records,onsets_and_ends
from eeg_model2.shape import GaborEncoder,dog
from eeg_model2.neural import stability,PRIOR
from eeg_model2.preprocess import prepare_fold,quality_weights
from eeg_model2.response import SourceBank,targets,fit_response,FEATURE_SETTING
from eeg_model2.features import build_features,apply_transform,time_features,fit_classifier,PROTOCOLS
from eeg_model2.evaluation import restricted_permutation
from eeg_model2.diagnostics import residualize,ecg_adjust

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('EEG_TEST_DATA',str(ROOT/'data')))
OUT=ROOT/'outputs/results'


class BasicTests(unittest.TestCase):
    def test_onset_plateaus(self):
        a,b=onsets_and_ends(np.array([0,0,-1,-1,0,1,1,0]))
        np.testing.assert_array_equal(a,[2,5]);np.testing.assert_array_equal(b,[4,7])

    def test_shape_mirror_and_dc(self):
        enc=GaborEncoder()
        np.testing.assert_allclose(enc.phi[0],enc.phi[1,::-1],atol=1e-12)
        self.assertGreater(enc.phi[0,0],enc.phi[0,1])
        np.testing.assert_allclose(dog(np.ones((65,65))),0,atol=1e-12)

    def test_neural_stable_and_config_roundtrip(self):
        self.assertTrue(stability(PRIOR)['stable'])
        self.assertEqual(Config().to_dict(),json.loads(json.dumps(Config().to_dict())))

    def test_time_feature_channel_order(self):
        t=np.arange(-51,206)/256
        x=np.broadcast_to(np.array([1.,2.,3.])[None,:,None],(2,3,len(t)))
        np.testing.assert_array_equal(time_features(x,t),np.tile(np.repeat([1.,2.,3.],8),(2,1)))

    def test_primary_ignores_qc_weights(self):
        rng=np.random.default_rng(41);x=rng.normal(size=(40,24));y=np.tile([-1,1],20)
        a=fit_classifier(x,y,np.ones(40));b=fit_classifier(x,y,np.linspace(.05,1,40))
        self.assertEqual(a,b)
        self.assertFalse(a['qc_weighted']);self.assertEqual(a['C'],.1)

    def test_confound_regression_excludes_test_covariates_and_targets(self):
        rng=np.random.default_rng(11);c=rng.normal(size=(40,3));x=c@rng.normal(size=(3,24))+rng.normal(size=(40,24))
        train=np.arange(40)<30
        a,ma=residualize(x,c,train)
        xx=x.copy();cc=c.copy();xx[~train]+=100000;cc[~train]*=1000
        b,mb=residualize(xx,cc,train)
        self.assertEqual(ma,mb);np.testing.assert_array_equal(a[train],b[train])


@unittest.skipUnless((DATA/'VisualCogA_Task-1.mat').exists(),'需要原始数据')
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=Config();cls.records=load_records(DATA,cls.cfg)
        cls.bank=SourceBank(cls.records[0],GaborEncoder().phi,cls.cfg)
        cls.prepared=[prepare_fold(r,{1,2,3,4},cls.cfg) for r in cls.records]
        cls.labels=[r.labels.copy() for r in cls.records]

    def test_same_valid_cohort_and_qc_boundary(self):
        self.assertEqual([int(r.valid.sum()) for r in self.records],[83,86,85,79])
        for r,p in zip(self.records,self.prepared):
            metrics=r.metrics.copy();metrics[r.folds==0]*=1000
            _,center,scale=quality_weights(metrics,p.train,r.valid,self.cfg)
            np.testing.assert_array_equal(center,p.audit['qc_center']);np.testing.assert_array_equal(scale,p.audit['qc_scale'])

    def test_response_shape_and_baseline(self):
        for structure in ('legacy','transient_contrast'):
            b=self.bank.basis(52/256,structure)
            self.assertEqual(b.shape,(2,6,257))
            self.assertTrue(np.isfinite(b).all())
            np.testing.assert_allclose(b[...,self.bank.t<0].mean(axis=-1),0,atol=1e-10)

    def test_response_candidate_only_changes_contrast_tail_basis(self):
        a=self.bank.basis(52/256,'legacy');b=self.bank.basis(52/256,'transient_contrast')
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1,:4],b[1,:4])
        self.assertGreater(np.linalg.norm(a[1,4:]-b[1,4:]),1)

    def test_ecg_regression_excludes_test_data_and_preserves_input(self):
        import copy
        r=self.records[0];train=self.prepared[0].train
        original=r.light.copy();a,ma=ecg_adjust(r,train)
        altered=copy.copy(r);altered.light=r.light.copy();altered.ecg=r.ecg.copy()
        altered.light[~train]+=10000;altered.ecg[~train]*=100
        b,mb=ecg_adjust(altered,train)
        self.assertEqual(ma,mb);np.testing.assert_array_equal(a[train],b[train])
        np.testing.assert_array_equal(original,r.light)

    def test_heldout_labels_cannot_change_fit_or_features(self):
        changed=[np.where(r.folds==0,-y,y) for r,y in zip(self.records,self.labels)]
        ca=targets(self.records,self.prepared,self.labels,self.cfg)
        cb=targets(self.records,self.prepared,changed,self.cfg)
        for a,b in zip(ca,cb):np.testing.assert_array_equal(a,b)
        a=fit_response(self.records,self.bank,ca,FEATURE_SETTING)
        b=fit_response(self.records,self.bank,cb,FEATURE_SETTING)
        np.testing.assert_array_equal(a.coef,b.coef)
        fa,ta=build_features(self.records,self.prepared,self.labels,self.bank,a,ca)
        fb,tb=build_features(self.records,self.prepared,changed,self.bank,b,cb)
        for j,r in enumerate(self.records):
            np.testing.assert_array_equal(fa['contrast25'][j],fb['contrast25'][j])
            self.assertTrue(np.isfinite(fa['contrast25'][j]).all())
            for family in fa:np.testing.assert_allclose(fa[family][j],apply_transform(r.light,r.t,ta[j],family),atol=1e-12)

    def test_common_contrast_identity_and_serialization(self):
        cells=targets(self.records,self.prepared,self.labels,self.cfg)
        model=fit_response(self.records,self.bank,cells,FEATURE_SETTING)
        p=model.predict(self.bank,0,[-1,1],[.203125]*2)
        common=model.predict(self.bank,0,[-1,1],[.203125]*2,True)
        np.testing.assert_allclose(p.mean(axis=0),common[0],atol=1e-10)
        self.assertGreater(model.effective_df,0)
        self.assertLessEqual(model.effective_df,model.coef.size+1e-6)
        self.assertTrue(np.isfinite(np.array(json.loads(json.dumps(model.to_dict()))['coefficients'])).all())

    def test_permutation_preserves_block_counts(self):
        sy=restricted_permutation(self.records,self.labels,np.random.default_rng(1))
        for r,y,s in zip(self.records,self.labels,sy):
            for block in range(5):
                use=r.valid&(r.folds==block)
                np.testing.assert_array_equal(np.sort(y[use]),np.sort(s[use]))
            np.testing.assert_array_equal(y[~r.valid],s[~r.valid])


@unittest.skipUnless((OUT/'final_model.json').exists(),'先运行main.py')
class ResultsTests(unittest.TestCase):
    def test_oof_coverage_and_reported_metrics(self):
        pred=pd.read_csv(OUT/'oof_predictions.csv');metrics=pd.read_csv(OUT/'classification_metrics.csv')
        for variant,s in pred.groupby('variant'):
            self.assertEqual(len(s),333);self.assertFalse(s.duplicated(['record','trial']).any())
            for group in ('all','task1','task2',*s.record.unique()):
                q=s if group=='all' else s[s.task==int(group[-1])] if group.startswith('task') else s[s.record==group]
                reported=metrics[(metrics.group==group)&(metrics.variant==variant)].ba.iloc[0]
                self.assertAlmostEqual(reported,balanced_accuracy_score(q.truth,q.prediction),places=12)
        self.assertEqual(set(pred.variant),set(PROTOCOLS))

    def test_training_excludes_outer_and_selection_uses_inner(self):
        for a in json.loads((OUT/'training_audit.json').read_text(encoding='utf-8')):
            self.assertNotIn(a['fold'],a['training_blocks'])
            self.assertTrue(all(int(i)//20!=a['fold'] for i in a['training_trials']))
        for a in json.loads((OUT/'classifier_audit.json').read_text(encoding='utf-8')):
            self.assertEqual(a['selection'],'fixed_before_evaluation')
            self.assertNotIn(a['fold'],a['training_blocks'])
            for trials in a['training_trials'].values():self.assertTrue(all(i//20!=a['fold'] for i in trials))
            if a['variant']=='main':
                self.assertEqual(len(a['records']),1);self.assertFalse(a['classifier']['qc_weighted'])
                self.assertEqual(a['classifier']['estimator'],'lr');self.assertEqual(a['classifier']['C'],.1)
        for a in json.loads((OUT/'response_parameters.json').read_text(encoding='utf-8')):
            selected=min(a['candidates'],key=lambda row:row['inner_loss'])['setting']
            self.assertEqual([a['main']['ridge'],a['main']['pool'],a['main']['structure']],selected)

    def test_confound_audit_excludes_outer_fold(self):
        for a in json.loads((OUT/'confound_audit.json').read_text(encoding='utf-8')):
            for name in ('ecg','qc','order','combined'):
                self.assertTrue(all(i//20!=a['fold'] for i in a[name]['training_trials']))

    def test_waveform_baselines_have_identical_scoring_targets(self):
        frame=pd.read_csv(OUT/'erp_validation.csv')
        for _,rows in frame.groupby(['fold','record','split','component','window']):
            self.assertEqual(set(rows.model),{'response','legacy_response','train_erp','common_only'})
            np.testing.assert_allclose(rows.sst,rows.sst.iloc[0],rtol=0,atol=1e-9)
        template=frame[(frame.model=='train_erp')&(frame.split=='train')]
        np.testing.assert_allclose(template.sse,0,atol=0)

    def test_legacy_erp_scores_reproduce_previous_model(self):
        baseline=pd.read_csv(ROOT/'references/review_baseline_erp_validation.csv')
        current=pd.read_csv(OUT/'erp_validation.csv')
        current=current[(current.model=='legacy_response')&(current.component=='joint')&(current.window=='0_800')]
        merged=baseline.merge(current,on=['fold','record','split'],suffixes=('_old','_new'))
        self.assertEqual(len(merged),40)
        np.testing.assert_allclose(merged.r2_old,merged.r2_new,rtol=1e-8,atol=1e-9)

    def test_saved_outer_classifier_reproduces_oof(self):
        from eeg_model2.features import classify
        audits=json.loads((OUT/'classifier_audit.json').read_text(encoding='utf-8'))
        pred=pd.read_csv(OUT/'oof_predictions.csv');z=np.load(OUT/'heldout_waveforms.npz')
        for a in audits:
            if a['variant']!='main':continue
            record=a['records'][0]
            rows=pred[(pred.variant=='main')&(pred.record==record)&(pred.fold==a['fold'])]
            x=time_features(z[record+'__actual'][rows.trial.to_numpy()-1],z['time_s'])
            label,prob=classify(a['classifier'],x)
            np.testing.assert_array_equal(label,rows.prediction);np.testing.assert_allclose(prob,rows.right_probability,atol=1e-14)

    def test_permutation_coverage_and_p_values(self):
        scores=pd.read_csv(OUT/'permutation_scores.csv')
        results=json.loads((OUT/'permutation_test.json').read_text(encoding='utf-8'))
        for row in results:
            s=scores[(scores.group==row['group'])&(scores.permutation<row['permutations'])]
            self.assertEqual(len(s),row['permutations']);self.assertFalse(s.permutation.duplicated().any())
            if len(s):self.assertAlmostEqual(row['p_value'],(1+(s.ba>=row['ba']).sum())/(1+len(s)),places=12)

    def test_unknown_trial_prediction_without_true_label(self):
        from predict import predict
        model=json.loads((OUT/'final_model.json').read_text(encoding='utf-8'))
        z=np.load(OUT/'heldout_waveforms.npz');rec=model['records'][0]
        x=z[rec+'__actual'][z[rec+'__valid']][:5]
        label,prob,features=predict(x,model,rec)
        self.assertEqual(len(model['classifiers']),4)
        self.assertEqual({c['record'] for c in model['classifiers']},set(model['records']))
        self.assertEqual(len(label),5);self.assertTrue(np.isfinite(features).all())
        self.assertTrue(((prob>=0)&(prob<=1)).all())
        with self.assertRaises(ValueError):predict(x,model,'unknown')

    def test_result_sources_match_current_code(self):
        import hashlib
        manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        # V2 is a historical archive after V3: its artifacts retain their original
        # fingerprints. Current V3 source fingerprints are checked in test_v3.py.
        self.assertEqual(manifest['signature']['schema_version'],2)
        for file,value in manifest['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((OUT/file).read_bytes()).hexdigest(),value)

    def test_frozen_preprocessing_and_raw_data(self):
        import hashlib
        locked=json.loads((ROOT/'references/review_source_lock.json').read_text(encoding='utf-8'))
        for name,value in locked['frozen_code'].items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),value)
        audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
        self.assertEqual({r['file'].replace('.mat',''):r['sha256'] for r in audit},locked['raw_data'])


if __name__=='__main__':unittest.main()
