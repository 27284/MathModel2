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
from eeg_model2.features import build_features,apply_transform
from eeg_model2.evaluation import restricted_permutation

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
        b=self.bank.basis(52/256)
        self.assertEqual(b.shape,(2,6,257))
        self.assertTrue(np.isfinite(b).all())
        np.testing.assert_allclose(b[...,self.bank.t<0].mean(axis=-1),0,atol=1e-10)

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
            for task in (1,2):
                q=s[s.task==task]
                reported=metrics[(metrics.task==task)&(metrics.variant==variant)].ba.iloc[0]
                self.assertAlmostEqual(reported,balanced_accuracy_score(q.truth,q.prediction),places=12)

    def test_training_excludes_outer_and_selection_uses_inner(self):
        for a in json.loads((OUT/'training_audit.json').read_text(encoding='utf-8')):
            self.assertNotIn(a['fold'],a['training_blocks'])
            self.assertTrue(all(int(i)//20!=a['fold'] for i in a['training_trials']))
        for a in json.loads((OUT/'classifier_selection.json').read_text(encoding='utf-8')):
            expected=max(a['candidates'],key=lambda r:r['inner_ba'])['setting']
            self.assertEqual(a['setting'],expected)

    def test_unknown_trial_prediction_without_true_label(self):
        from predict import predict
        model=json.loads((OUT/'final_model.json').read_text(encoding='utf-8'))
        z=np.load(OUT/'heldout_waveforms.npz');rec=model['records'][0]
        x=z[rec+'__actual'][z[rec+'__valid']][:5]
        label,prob,features=predict(x,model,rec)
        self.assertEqual(len(label),5);self.assertTrue(np.isfinite(features).all())
        self.assertTrue(((prob>=0)&(prob<=1)).all())
        with self.assertRaises(ValueError):predict(x,model,'unknown')

    def test_result_sources_match_current_code(self):
        import hashlib
        manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        for file,value in manifest['signature']['code_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/file).read_bytes()).hexdigest(),value)


if __name__=='__main__':unittest.main()
