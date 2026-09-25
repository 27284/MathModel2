"""V3 regression, isolation and numerical contract tests."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from eeg_model2.slow import fit_slow, SLOW_SETTINGS, slow_design
from eeg_model2.basis import stretch_basis, BasisBank, WARP_GROUPS
from eeg_model2.forward import geometry_prior, fit_constrained_forward, apply_forward
from eeg_model2.response_v3 import Setting, fit_response_v3
from eeg_model2.config import Config
from eeg_model2.data import load_records
from eeg_model2.response import SourceBank
from eeg_model2.shape import GaborEncoder
from eeg_model2.evaluation_v3 import V3Engine
from eeg_model2.nuisance import nuisance_features

ROOT=Path(__file__).resolve().parents[1]


class SlowTests(unittest.TestCase):
    def test_absolute_time_and_baseline(self):
        t=np.arange(-307,462)/256
        onsets=np.arange(20.)*10+100
        absolute=onsets[:,None]+t
        y=np.stack([3+2*absolute,10-.5*absolute,4+.002*absolute**2],axis=1)
        model=fit_slow(y,onsets,'quadratic',t)
        test_on=np.array([75.,350.]);core=np.arange(-51,206)/256
        prediction=model.predict(test_on,core,baseline=False)
        a=test_on[:,None]+core
        expected=np.stack([3+2*a,10-.5*a,4+.002*a**2],axis=1)
        np.testing.assert_allclose(prediction,expected,atol=1e-9)
        centered=model.predict(test_on,core)
        np.testing.assert_allclose(centered[...,core<0].mean(axis=-1),0,atol=1e-10)
        self.assertGreater(np.linalg.norm(centered[0,2]-centered[1,2]),1.)

    def test_spline_four_df_and_linear_extrapolation(self):
        x=np.array([-3.,-2.,-1.,0.,1.,2.,3.,4.])
        b=slow_design(x,'natural_spline_df4',0.,1.,[-1.,0.,1.,2.])
        self.assertEqual(b.shape,(8,4))
        np.testing.assert_allclose(b[7]-2*b[6]+b[5],0,atol=1e-12)
        np.testing.assert_allclose(b[0]-2*b[1]+b[2],0,atol=1e-12)

    def test_shift_stretch_definition(self):
        t=np.linspace(-.3,1.2,1501);base=np.exp(-.5*((t-.3)/.04)**2)[None]
        warped=stretch_basis(base,t,1.4,.08)
        self.assertAlmostEqual(t[np.argmax(warped)],.5,places=3)


class ForwardTests(unittest.TestCase):
    def test_true_two_source_symmetry_and_recovery(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(400,2))
        g=np.array([[.4,.4],[1.,.2],[.2,1.]])
        fit=fit_constrained_forward(x,apply_forward(g,x),strength=0.)
        np.testing.assert_allclose(g,fit,atol=1e-12)
        self.assertEqual(fit[0,0],fit[0,1]);self.assertEqual(fit[1,0],fit[2,1])

    def test_geometry_correction_shrinks_to_prior(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(400,2));y=rng.normal(size=(400,3))
        a=fit_constrained_forward(x,y,'geometry_prior_forward',.1)
        b=fit_constrained_forward(x,y,'geometry_prior_forward',100.)
        self.assertLess(np.linalg.norm(b-geometry_prior()),np.linalg.norm(a-geometry_prior()))


class V3DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=Config();cls.records=load_records(ROOT/'data',cls.cfg)
        cls.source=SourceBank(cls.records[0],GaborEncoder().phi,cls.cfg)
        cls.engine=V3Engine(cls.records,cls.source,cls.cfg)

    def test_distinct_basis_and_baseline(self):
        bank=BasisBank(self.source)
        for warp in WARP_GROUPS:
            common,contrast=bank.get(.203125,warp=warp)
            self.assertGreater(len(common),len(contrast))
            for b in (common,contrast):
                self.assertTrue(np.isfinite(b).all())
                np.testing.assert_allclose(b[:,bank.t<0].mean(axis=1),0,atol=1e-10)
        self.assertEqual(bank.get(.203125)[0].shape[0],3)
        self.assertEqual(bank.get(.203125)[1].shape[0],2)

    def test_all_learning_excludes_heldout_signals_onsets_and_labels(self):
        records=copy.deepcopy(self.records)
        for r in records:
            held=r.folds==0
            r.light[held]+=100000
            r.raw_epochs[held]*=1000
            r.ecg[held]*=1000
            r.metrics[held]*=1000
            r.onsets[held]+=1000000
            r.labels[held]*=-1
        changed=V3Engine(records,self.source,self.cfg)
        blocks={1,2,3,4}
        for slow in SLOW_SETTINGS:
            a=self.engine.cells(blocks,slow);b=changed.cells(blocks,slow)
            # Predictions for test onset can change, but training calibration cannot.
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual([m.to_dict() for m in self.engine.slow_models(blocks,slow)],
                             [m.to_dict() for m in changed.slow_models(blocks,slow)])
        setting=Setting(slow='quadratic',warp='stretch',lambda_common=.01,lambda_contrast=.1,
                        forward='geometry_prior_forward')
        a=self.engine.fit(blocks,setting);b=changed.fit(blocks,setting)
        self.assertEqual(a.to_dict(),b.to_dict())
        selected_a, audit_a = self.engine.stages(blocks)
        selected_b, audit_b = changed.stages(blocks)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(audit_a, audit_b)
        a,aa=nuisance_features(self.records[0],self.engine.prepare(blocks)[0].train,256)
        b,bb=nuisance_features(records[0],changed.prepare(blocks)[0].train,256)
        self.assertEqual(aa,bb)
        for key in a:
            np.testing.assert_array_equal(a[key][self.engine.prepare(blocks)[0].train],b[key][self.engine.prepare(blocks)[0].train])

    def test_failed_ecg_gate_is_not_silently_scored_as_adjusted(self):
        record=copy.copy(self.records[0]);record.ecg=np.zeros_like(record.ecg)
        features,audit=nuisance_features(record,self.engine.prepare({1,2,3,4})[0].train,256)
        self.assertFalse(audit['ecg_verification']['usable_ecg_like'])
        self.assertNotIn('ecg_adjusted',features);self.assertNotIn('all_adjusted',features)
        self.assertIn('qc_adjusted',features);self.assertIn('order_adjusted',features)

    def test_independent_penalties_and_component_identity(self):
        blocks={1,2,3,4}
        a=self.engine.fit(blocks,Setting(lambda_common=.1,lambda_contrast=.1))
        b=self.engine.fit(blocks,Setting(lambda_common=.1,lambda_contrast=.001))
        np.testing.assert_array_equal(a.coefficients[0],b.coefficients[0])
        self.assertGreater(np.linalg.norm(a.coefficients[1]-b.coefficients[1]),1e-4)
        cells=self.engine.cells(blocks,'none',{0})
        pred=self.engine.prediction(a,cells,0)
        neural=a.neural(self.engine.basis,0,cells['durations'][0])
        np.testing.assert_allclose(pred.mean(axis=0),neural[0],atol=1e-10)
        np.testing.assert_allclose((pred[1]-pred[0])/2,neural[1],atol=1e-10)


@unittest.skipUnless((ROOT/'outputs/results_v3/ablation_v3.csv').exists(),'Run V3 first')
class V3ResultsTests(unittest.TestCase):
    def test_complete_fair_targets_and_paired_deltas(self):
        out=ROOT/'outputs/results_v3'
        frame=pd.read_csv(out/'component_metrics_v3.csv')
        for _,s in frame.groupby(['record','fold','component','window']):
            self.assertTrue(set(f'M{i}' for i in range(6)).issubset(set(s.model)))
            np.testing.assert_allclose(s.sst,s.sst.iloc[0],rtol=0,atol=1e-9)
        paired=pd.read_csv(out/'foldwise_delta_sse_v3.csv')
        np.testing.assert_allclose(paired.delta_sse,paired.template_sse-paired.sse,atol=1e-8)
        np.testing.assert_allclose(paired.delta_r2,paired.r2-paired.template_r2,atol=1e-10)
        summary=pd.read_csv(out/'ablation_v3.csv')
        for _,r in summary[summary.scope=='pooled'].iterrows():
            s=frame[(frame.model==r.model)&(frame.component==r.component)&(frame.window==r.window)]
            self.assertEqual(len(s),20)
            self.assertAlmostEqual(r.r2,1-s.sse.sum()/s.sst.sum(),places=10)

    def test_frozen_classification_and_current_manifest(self):
        out=ROOT/'outputs/results_v3'
        old=pd.read_csv(ROOT/'outputs/results/oof_predictions.csv')
        new=pd.read_csv(out/'oof_predictions.csv')
        cols=['record','trial','fold','truth','prediction','right_probability']
        a=old[old.variant=='main'].sort_values(['record','trial'])[cols].reset_index(drop=True)
        b=new[new.variant=='main'].sort_values(['record','trial'])[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,atol=1e-12,rtol=0)
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        for name,digest in manifest['signature']['code_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
        for name,digest in manifest['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),digest)

    def test_outer_audit_and_selected_inner_minimum(self):
        path=ROOT/'outputs/results_v3/response_parameters_v3.json'
        for fold in json.loads(path.read_text(encoding='utf-8')):
            self.assertNotIn(fold['fold'],fold['training_blocks'])
            for blocks in fold['inner_training_blocks']:
                self.assertNotIn(fold['fold'],blocks);self.assertEqual(len(blocks),3)
            for name,scores in fold['selection'].items():
                best=min(scores,key=lambda r:(r['inner_loss'],r['inner_full_loss']))
                self.assertEqual(best['setting'],fold['models'][name]['setting'])

    def test_ecg_permutation_counts_pvalues_and_separate_seed(self):
        out=ROOT/'outputs/results_v3'
        result=json.loads((out/'ecg_permutation_test.json').read_text(encoding='utf-8'))
        if not isinstance(result,list):
            self.assertEqual(result['status'],'not_run_incomplete_ecg_gate_coverage')
            return
        scores=pd.read_csv(out/'ecg_permutation_scores.csv')
        for row in result:
            self.assertEqual(row['seed_offset'],20000)
            values=scores[(scores.group==row['group'])&(scores.permutation<row['permutations'])]
            self.assertEqual(len(values),row['permutations'])
            self.assertFalse(values.permutation.duplicated().any())
            if len(values):
                self.assertAlmostEqual(row['p_value'],(1+(values.ba>=row['ba']).sum())/(1+len(values)),places=12)

    def test_v3_saved_classifier_remains_predict_compatible(self):
        from predict import predict
        out=ROOT/'outputs/results_v3'
        model=json.loads((out/'final_model.json').read_text(encoding='utf-8'))
        self.assertEqual(model['response_version'],3)
        archive=np.load(ROOT/'outputs/results/heldout_waveforms.npz')
        record=model['records'][0]
        epochs=archive[record+'__actual'][archive[record+'__valid']][:4]
        label,probability,features=predict(epochs,model,record)
        self.assertEqual(features.shape,(4,24));self.assertEqual(label.shape,(4,))
        self.assertTrue(np.isfinite(probability).all())


if __name__=='__main__':unittest.main()
