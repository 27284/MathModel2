"""Versioned V3 artifacts and frozen-input audits."""
import hashlib
import importlib.metadata
import json
import logging
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import Config
from .data import load_records
from .shape import GaborEncoder
from .response import SourceBank
from .evaluation import write_json, classification_metrics, permutation_test
from .evaluation_v3 import V3Engine, LAMBDAS, POOLS
from .report_v3 import figures_v3, report_v3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_v3(root, args):
    out=args.output.resolve()
    if out == (root/'outputs/results').resolve():
        raise ValueError('V3 不可写入 V2 归档 outputs/results；请使用 results_v3 或新的目录')
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log',encoding='utf-8')])
    cfg=Config();records=load_records(args.data,cfg)
    locked=json.loads((root/'references/review_source_lock.json').read_text(encoding='utf-8'))
    for name,digest in locked['frozen_code'].items():
        if sha(root/name)!=digest:raise ValueError(f'冻结预处理代码变化: {name}')
    if {r.name:r.audit['sha256'] for r in records} != locked['raw_data']:
        raise ValueError('冻结数据哈希变化')
    if [int(r.valid.sum()) for r in records] != [83,86,85,79]:
        raise ValueError('冻结 Trial 集合变化')
    signature=dict(schema_version=3,config=cfg.to_dict(),python=sys.version,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','matplotlib','scikit-learn','numba')},
        code_sha256={str(p.relative_to(root)):sha(p) for p in sorted((root/'eeg_model2').glob('*.py'))+[root/'main.py',root/'predict.py']},
        data_sha256=locked['raw_data'],lambda_grid=LAMBDAS,pool_grid=POOLS,
        protocol='staged_inner_CV_250_500_joint_plus_contrast; main_ladder_pool0; frozen_temporal24_LR_C0.1')
    signature=json.loads(json.dumps(signature))
    manifest=out/'manifest.json';previous=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    if previous and previous.get('signature')!=signature:
        raise ValueError('结果目录代码/数据/环境不同，请指定新的 --output；不覆盖旧结果')
    required=['component_metrics_v3.csv','ablation_v3.csv','foldwise_delta_sse_v3.csv',
              'forward_metrics_v3.csv','forward_parameters_v3.csv','pooling_metrics_v3.csv',
              'slow_decomposition_v3.csv','heldout_erp_v3.npz','response_parameters_v3.json',
              'oof_predictions.csv','confound_audit_v3.json','classifier_audit_v3.json',
              'nuisance_status_v3.csv','final_model.json']
    saved=previous.get('artifact_sha256',{})
    resume=all((out/n).exists() and saved.get(n)==sha(out/n) for n in required)
    current=dict(signature=signature,status='running',started_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest,current)
    write_json(out/'data_audit.json',[dict(**r.audit,valid_trials=np.flatnonzero(r.valid),folds=r.folds) for r in records])
    bank=SourceBank(records[0],GaborEncoder().phi,cfg);engine=V3Engine(records,bank,cfg)
    if resume:
        logging.info('V3 fingerprints match; reuse response and classification artifacts')
        pred=pd.read_csv(out/'oof_predictions.csv')
    else:
        pred=engine.run_classification(out)
        # Freeze the complete out-of-fold classification predictions, not just rounded BA.
        baseline_path=root/'outputs/results/oof_predictions.csv'
        if baseline_path.exists():
            baseline=pd.read_csv(baseline_path)
            a=pred[pred.variant=='main'].sort_values(['record','trial'])
            b=baseline[baseline.variant=='main'].sort_values(['record','trial'])
            if len(a)!=333 or len(b)!=333:raise ValueError('分类冻结核验数量不符')
            for column in ('record','trial','fold','truth','prediction'):
                if not np.array_equal(a[column].to_numpy(),b[column].to_numpy()):
                    raise ValueError(f'主分类冻结核验失败: {column}')
            np.testing.assert_allclose(a.right_probability,b.right_probability,atol=1e-12,rtol=0)
        engine.run_responses(out)
    current['artifact_sha256']={n:sha(out/n) for n in required};write_json(manifest,current)
    metrics=classification_metrics(pred,cfg)
    metrics['coverage_complete']=[int(row.n)==int(metrics[(metrics.variant=='main')&(metrics.group==row.group)].n.iloc[0]) for _,row in metrics.iterrows()]
    metrics.to_csv(out/'classification_metrics.csv',index=False,encoding='utf-8-sig')
    metrics[metrics.variant!='main'].to_csv(out/'confound_metrics.csv',index=False,encoding='utf-8-sig')
    figures_v3(out)
    labels=[r.labels.copy() for r in records]
    permutation=permutation_test(engine,labels,pred,args.permutations,out)
    engine.ecg_permutation_test(pred,args.ecg_permutations,out)
    report_v3(out,permutation)
    current.update(status='complete',completed_permutations=args.permutations,
                   requested_ecg_permutations=args.ecg_permutations,
                   completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest,current)
    logging.info('V3 complete: %s',out/'第二问V3结果报告.md')
