"""唯一主入口：首次完整计算；相同代码和数据再次运行自动复用结果并续跑置换。"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse
import hashlib
import importlib.metadata
import json
import logging
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from eeg_model2.config import Config
from eeg_model2.data import load_records
from eeg_model2.shape import GaborEncoder
from eeg_model2.response import SourceBank,SETTINGS,FEATURE_SETTING
from eeg_model2.features import PRIMARY_SETTING,PROTOCOLS
from eeg_model2.evaluation import Engine,write_json,classification_metrics,waveform_metrics,permutation_test,WAVE_MODELS
from eeg_model2.diagnostics import trend_metrics
from eeg_model2.report import fitted_curves,figures,make_report

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--version',choices=('v2','v3'),default='v3')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--permutations',type=int,default=200)
    parser.add_argument('--ecg-permutations',type=int,default=200,help='V3 ECG敏感性独立置换次数')
    args=parser.parse_args()
    if args.permutations<0 or args.ecg_permutations<0:raise ValueError('置换次数不可为负')
    if args.output is None:
        args.output=ROOT/('outputs/results_v3' if args.version=='v3' else 'outputs/results_v2_recompute')
    if args.version=='v3':
        from eeg_model2.pipeline_v3 import run_v3
        return run_v3(ROOT,args)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log',encoding='utf-8')])
    cfg=Config();records=load_records(args.data,cfg);labels=[r.labels.copy() for r in records]
    encoder=GaborEncoder();bank=SourceBank(records[0],encoder.phi,cfg)
    code=sorted(ROOT.glob('eeg_model2/*.py'))+[ROOT/'main.py',ROOT/'predict.py']
    signature=dict(config=cfg.to_dict(),python=sys.version,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','matplotlib','scikit-learn','numba')},
                   data_sha256={r.name:r.audit['sha256'] for r in records},
                   code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
                   response_settings=SETTINGS,feature_response_setting=FEATURE_SETTING,
                   primary_classifier=PRIMARY_SETTING,classification_protocols=PROTOCOLS,schema_version=2)
    signature=json.loads(json.dumps(signature));manifest=out/'manifest.json'
    previous=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    if previous and previous['signature']!=signature:
        raise ValueError('结果目录属于不同代码/数据/环境，请指定新的--output，不能混用旧成绩')
    current=dict(signature=signature,executable=sys.executable,status='running',started_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest,current);write_json(out/'data_audit.json',[r.audit for r in records])
    pd.DataFrame(encoder.checks()).to_csv(out/'shape_checks.csv',index=False,encoding='utf-8-sig')
    engine=Engine(records,bank,cfg)
    required=['oof_predictions.csv','heldout_waveforms.npz','heldout_erp.npz','final_model.json',
              'fitted_curves.npz','curve_fit_metrics.csv','erp_validation.csv','classifier_audit.json',
              'training_audit.json','response_parameters.json','confound_audit.json','trend_metrics.csv']
    # 只有完整且内容指纹一致的结果才续跑；部分中断或输出损坏会重新计算。
    saved=previous.get('artifact_sha256',{})
    resume=all((out/name).exists() and saved.get(name)==hashlib.sha256((out/name).read_bytes()).hexdigest()
               for name in required)
    if resume:
        logging.info('代码/数据一致，复用留出结果及全数据展示模型')
        predictions=pd.read_csv(out/'oof_predictions.csv')
        z=np.load(out/'heldout_waveforms.npz')
        waves={name:[z[r.name+'__'+name] for r in records] for name in WAVE_MODELS}
        model=json.loads((out/'final_model.json').read_text(encoding='utf-8'))
        curves=dict(np.load(out/'fitted_curves.npz'))
        fit_metrics=pd.read_csv(out/'curve_fit_metrics.csv')
    else:
        logging.info('开始333 Trial的外层/内层验证：第一问处理保持不变')
        predictions,waves=engine.run(labels,out)
        model,response,cells=engine.final_model(labels)
        write_json(out/'final_model.json',model)
        fit_metrics,curves=fitted_curves(records,bank,response,cells)
        fit_metrics.to_csv(out/'curve_fit_metrics.csv',index=False,encoding='utf-8-sig')
        np.savez_compressed(out/'fitted_curves.npz',**curves)
        pd.DataFrame(trend_metrics(records,cells)).to_csv(out/'trend_metrics.csv',index=False,encoding='utf-8-sig')
    current['artifact_sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in required}
    write_json(manifest,current)
    metrics=classification_metrics(predictions,cfg);metrics.to_csv(out/'classification_metrics.csv',index=False,encoding='utf-8-sig')
    metrics[metrics.variant.isin(['main','ecg_adjusted','qc_adjusted','order_adjusted','all_adjusted','prestim_only'])].to_csv(out/'confound_metrics.csv',index=False,encoding='utf-8-sig')
    waveform=waveform_metrics(records,waves);waveform.to_csv(out/'waveform_metrics.csv',index=False,encoding='utf-8-sig')
    erp=pd.read_csv(out/'erp_validation.csv')
    figures(records,encoder,metrics,predictions,curves,out)
    make_report(metrics,waveform,fit_metrics,erp,[],model,out,ROOT/'references')
    permutation=permutation_test(engine,labels,predictions,args.permutations,out)
    make_report(metrics,waveform,fit_metrics,erp,permutation,model,out,ROOT/'references')
    current.update(status='complete',completed_permutations=args.permutations,completed_utc=datetime.now(timezone.utc).isoformat(),
                   artifact_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in required})
    write_json(manifest,current)
    print(metrics[(metrics.variant=='main')][['group','n','ba','accuracy','conditional_low','conditional_high']].to_string(index=False))
    logging.info('完成：%s',out/'第二问结果报告.md')


if __name__=='__main__':main()
