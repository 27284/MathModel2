"""预测未知Trial，不接受真实标签；输入须沿用相同轻处理，指定已校准记录。"""
from pathlib import Path
import argparse
import json
import numpy as np
from eeg_model2.features import apply_transform,classify


def predict(epochs,model,record):
    if model.get('schema_version')!=2:raise ValueError('模型格式不匹配，请使用当前main.py生成的逐记录模型')
    epochs=np.asarray(epochs,float);t=np.asarray(model['time_s'])
    if epochs.ndim!=3 or epochs.shape[1:]!=(3,len(t)) or not np.isfinite(epochs).all():
        raise ValueError(f'epochs必须为有限N×3×{len(t)}数组，通道Fz/F3/F4')
    if not np.allclose(epochs[...,t<0].mean(axis=-1),0,atol=1e-5):raise ValueError('需先执行相同基线校正')
    try:transform=next(m for m in model['transforms'] if m['record']==record)
    except StopIteration:raise ValueError('未知记录：需要独立训练校准，不能套用A/B记录参数') from None
    classifier=next(m for m in model['classifiers'] if m['record']==record)
    features=apply_transform(epochs,t,transform,classifier['family'])
    labels,probability=classify(classifier,features)
    return labels,probability,features


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz',type=Path,help='epochs键：已轻处理、基线校正的N×3×257数组')
    parser.add_argument('--record',required=True)
    parser.add_argument('--model',type=Path,default=Path(__file__).parent/'outputs/results/final_model.json')
    parser.add_argument('--output',type=Path,default=Path('predictions.csv'))
    a=parser.parse_args();import pandas as pd
    labels,probability,_=predict(np.load(a.npz)['epochs'],json.loads(a.model.read_text(encoding='utf-8')),a.record)
    pd.DataFrame(dict(prediction=labels,right_probability=probability)).to_csv(a.output,index=False,encoding='utf-8-sig')
