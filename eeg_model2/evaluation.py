"""全部训练相关操作都在时间块内部；置换重做模板和模型选择。"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from .preprocess import prepare_fold
from .response import targets,fit_response,validation_loss,SETTINGS as WAVE_SETTINGS,FEATURE_SETTING
from .features import build_features,concatenate,fit_classifier,classify,SETTINGS


def write_json(path,data):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=default),encoding='utf-8')


class Engine:
    def __init__(self,records,bank,cfg):
        self.records,self.bank,self.cfg=records,bank,cfg
        self.prepared={};self.learned={};self.valid_targets={}

    def prepare(self,blocks):
        key=tuple(sorted(blocks))
        if key not in self.prepared:self.prepared[key]=[prepare_fold(r,blocks,self.cfg) for r in self.records]
        return self.prepared[key]

    def learn(self,blocks,labels):
        key=tuple(sorted(blocks))
        if key not in self.learned:
            p=self.prepare(blocks)
            cells=targets(self.records,p,labels,self.cfg)
            feature_model=fit_response(self.records,self.bank,cells,FEATURE_SETTING)
            variants,transforms=build_features(self.records,p,labels,self.bank,feature_model,cells)
            self.learned[key]=(p,cells,variants,transforms)
        return self.learned[key]

    def select_classifiers(self,blocks,labels):
        inner=[(b,self.learn(blocks-{b},labels)) for b in sorted(blocks)]
        result={}
        for task in (1,2):
            scores=[]
            for setting in SETTINGS:
                truth=[];guesses=[]
                for b,(p,_,variants,_) in inner:
                    x,y,w,_=concatenate(self.records,p,labels,variants[setting[0]],task)
                    xv,yv,_,_=concatenate(self.records,p,labels,variants[setting[0]],task,{b})
                    pred,_=classify(fit_classifier(x,y,w,setting),xv)
                    truth.extend(yv);guesses.extend(pred)
                scores.append(float(balanced_accuracy_score(truth,guesses)))
            result[task]=dict(setting=SETTINGS[int(np.argmax(scores))],
                              candidates=[dict(setting=s,inner_ba=v) for s,v in zip(SETTINGS,scores)])
        return result

    def select_response(self,blocks,labels):
        inner=[]
        for b in sorted(blocks):
            p,cells,_,_=self.learn(blocks-{b},labels)
            key=(tuple(sorted(blocks-{b})),b)
            if key not in self.valid_targets:
                self.valid_targets[key]=targets(self.records,p,labels,self.cfg,{b})
            inner.append((cells,self.valid_targets[key]))
        scores=[]
        for setting in WAVE_SETTINGS:
            loss=[]
            for cells,valid in inner:
                model=fit_response(self.records,self.bank,cells,setting)
                loss.append(validation_loss(self.records,self.bank,model,cells,valid))
            scores.append(float(np.mean(loss)))
        return WAVE_SETTINGS[int(np.argmin(scores))],[dict(setting=s,inner_loss=v) for s,v in zip(WAVE_SETTINGS,scores)]

    def run(self,labels,out=None,full=True):
        self.learned.clear();self.valid_targets.clear()
        rows=[];choices=[];audits=[];parameters=[];erp_rows=[];erp_arrays=[]
        waves={name:[np.full_like(r.light,np.nan) for r in self.records] for name in ('response','common_only','train_erp')} if full else {}
        for fold in range(self.cfg.folds):
            blocks=set(range(self.cfg.folds))-{fold}
            p,cells,variants,transforms=self.learn(blocks,labels)
            selected=self.select_classifiers(blocks,labels)
            for task in (1,2):
                chosen=selected[task]['setting']
                wanted=[('main',chosen)]
                if full:
                    for family in ('amplitude4','temporal24','contrast25'):
                        candidates=[v for v in selected[task]['candidates'] if v['setting'][0]==family]
                        best=max(candidates,key=lambda z:z['inner_ba'])['setting']
                        wanted.append((family,best))
                for variant,setting in wanted:
                    x,y,w,_=concatenate(self.records,p,labels,variants[setting[0]],task)
                    xv,yv,_,indices=concatenate(self.records,p,labels,variants[setting[0]],task,{fold})
                    classifier=fit_classifier(x,y,w,setting)
                    prediction,probability=classify(classifier,xv)
                    for (j,i),truth,pred,prob in zip(indices,yv,prediction,probability):
                        rows.append(dict(record=self.records[j].name,task=task,trial=i+1,fold=fold,
                                         variant=variant,truth=int(truth),prediction=int(pred),right_probability=float(prob)))
                choices.append(dict(fold=fold,task=task,training_blocks=sorted(blocks),**selected[task]))
            if full:
                setting,scores=self.select_response(blocks,labels)
                model=fit_response(self.records,self.bank,cells,setting)
                parameters.append(dict(fold=fold,training_blocks=sorted(blocks),candidates=scores,**model.to_dict()))
                test_cells=targets(self.records,p,labels,self.cfg,{fold})
                for j,(r,prep,transform,y) in enumerate(zip(self.records,p,transforms,labels)):
                    ids=(r.folds==fold)&r.valid
                    waves['response'][j][ids]=model.predict(self.bank,j,y[ids],r.duration[ids])
                    waves['common_only'][j][ids]=model.predict(self.bank,j,y[ids],r.duration[ids],True)
                    waves['train_erp'][j][ids]=cells[0][j,(y[ids]>0).astype(int)]
                    audits.append({'fold':fold,**prep.audit,**transform})
                    for kind,curves in [('train',cells),('heldout',test_cells)]:
                        actual=curves[0][j]
                        predicted=model.predict(self.bank,j,[-1,1],[curves[2][j]]*2)
                        for sign in (0,1):
                            erp_arrays.append((fold,r.name,kind,sign,actual[sign],predicted[sign]))
                        m=self.bank.t>=0
                        sse=np.sum((actual[...,m]-predicted[...,m])**2)
                        centered=actual[...,m]-actual[...,m].mean(axis=(0,2),keepdims=True)
                        erp_rows.append(dict(fold=fold,record=r.name,task=r.task,split=kind,
                                            rmse=float(np.sqrt(np.mean((actual[...,m]-predicted[...,m])**2))),
                                            r2=float(1-sse/max(np.sum(centered**2),1e-12)),sse=float(sse),sst=float(np.sum(centered**2))))
                logging.info('外层时间块 %d/5：波形及分类验证完成',fold+1)
        pred=pd.DataFrame(rows)
        if out:
            pred.to_csv(out/'oof_predictions.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(erp_rows).to_csv(out/'erp_validation.csv',index=False,encoding='utf-8-sig')
            write_json(out/'classifier_selection.json',choices);write_json(out/'response_parameters.json',parameters)
            write_json(out/'training_audit.json',audits)
            archive={'time_s':self.bank.t}
            for j,r in enumerate(self.records):
                archive[r.name+'__valid']=r.valid
                archive[r.name+'__actual']=r.light
                for name in waves:archive[r.name+'__'+name]=waves[name][j]
            np.savez_compressed(out/'heldout_waveforms.npz',**archive)
            np.savez_compressed(out/'heldout_erp.npz',time_s=self.bank.t,
                                **{f'{fold}__{rec}__{kind}__{sign}__{name}':x
                                   for fold,rec,kind,sign,actual,predicted in erp_arrays
                                   for name,x in [('actual',actual),('predicted',predicted)]})
        return pred,waves

    def final_model(self,labels):
        self.learned.clear();self.valid_targets.clear()
        blocks=set(range(self.cfg.folds));p,cells,variants,transforms=self.learn(blocks,labels)
        selected=self.select_classifiers(blocks,labels)
        setting,scores=self.select_response(blocks,labels)
        response=fit_response(self.records,self.bank,cells,setting)
        classifiers=[]
        for task in (1,2):
            s=selected[task]['setting']
            x,y,w,_=concatenate(self.records,p,labels,variants[s[0]],task)
            classifiers.append(dict(task=task,**fit_classifier(x,y,w,s)))
        result=dict(config=self.cfg.to_dict(),time_s=self.bank.t.tolist(),records=[r.name for r in self.records],
                    response=response.to_dict(),response_candidates=scores,
                    transforms=transforms,classifiers=classifiers,source_audit=self.bank.audit(),
                    note='全数据再拟合用于展示及后续同记录条件预测，不参与留出成绩。分类模板固定正则，与波形调参分开。')
        return result,response,cells


def restricted_permutation(records,labels,rng):
    result=[]
    for r,y in zip(records,labels):
        sy=y.copy()
        for block in range(5):
            ids=np.flatnonzero(r.valid&(r.folds==block));sy[ids]=rng.permutation(sy[ids])
        result.append(sy)
    return result


def permutation_test(engine,labels,observed,count,out):
    path=out/'permutation_scores.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    done={int(r['permutation']) for r in rows}
    for index in range(count):
        if index in done:continue
        shuffled=restricted_permutation(engine.records,labels,np.random.default_rng(engine.cfg.seed+10000+index))
        pred,_=engine.run(shuffled,full=False)
        for task in (1,2):
            s=pred[pred.task==task]
            rows.append(dict(permutation=index,task=task,ba=float(balanced_accuracy_score(s.truth,s.prediction))))
        pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
        logging.info('完整分类流程置换 %d/%d',index+1,count)
    result=[]
    for task in (1,2):
        s=observed[(observed.task==task)&(observed.variant=='main')]
        obs=float(balanced_accuracy_score(s.truth,s.prediction))
        null=np.array([r['ba'] for r in rows if r['task']==task and r['permutation']<count])
        result.append(dict(task=task,ba=obs,permutations=len(null),
                           p_value=float((1+(null>=obs).sum())/(1+len(null))) if len(null) else None))
    write_json(out/'permutation_test.json',result)
    return result


def classification_metrics(pred,cfg):
    rows=[];rng=np.random.default_rng(cfg.seed)
    for task in (1,2):
        for variant,s in pred[pred.task==task].groupby('variant',sort=False):
            cm=confusion_matrix(s.truth,s.prediction,labels=[-1,1]);boot=np.zeros((cfg.bootstrap,2,2))
            for _,r in s.groupby('record'):
                blocks=np.array([confusion_matrix(b.truth,b.prediction,labels=[-1,1]) for _,b in r.groupby('fold')])
                boot+=blocks[rng.integers(0,len(blocks),(cfg.bootstrap,len(blocks)))].sum(axis=1)
            denom=boot.sum(axis=-1);keep=(denom>0).all(axis=1)
            ba=(boot[keep].diagonal(axis1=-2,axis2=-1)/denom[keep]).mean(axis=-1)
            lo,hi=np.quantile(ba,[.025,.975])
            rows.append(dict(task=task,variant=variant,n=len(s),ba=balanced_accuracy_score(s.truth,s.prediction),
                             conditional_low=lo,conditional_high=hi,LL=int(cm[0,0]),LR=int(cm[0,1]),RL=int(cm[1,0]),RR=int(cm[1,1])))
    return pd.DataFrame(rows)


def waveform_metrics(records,waves):
    rows=[]
    for j,r in enumerate(records):
        for name,arrays in waves.items():
            for window,mask in [('0_500',(r.t>=0)&(r.t<=.5)),('0_800',r.t>=0)]:
                x=r.light[r.valid][...,mask];p=arrays[j][r.valid][...,mask]
                sse=np.sum((x-p)**2);sst=np.sum((x-x.mean(axis=(0,2),keepdims=True))**2)
                rows.append(dict(record=r.name,task=r.task,model=name,window=window,
                                 rmse=float(np.sqrt(np.mean((x-p)**2))),r2=float(1-sse/max(sst,1e-12)),sse=float(sse),sst=float(sst)))
    return pd.DataFrame(rows)
