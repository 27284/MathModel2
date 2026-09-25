"""保留时序幅值的24维特征，以及噪声白化的共同/差异匹配。"""
import numpy as np
from scipy.special import expit
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

FAMILIES=('amplitude4','temporal24','contrast25')
SETTINGS=tuple((family,'lr',c) for family in FAMILIES for c in (.01,.1,1.))+(('temporal24','lda',0.),)


def four_features(x,t):
    main=(t>=.25)&(t<=.5);early=(t>=.25)&(t<.4);late=(t>=.4)&(t<=.5)
    a=x[...,main].mean(axis=-1)
    return np.column_stack([a[:,0],a[:,1:].mean(axis=1),a[:,2]-a[:,1],
                            x[:,0,late].mean(axis=-1)-x[:,0,early].mean(axis=-1)])


def time_features(x,t):
    return np.stack([x[..., (t>=a)&(t<b)].mean(axis=-1)
                     for a,b in zip(np.linspace(.1,.5,9)[:-1],np.linspace(.1,.5,9)[1:])],axis=-1).reshape(len(x),24)


def build_features(records,prepared,labels,bank,model,cells):
    variants={name:[] for name in FAMILIES};transforms=[]
    for j,(r,p,y) in enumerate(zip(records,prepared,labels)):
        x=time_features(r.light,r.t)
        train=x[p.train];truth=y[p.train]
        center=train.mean(axis=0)
        scale=np.maximum(train.std(axis=0),1e-6)
        normalized=(x-center)/scale
        # 类内残差只从训练Trial估计，留出标签不参与协方差/模板。
        residual=train.copy()
        for sign in (-1,1):residual[truth==sign]-=train[truth==sign].mean(axis=0)
        covariance=LedoitWolf().fit(residual).covariance_
        covariance+=max(np.trace(covariance)/24,1e-12)*1e-6*np.eye(24)
        duration=float(np.median(r.duration[p.train]))
        predicted=model.predict(bank,j,[-1,1],[duration]*2)
        template=time_features(predicted,r.t)
        m=template.mean(axis=0);d=(template[1]-template[0])/2
        w=np.linalg.solve(covariance,d)
        norm=float(np.sqrt(max(d@w,1e-12)))
        w/=norm
        score=(x-m)@w
        variants['amplitude4'].append(four_features(r.light,r.t))
        variants['temporal24'].append(normalized)
        variants['contrast25'].append(np.column_stack([normalized,score]))
        transforms.append(dict(record=r.name,task=r.task,center=center.tolist(),scale=scale.tolist(),
                               template_common=m.tolist(),matched_weight=w.tolist(),
                               mahalanobis_contrast=norm,training_trials=np.flatnonzero(p.train).tolist()))
    return variants,transforms


def apply_transform(epochs,t,transform,family):
    if family=='amplitude4':return four_features(epochs,t)
    x=time_features(epochs,t)
    z=(x-np.asarray(transform['center']))/np.asarray(transform['scale'])
    if family=='temporal24':return z
    score=(x-np.asarray(transform['template_common']))@np.asarray(transform['matched_weight'])
    return np.column_stack([z,score])


def concatenate(records,prepared,labels,features,task,blocks=None):
    xs=[];ys=[];ws=[];indices=[]
    for j,(r,p,y,x) in enumerate(zip(records,prepared,labels,features)):
        if r.task!=task:continue
        mask=p.train if blocks is None else np.isin(r.folds,list(blocks))&r.valid
        ix=np.flatnonzero(mask)
        xs.append(x[ix]);ys.append(y[ix]);ws.append(p.weights[ix].mean(axis=1))
        indices.extend((j,int(i)) for i in ix)
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws),indices


def fit_classifier(x,y,w,setting):
    family,kind,c=setting
    scaler=StandardScaler().fit(x,sample_weight=w/w.mean())
    xx=scaler.transform(x)
    if kind=='lr':
        clf=LogisticRegression(C=c,class_weight='balanced',max_iter=2000).fit(xx,y,sample_weight=w/w.mean())
    else:clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]).fit(xx,y)
    return dict(family=family,estimator=kind,C=c,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                coefficients=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]),
                note='LDA不使用样本权重；LR使用QC及类别平衡')


def classify(model,x):
    xx=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    probability=expit(xx@np.asarray(model['coefficients'])+model['intercept'])
    return np.where(probability>=.5,1,-1),probability
