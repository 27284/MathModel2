"""共享神经源的共同/差异观测；神经参数固定，读出系数受收缩约束。"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import sosfiltfilt,lfilter
from .neural import Simulator,PRIOR,stability
from .data import recenter
from .preprocess import huber_erp

# 在读取本轮留出成绩前固定；从强正则到弱正则排序，平手优先强约束。
STRUCTURES=('transient_contrast','legacy')
SETTINGS=tuple((ridge,pool,structure) for structure in STRUCTURES
               for ridge in (.1,.01,.001,.0001) for pool in (1.,.1,0.))
FEATURE_SETTING=(.01,.1,'legacy')


class SourceBank:
    def __init__(self,record,phi,cfg):
        self.sim=Simulator(record,phi,cfg)
        self.t,self.pt,self.core=record.t,record.padded_t,record.core
        self.cfg=cfg
        self.phi=np.asarray(phi)
        self.theta=PRIOR.copy();self.theta[3]=1.
        self.cache={}
        mask=self.t>=0
        self.scales={name:np.sqrt(np.mean(self._build(52/cfg.fs,name)[...,mask]**2,axis=-1))
                     for name in STRUCTURES}
        if any(np.any(scale<1e-10) for scale in self.scales.values()):
            raise ValueError('形状驱动不能形成非零共同/差异源')

    def _build(self,duration,structure='legacy'):
        # 未处理神经源先进入因果记忆状态，再施加与观测相同的滤波和基线。
        q=self.sim.sources(self.theta,[-1],[duration],[1],states=True)[0,:,2]
        common=q.mean(axis=0)
        contrast=(q[0]-q[1])/2  # 左图偏好源减右图偏好源；类别符号在读出处作用。
        parts=[]
        for component,source in enumerate((common,contrast)):
            group=[source]
            for tau in (.08,.25,.75):
                a=np.exp(-1/(self.cfg.fs*tau))
                group.append(lfilter([1-a],[1,-a],source))
            # 提示后持续状态可延续至本分析窗末尾，不强制800ms回零。
            if structure=='transient_contrast' and component==1:
                # 少量预设潜伏期变化；差异项不再直接包含持续阶跃。
                for lag in (-.04,.04):
                    group.append(np.interp(self.pt-lag,self.pt,source,left=0.,right=source[-1]))
            else:
                step=(self.pt>=self.theta[2]).astype(float)
                for tau in (.3,.8):
                    a=np.exp(-1/(self.cfg.fs*tau))
                    group.append(lfilter([1-a],[1,-a],step))
            filtered=sosfiltfilt(self.sim.sos,np.array(group),axis=-1)[...,self.core]
            parts.append(recenter(filtered,self.t))
        return np.array(parts)

    def basis(self,duration,structure='legacy'):
        if structure not in STRUCTURES:raise ValueError('未知响应结构')
        key=(round(float(duration),10),structure)
        if key not in self.cache:
            self.cache[key]=self._build(duration,structure)/self.scales[structure][...,None]
        return self.cache[key]

    def audit(self):
        return dict(theta=self.theta.tolist(),stability=stability(self.theta),
                    memory_tau_s=[.08,.25,.75],persistent_tau_s=[.3,.8],
                    structures=list(STRUCTURES),contrast_lags_s=[-.04,.04],
                    note='固定稳定动力学；受限差异候选移除直接持续阶跃，不能保证消除伪影；无空间头模型')


def targets(records,prepared,labels,cfg,blocks=None):
    curves=[];sizes=[];durations=[]
    for r,p,y in zip(records,prepared,labels):
        selection=p.train if blocks is None else np.isin(r.folds,list(blocks))&r.valid
        erps=[];ns=[]
        for sign in (-1,1):
            ids=selection&(y==sign)
            if ids.sum()<2:raise ValueError('记录/条件中有效Trial不足两个')
            erps.append(huber_erp(r.light[ids],p.weights[ids],r.t,cfg.huber_c))
            w=p.weights[ids].mean(axis=1)
            ns.append(w.sum()**2/(w@w))
        curves.append(erps);sizes.append(ns)
        durations.append(float(np.median(r.duration[selection])))
    return np.array(curves),np.array(sizes),np.array(durations)


@dataclass
class ResponseFit:
    setting: tuple
    coef: np.ndarray  # record, common/contrast, source, channel
    effective_df: float
    durations: np.ndarray

    def predict(self,bank,record_index,labels,durations,common_only=False):
        result=[]
        for sign,dur in zip(labels,durations):
            b=bank.basis(dur,self.setting[2])
            m=self.coef[record_index,0].T@b[0]
            d=self.coef[record_index,1].T@b[1]
            result.append(m if common_only else m+sign*d)
        return np.array(result)

    def to_dict(self):
        return dict(ridge=self.setting[0],pool=self.setting[1],structure=self.setting[2],coefficients=self.coef.tolist(),
                    effective_linear_df=self.effective_df,nominal_coefficients=int(self.coef.size),
                    training_duration=self.durations.tolist())


def fit_response(records,bank,cells,setting):
    curves,neff,durations=cells
    m=curves.mean(axis=1);d=(curves[:,1]-curves[:,0])/2
    n=len(records);p=6
    # 只在同任务记录间收缩；标签两条件始终共享同一套系数。
    lap=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if records[i].task==records[j].task:
                lap[i,i]+=.5;lap[j,j]+=.5;lap[i,j]-=.5;lap[j,i]-=.5
    mask=bank.t>=0
    coef=[];degrees=0.
    for component,target in enumerate((m,d)):
        matrix=np.zeros((n*p,n*p));rhs=np.zeros((n*p,3))
        weight=1/(1/neff[:,0]+1/neff[:,1]);weight/=weight.mean()
        for r in range(n):
            b=bank.basis(durations[r],setting[2])[component][:,mask].T
            if b.shape[1]!=p:raise ValueError('时间基底形状异常')
            a=slice(r*p,(r+1)*p)
            matrix[a,a]=weight[r]*(b.T@b)/len(b)
            rhs[a]=weight[r]*b.T@target[r][:,mask].T/len(b)
        ridge,pool,_=setting
        penalty=(ridge*(10 if component else 1))*np.eye(n*p)
        penalty+=pool*(2 if component else 1)*np.kron(lap,np.eye(p))
        solved=np.linalg.solve(matrix+penalty,rhs)
        degrees+=3*float(np.trace(np.linalg.solve(matrix+penalty,matrix)))
        coef.append(solved.reshape(n,p,3))
    return ResponseFit(tuple(setting),np.stack(coef,axis=1),degrees,durations)


def validation_loss(records,bank,model,train_cells,valid_cells):
    train,ns,_=train_cells;observed,_,durations=valid_cells
    mask=bank.t>=0
    losses=[]
    for j in range(len(records)):
        predicted=model.predict(bank,j,[-1,1],[durations[j]]*2)
        common_error=predicted.mean(axis=0)-observed[j].mean(axis=0)
        contrast_error=(predicted[1]-predicted[0]-observed[j,1]+observed[j,0])/2
        # 每通道尺度仅来自训练ERP，避免大幅值记录完全支配模型选择。
        scale=np.maximum(np.mean(train[j][...,mask]**2,axis=(0,2)),1.)
        # 明确给差波一个固定权重，而非按外层成绩事后调整。
        losses.append(float(np.mean((common_error[:,mask]**2+contrast_error[:,mask]**2)/scale[:,None])))
    return float(np.mean(losses))
