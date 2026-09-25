"""受约束Wilson–Cowan动力学；固定源尺度；左右共享所有神经参数。"""
import numpy as np
from scipy.special import expit
from scipy.optimize import root
from scipy.signal import butter, sosfiltfilt
from numba import njit
from .data import recenter

PRIOR = np.array([.060,.120,.070,1.])
# 固定连接及Sigmoid尺度属于建模约定，不声称是文献唯一给定的突触参数。
WEE, WEI, WIE, WII, BE, BI = 5.,6.,5.,1.,-2.,-2.


def equilibrium():
    def f(z):
        e,i=z
        return [-e+expit(WEE*e-WEI*i+BE),-i+expit(WIE*e-WII*i+BI)]
    result=root(f,[.1,.2])
    if not result.success or np.max(np.abs(f(result.x)))>1e-9:
        raise RuntimeError('静息平衡点求解失败')
    return result.x


def stability(theta):
    e,i=equilibrium()
    te,ti=theta[:2]
    jac=np.array([[(-1+WEE*e*(1-e))/te,-WEI*e*(1-e)/te],
                  [WIE*i*(1-i)/ti,(-1-WII*i*(1-i))/ti]])
    ev=np.linalg.eigvals(jac)
    return dict(equilibrium=[float(e),float(i)],
                eigenvalue_real=ev.real.tolist(), eigenvalue_imag=ev.imag.tolist(),
                stable=bool(np.max(ev.real)<0))


@njit(cache=True)
def _simulate(theta, phi, durations, tasks, pt, eq, rho, tau_s):
    n=len(durations)
    states=np.empty((n,2,3,len(pt)))
    dt=(pt[1]-pt[0])/4
    for j in range(n):
        for k in range(2):
            e,i=eq[0],eq[1]
            q=0.
            g=1. if tasks[j]==1 else theta[3]
            for a in range(len(pt)):
                states[j,k,0,a]=e
                states[j,k,1,a]=i
                states[j,k,2,a]=q
                for sub in range(4):
                    time=pt[a]+(sub+.5)*dt
                    # 固定一个采样间隔的线性上升/下降，使延迟可连续优化。
                    width=pt[1]-pt[0]
                    pulse=0.
                    if durations[j]>0:
                        pulse=max(0.,min(1.,(time-theta[2])/width+.5))*max(0.,min(1.,(theta[2]+durations[j]-time)/width+.5))
                    u=g*phi[j,k]*pulse
                    se=1/(1+np.exp(-(WEE*e-WEI*i+u+BE)))
                    si=1/(1+np.exp(-(WIE*e-WII*i+BI)))
                    de=(-e+se)/theta[0]
                    di=(-i+si)/theta[1]
                    # midpoint RK2，时间步长小于最短时间常数的1/20。
                    em=e+.5*dt*de
                    im=i+.5*dt*di
                    e+=dt*(-em+1/(1+np.exp(-(WEE*em-WEI*im+u+BE))))/theta[0]
                    i+=dt*(-im+1/(1+np.exp(-(WIE*em-WII*im+BI))))/theta[1]
                    target=em-eq[0]-rho*(im-eq[1])
                    q=target+(q-target)*np.exp(-dt/tau_s)
    return states


class Simulator:
    def __init__(self, record, phi, cfg):
        self.t,self.pt,self.core=record.t,record.padded_t,record.core
        self.phi=phi
        self.cfg=cfg
        self.eq=equilibrium()
        self.sos=butter(cfg.filter_order,cfg.lowpass,fs=cfg.fs,output='sos')

    def sources(self, theta, labels, durations, tasks, common=False, states=False):
        phi=np.full((len(labels),2),.5) if common else self.phi[(np.asarray(labels)>0).astype(int)]
        s=_simulate(np.asarray(theta,float),phi,np.asarray(durations,float),np.asarray(tasks,int),
                    self.pt,self.eq,self.cfg.source_rho,self.cfg.synaptic_tau)
        if states:
            return s
        q=sosfiltfilt(self.sos,s[:,:,2,:],axis=-1)[...,self.core]
        q=recenter(q,self.t)
        # 共同驱动的两个源完全相同，只保留一个，避免不可辨识的两列G。
        return q[:,:1] if common else q



