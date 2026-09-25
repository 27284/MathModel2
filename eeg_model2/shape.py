"""零直流DoG + 正交相位Gabor位置/方向能量；不将图形指向等同于半视野。"""
import numpy as np
from scipy.ndimage import gaussian_filter, shift
from scipy.signal import fftconvolve
from scipy.special import softmax
def triangles(size=65):
    y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    right=((x>=-.55)&(x<=.55)&(np.abs(y)<=(.55-x)*.65/1.1)).astype(float)
    return np.stack([right[:,::-1],right])

ANGLES=np.arange(12)*np.pi/12


def dog(image):
    # 两个归一化高斯之差的DC增益为0；参考资料K=.75不满足亮度不变。
    return gaussian_filter(image,1.,mode='reflect')-gaussian_filter(image,2.8,mode='reflect')


def gabor_pair(theta,sigma=3.,wavelength=7.,gamma=.65):
    y,x=np.mgrid[-15:16,-15:16]
    xp=x*np.cos(theta)+y*np.sin(theta)
    yp=-x*np.sin(theta)+y*np.cos(theta)
    envelope=np.exp(-(xp*xp+gamma*gamma*yp*yp)/(2*sigma*sigma))
    kernels=[]
    for phase in [0.,np.pi/2]:
        g=envelope*np.cos(2*np.pi*xp/wavelength+phase)
        g-=g.mean()
        g/=np.linalg.norm(g)
        kernels.append(g)
    return np.array(kernels)


def energy_maps(image):
    lgn=dog(np.asarray(image,float))
    pad=np.pad(lgn,15,mode='reflect')
    maps=[]
    for angle in ANGLES:
        even,odd=gabor_pair(angle)
        a=fftconvolve(pad,even,mode='valid')
        b=fftconvolve(pad,odd,mode='valid')
        maps.append(np.hypot(a,b))
    return lgn,np.array(maps)


class GaborEncoder:
    def __init__(self):
        self.images=triangles(65)
        self.templates=np.array([energy_maps(im)[1] for im in self.images])
        self.phi=np.array([self.encode(im)[1] for im in self.images])

    def encode(self,image):
        _,maps=energy_maps(image)
        x=maps.ravel()
        z=self.templates.reshape(2,-1)
        scores=z@x/np.maximum(np.linalg.norm(z,axis=1)*np.linalg.norm(x),1e-12)
        return scores,softmax(6*scores)

    def checks(self):
        rows=[]
        left,right=self.images
        for name,im in [('left',left),('right',right),('left_shift_2px',shift(left,(0,2),order=0,mode='constant')),
                        ('left_missing_tip',left*(np.indices(left.shape)[1]>22))]:
            scores,phi=self.encode(im)
            lgn,energy=energy_maps(im)
            total=energy.sum(axis=0)
            x=np.arange(im.shape[1])-(im.shape[1]-1)/2
            # 中线能量不全部偏给某一侧，保证奇数尺寸镜像SLI严格交换符号。
            sli=float(np.sum(total*np.sign(x)[None,:])/max(total.sum(),1e-12))
            rows.append(dict(image=name,r_left=float(scores[0]),r_right=float(scores[1]),
                         phi_left=float(phi[0]),phi_right=float(phi[1]),
                         lgn_total_abs=float(np.abs(lgn).sum()),image_right_minus_left_sli=sli))
        return rows


def dipole_demo():
    """均匀无限导体的几何示意，含方向点积；与真实EEG拟合完全分开。"""
    electrodes=np.array([[0,.4,1],[-.6,0,1],[.6,0,1]])
    sources=np.array([[-.35,0,.6],[.35,0,.6]])
    orientations=np.array([[0,0,1],[0,0,1]])
    displacement=electrodes[:,None,:]-sources[None,:,:]
    return np.einsum('eck,ck->ec',displacement,orientations)/np.linalg.norm(displacement,axis=-1)**3


