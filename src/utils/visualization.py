import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import logging
from scipy.interpolate import griddata

class OceanVisualizer:
    def __init__(self,output_dir,fig_size=(12,8),dpi=300):
        self.output_dir=Path(output_dir)
        self.fig_size=fig_size
        self.dpi=dpi
        self.logger=logging.getLogger(__name__)
        for s in['plots','attention_maps','predictions']:
            (self.output_dir/s).mkdir(parents=True,exist_ok=True)
        self.projection=ccrs.PlateCarree()
    def _get_grid(self,shape,tl,tlg):
        if tl is not None and tlg is not None:
            la1,la2=float(np.min(tl)),float(np.max(tl))
            lo1,lo2=float(np.min(tlg)),float(np.max(tlg))
        else:
            la1,la2=0,65
            lo1,lo2=-80,0
        lg=np.linspace(la1,la2,shape[0])
        ln=np.linspace(lo1,lo2,shape[1])
        return lg,ln
    def _r(self,a,sh):
        hi,wi=a.shape
        ho,wo=sh
        yi=np.linspace(0,1,hi)
        xi=np.linspace(0,1,wi)
        yo=np.linspace(0,1,ho)
        xo=np.linspace(0,1,wo)
        xi2,yi2=np.meshgrid(xi,yi)
        pts=np.vstack([xi2.ravel(),yi2.ravel()]).T
        vals=a.ravel()
        xo2,yo2=np.meshgrid(xo,yo)
        opts=np.vstack([xo2.ravel(),yo2.ravel()]).T
        r=griddata(pts,vals,opts,method='linear')
        return r.reshape(ho,wo)
    def plot_attention_maps(self,attention_maps,tlat=None,tlong=None,save_path=None):
        if not attention_maps:
            return
        k=list(attention_maps.keys())
        fig,axs=plt.subplots(1,len(k),figsize=(5*len(k),5),subplot_kw={'projection':self.projection})
        if len(k)==1:
            axs=[axs]
        sh=(120,200)
        la,ln=self._get_grid(sh,tlat,tlong)
        for ax,n in zip(axs,k):
            a=attention_maps[n]
            if len(a)==0:
                r=None
            else:
                a=a[0]
                if a.dim()==2 and a.shape[0]==a.shape[1]:
                    v=a.mean(dim=0).cpu().numpy()
                    sz=int(v.shape[0]**0.5)
                    if sz*sz==v.shape[0]:
                        v=v.reshape(sz,sz)
                        r=self._r(v,sh)
                    else:
                        r=np.zeros(sh)
                else:
                    r=np.zeros(sh)
            if r is None:
                r=np.zeros(sh)
            im=ax.pcolormesh(ln,la,r,transform=self.projection,cmap='viridis',shading='auto')
            ax.add_feature(cfeature.LAND,zorder=99,edgecolor='black',facecolor='lightgray')
            ax.coastlines()
            ax.set_extent([-80,0,0,65],crs=self.projection)
            ax.set_title(n.upper())
            plt.colorbar(im,ax=ax,orientation='horizontal',pad=0.05)
        plt.tight_layout()
        if save_path:
            sp=self.output_dir/'attention_maps'/f'{save_path}.png'
            plt.savefig(sp,dpi=self.dpi,bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    def plot_predictions(self,preds,tg,time_indices=None,save_path=None):
        if time_indices is None:
            time_indices=np.arange(len(preds))
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8))
        ax1.scatter(tg,preds,alpha=0.5,color='blue')
        l1=min(np.min(tg),np.min(preds))
        l2=max(np.max(tg),np.max(preds))
        ax1.plot([l1,l2],[l1,l2],'r--')
        ax1.set_xlabel("True Heat Transport")
        ax1.set_ylabel("Pred Heat Transport")
        ax1.grid(True)
        ax2.plot(time_indices,tg,'b-',label='True')
        ax2.plot(time_indices,preds,'r--',label='Predicted')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        if save_path:
            sp=self.output_dir/'predictions'/f'{save_path}.png'
            plt.savefig(sp,dpi=self.dpi,bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    def plot_error_histogram(self,preds,tg,save_path=None):
        e=preds-tg
        fig,ax=plt.subplots(figsize=(8,6))
        sns.histplot(e,kde=True,color='orange',ax=ax)
        ax.axvline(x=0,color='r',linestyle='--')
        ax.grid(True)
        if save_path:
            sp=self.output_dir/'plots'/f'{save_path}.png'
            plt.savefig(sp,dpi=self.dpi,bbox_inches='tight')
            plt.close()
        else:
            plt.show()
