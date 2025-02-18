import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

class PatchEmbed(nn.Module):
    def __init__(self,in_channels,embed_dim,patch_size):
        super().__init__()
        self.proj=nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x,mask=None):
        x=self.proj(x)
        b,c,h,w=x.shape
        x=x.flatten(2).transpose(1,2)
        return x,None

class MultiModalFusion(nn.Module):
    def __init__(self,d_model,dropout=0.3):
        super().__init__()
        self.fu=nn.Sequential(nn.Linear(2*d_model,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
    def forward(self,a,b):
        return self.fu(torch.cat([a,b],dim=-1))

class SpatialPositionalEncoding2D(nn.Module):
    def __init__(self,d_model,dropout=0.3):
        super().__init__()
        self.d_model=d_model
        self.dr=nn.Dropout(dropout)
    def forward(self,x,h,w):
        b,l,e=x.shape
        if e!=self.d_model:
            raise RuntimeError()
        if l!=h*w:
            raise RuntimeError()
        p=self._make_pe(h,w).to(x.device)
        x=x+p[:,:l]
        return self.dr(x)
    def _make_pe(self,hh,ww):
        d=self.d_model
        out=torch.zeros(1,hh*ww,d)
        r=torch.arange(hh).float().unsqueeze(1)
        c=torch.arange(ww).float().unsqueeze(1)
        dh=d//2
        dr=torch.exp(torch.arange(0,dh,2).float()*(-math.log(10000.0)/dh))
        dc=torch.exp(torch.arange(0,dh,2).float()*(-math.log(10000.0)/dh))
        rr=torch.zeros(hh,dh)
        cc=torch.zeros(ww,dh)
        for i in range(0,dh,2):
            ix=i//2
            rr[:,i]=torch.sin(r.squeeze(1)*dr[ix])
            rr[:,i+1]=torch.cos(r.squeeze(1)*dr[ix])
            cc[:,i]=torch.sin(c.squeeze(1)*dc[ix])
            cc[:,i+1]=torch.cos(c.squeeze(1)*dc[ix])
        tmp=torch.zeros(hh,ww,d)
        for i in range(hh):
            for j in range(ww):
                tmp[i,j,:dh]=rr[i]
                tmp[i,j,dh:]=cc[j]
        return tmp.view(1,hh*ww,d)

class OceanTransformer(nn.Module):
    def __init__(self,spatial_size,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,dropout=0.1,patch_size=16):
        super().__init__()
        self.logger=logging.getLogger(__name__)
        self.d_model=d_model
        self.nhead=nhead
        self.patch_size=patch_size
        self.ssh_down=nn.Sequential(
            nn.Conv2d(1,d_model//4,3,2,1),nn.BatchNorm2d(d_model//4),nn.GELU(),
            nn.Conv2d(d_model//4,d_model//2,3,2,1),nn.BatchNorm2d(d_model//2),nn.GELU()
        )
        self.sst_down=nn.Sequential(
            nn.Conv2d(1,d_model//4,3,2,1),nn.BatchNorm2d(d_model//4),nn.GELU(),
            nn.Conv2d(d_model//4,d_model//2,3,2,1),nn.BatchNorm2d(d_model//2),nn.GELU()
        )
        self.ssh_patch=PatchEmbed(d_model//2,d_model,patch_size)
        self.sst_patch=PatchEmbed(d_model//2,d_model,patch_size)
        self.fusion=MultiModalFusion(d_model,dropout)
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_model))
        enc=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation=F.gelu,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(enc,num_layers)
        self.pos2d=SpatialPositionalEncoding2D(d_model,dropout)
        self.pre_fc=nn.Sequential(nn.Linear(d_model,d_model//2),nn.GELU(),nn.LayerNorm(d_model//2),nn.Dropout(dropout/2))
        self.out_fc=nn.Linear(d_model//2,1)
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,(nn.BatchNorm2d,nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    def forward(self,ssh,sst,attention_mask=None):
        b=ssh.size(0)
        se=self.ssh_down(ssh)
        te=self.sst_down(sst)
        p1,_=self.ssh_patch(se,None)
        p2,_=self.sst_patch(te,None)
        f=self.fusion(p1,p2)
        _,_,hh,ww=se.shape
        hs=hh//self.patch_size
        ws=ww//self.patch_size
        c=torch.cat([self.cls_token.expand(b,1,self.d_model),f],dim=1)
        xp=c[:,1:]
        xp=self.pos2d(xp,hs,ws)
        x=torch.cat([c[:,0:1],xp],dim=1)
        x=self.transformer(x)
        attn=torch.matmul(x,x.transpose(-2,-1))/math.sqrt(self.d_model)
        attn=F.softmax(attn,dim=-1)
        pooled=x[:,0]
        h=self.pre_fc(pooled)
        o=self.out_fc(h).squeeze(-1)
        m={}
        with torch.no_grad():
            cls_row=attn[:,0:1,1:].mean(1)
            m['cls']=cls_row
        return o,m
