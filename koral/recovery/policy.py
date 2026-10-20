"""History-conditioned action chunks with masked sequence and progress losses."""
import copy
import numpy as np
import torch
from torch import nn
from .environment import OBS_DIM,HISTORY
from .data import examples

class ChunkMember(nn.Module):
    def __init__(self,width=96,horizon=8):
        super().__init__();self.horizon=horizon
        self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(HISTORY*OBS_DIM,width),nn.SiLU(),
             nn.Linear(width,width),nn.SiLU())
        self.actions=nn.Linear(width,horizon*4);self.progress=nn.Linear(width,3)
    def forward(self,x):
        z=self.encoder(x)
        return torch.tanh(self.actions(z).reshape(-1,self.horizon,4)),self.progress(z)

class ChunkPolicy(nn.Module):
    def __init__(self,members=3,width=96,horizon=8):
        super().__init__();self.spec={'members':members,'width':width,'horizon':horizon}
        self.horizon=horizon
        self.models=nn.ModuleList([ChunkMember(width,horizon) for _ in range(members)])
        self.register_buffer('mean',torch.zeros(OBS_DIM));self.register_buffer('std',torch.ones(OBS_DIM))
    def forward(self,x):
        ys=[m((x-self.mean)/self.std) for m in self.models]
        return torch.stack([y[0] for y in ys]),torch.stack([y[1] for y in ys])
    @torch.no_grad()
    def chunk(self,history):
        self.eval();x=torch.as_tensor(history,dtype=torch.float32).unsqueeze(0)
        a,_=self(x);return a.mean(0)[0].cpu().numpy()
    @torch.no_grad()
    def disagreement(self,history):
        a,_=self(torch.as_tensor(history,dtype=torch.float32).unsqueeze(0))
        return float(a[:,0,:min(3,self.horizon)].var(dim=0,unbiased=False).mean())
    def save(self,path):
        torch.save({'spec':self.spec,'state_dict':self.state_dict(),'format':2},path)
    @classmethod
    def load(cls,path):
        s=torch.load(path,map_location='cpu',weights_only=True)
        if s.get('format')!=2:raise ValueError('incompatible checkpoint format')
        m=cls(**s['spec']);m.load_state_dict(s['state_dict']);m.eval();return m

def sequence_loss(pred,target,mask,progress_pred,progress_target,sequence=True):
    mask=mask.unsqueeze(-1);den=(mask.sum()*4).clamp_min(1)
    act=((pred-target).square()*mask).sum()/den
    loss=act
    if sequence and pred.shape[1]>1:
        valid=mask[:,1:]*mask[:,:-1]
        d=(pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])
        loss=loss+.08*(d.square()*valid).sum()/(valid.sum()*4).clamp_min(1)
    return loss+.03*(progress_pred-progress_target).square().mean()

def initialize(segments,seed=0,steps=600,width=96,horizon=8):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=ChunkPolicy(width=width,horizon=horizon)
    arr=examples(segments,horizon);x=torch.from_numpy(arr['x'])
    model.mean.copy_(x.reshape(-1,OBS_DIM).mean(0))
    model.std.copy_(x.reshape(-1,OBS_DIM).std(0).clamp_min(.025))
    return update(model,segments,[],steps=steps,seed=seed)[0]

def update(model,rehearsal,recovery,steps=80,seed=0,sequence=True):
    """Both branches use identical rehearsal draws and equal optimizer counts.

    Recovery adds an equally-sized batch, so FLOPs need not be equal. Counts and
    wall time are separately logged by the experiment; never called free compute.
    """
    model=copy.deepcopy(model);model.train();rng=np.random.default_rng(seed)
    base=examples(rehearsal,model.horizon);new=examples(recovery,model.horizon) if recovery else None
    def tensorize(a):return {k:torch.from_numpy(v) for k,v in a.items()}
    base=tensorize(base);new=tensorize(new) if new is not None else None
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5)
    history=[]
    # A separate generator ensures adding recovery does not alter rehearsal draws.
    rg_new=np.random.default_rng(seed+1987)
    for step in range(steps):
        ids=rng.integers(0,len(base['x']),size=min(128,len(base['x'])))
        a,p=model(base['x'][ids]);loss=0
        for j in range(len(model.models)):
            loss=loss+sequence_loss(a[j],base['y'][ids],base['mask'][ids],p[j],base['progress'][ids],sequence)
        loss=loss/len(model.models)
        if new is not None:
            ri=rg_new.integers(0,len(new['x']),size=min(128,len(new['x'])))
            aa,pp=model(new['x'][ri]);extra=0
            for j in range(len(model.models)):
                extra+=sequence_loss(aa[j],new['y'][ri],new['mask'][ri],pp[j],new['progress'][ri],sequence)
            loss=.35*loss+.65*extra/len(model.models)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if step in (0,steps-1):history.append(float(loss.detach()))
    model.eval();return model,history
