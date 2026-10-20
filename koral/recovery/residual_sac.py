"""Optional residual SAC for fine-tuning the learned recovery policy.

This module is separate from the acquisition study. Its gains are never imputed
to the segment-learning results. Executed actions are stored in the critic.
"""
import copy,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .environment import HISTORY,OBS_DIM,History,make_env,TaskConfig
from .experiment import write_json

DIM=HISTORY*OBS_DIM
class Actor(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIM,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,8))
    def sample(self,x,base,scale=.3,deterministic=False):
        mean,logstd=self.net(x).chunk(2,-1);logstd=logstd.clamp(-5,1)
        z=mean if deterministic else mean+torch.exp(logstd)*torch.randn_like(mean)
        latent=torch.atanh(base.clamp(-.98,.98))+scale*z;action=torch.tanh(latent)
        logn=-.5*((z-mean)/torch.exp(logstd)).square()-logstd-.5*math.log(2*math.pi)
        # Stable Jacobian: log(1-tanh(x)^2) = 2(log2-x-softplus(-2x)).
        logjac=2*(math.log(2)-latent-torch.nn.functional.softplus(-2*latent))
        logp=(logn-math.log(scale)-logjac).sum(-1,keepdim=True)
        return action,logp
class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIM+4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,x,a):return self.net(torch.cat([x,a],-1))
class ResidualAgent:
    def __init__(self,prior,seed=0):
        torch.manual_seed(seed);self.prior=copy.deepcopy(prior).eval()
        for p in self.prior.parameters():p.requires_grad=False
        self.actor=Actor();self.q=nn.ModuleList([Critic(),Critic()]);self.target=copy.deepcopy(self.q)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=3e-4);self.qo=torch.optim.Adam(self.q.parameters(),lr=3e-4)
    def inputs(self,h):
        x=(h-self.prior.mean)/self.prior.std
        with torch.no_grad():b,_=self.prior(h);base=b.mean(0)[:,0]
        return x.flatten(1),base
    @torch.no_grad()
    def action(self,history,deterministic=False):
        x,b=self.inputs(torch.as_tensor(history,dtype=torch.float32).unsqueeze(0))
        return self.actor.sample(x,b,deterministic=deterministic)[0][0].numpy()
    def learn(self,batch):
        h,a,r,nh,terminated=[torch.as_tensor(v,dtype=torch.float32) for v in batch]
        x,b=self.inputs(h);nx,nb=self.inputs(nh)
        with torch.no_grad():
            na,lp=self.actor.sample(nx,nb);qnext=torch.minimum(*[q(nx,na) for q in self.target])-.05*lp
            y=r+.98*(1-terminated)*qnext
        qloss=sum((q(x,a)-y).square().mean() for q in self.q)
        self.qo.zero_grad();qloss.backward();nn.utils.clip_grad_norm_(self.q.parameters(),10.);self.qo.step()
        for p in self.q.parameters():p.requires_grad=False
        pa,lp=self.actor.sample(x,b);loss=(.05*lp-torch.minimum(*[q(x,pa) for q in self.q])).mean()
        loss+=.03*(pa-b).square().mean()
        self.ao.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),10.);self.ao.step()
        for p in self.q.parameters():p.requires_grad=True
        with torch.no_grad():
            for t,q in zip(self.target.parameters(),self.q.parameters()):t.lerp_(q,.01)
        return {'actor_loss':float(loss.detach()),'critic_loss':float(qloss.detach())}

def train(prior,out,backend='contact_proxy',steps=5000,seed=0,config=None):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    agent=ResidualAgent(prior,seed);rng=np.random.default_rng(seed);buffer=[];logs=[];gradient_updates=0
    env=make_env(backend,config);episode=0;obs=env.reset(90000+seed*1000,'acquire');hist=History(obs)
    for i in range(steps):
        h=hist.array();a=agent.action(h);old_score=env.progress()
        nxt,_,te,tr,info=env.step(a);hist.append(nxt)
        r=env.progress()-old_score-.002+float(info['success'])
        # Normalized action is clipped only to [-1,1], already enforced by tanh.
        buffer.append((h,a,float(r),hist.array(),float(te)))
        if len(buffer)>100000:buffer.pop(0)
        if len(buffer)>=128:
            idx=rng.integers(0,len(buffer),size=128);cols=list(zip(*(buffer[j] for j in idx)))
            batch=[np.stack(c) for c in cols];batch[2]=batch[2][:,None];batch[4]=batch[4][:,None]
            metric=agent.learn(batch);gradient_updates+=1
            if i%100==0 or i==steps-1:logs.append(metric|{'environment_step':i+1})
        if te or tr:
            episode+=1;obs=env.reset(90000+seed*1000+episode,'acquire');hist=History(obs)
    torch.save({'actor':agent.actor.state_dict(),'critics':agent.q.state_dict(),'prior_spec':prior.spec,
                'prior':prior.state_dict(),'seed':seed,'steps':steps,'backend':backend},out/'residual_sac.pt')
    write_json(out/'training.json',{'updates':logs,'environment_steps':steps,'gradient_updates':gradient_updates,'teacher_actions':0,
                'backend':backend,'performance_evaluated':False})
    return agent
