"""Trajectory-bounded action chunks and policy-history-aligned recovery segments."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .environment import History, make_env
from .teacher import RecoveryTeacher

@dataclass
class Segment:
    histories: np.ndarray
    actions: np.ndarray
    next_histories: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_id: int
    split: str
    source: str='teacher'
    def validate(self):
        n=len(self.actions)
        if not n:raise ValueError('empty segment')
        if self.histories.shape!=self.next_histories.shape or len(self.histories)!=n:
            raise ValueError('inconsistent transitions')
        if self.actions.shape!=(n,4):raise ValueError('action shape')
        if not np.isfinite(self.actions).all() or not np.isfinite(self.histories).all():
            raise ValueError('nonfinite transition')
        if np.max(np.abs(self.actions))>1.00001:raise ValueError('unbounded action')
        if np.any(self.terminated[:-1]) or np.any(self.truncated[:-1]):
            raise ValueError('segment crosses episode boundary')
        if self.split not in ('train','acquire'):raise ValueError('held-out data cannot enter training')
        return self

def collect_segment(env, history, length, budget=None,candidate_id=0):
    if env.split not in ('train','acquire'):raise ValueError('cannot query held-out environment')
    if budget is not None:budget.begin(candidate_id,length)
    teacher=RecoveryTeacher();teacher.reset_from(env)
    hist=history.copy();hs=[];acts=[];nh=[];term=[];trunc=[]
    for _ in range(length):
        if env.done:break
        # Charge even if the expert subsequently fails. No oracle pre-screening.
        if budget is not None:budget.charge()
        action=teacher.action(env);hs.append(hist.array());acts.append(action.copy())
        obs,_,te,tr,_=env.step(action);hist.append(obs);nh.append(hist.array());term.append(te);trunc.append(tr)
        if budget is not None:budget.environment_steps+=1
    if not acts:raise ValueError('candidate already terminal')
    return Segment(np.stack(hs),np.stack(acts),np.stack(nh),np.array(term),np.array(trunc),
                   env.seed,env.split).validate()

def collect_demos(count=30,seed=1000,backend='contact_proxy',config=None):
    if count<2:raise ValueError('at least two demonstrations required')
    records=[];segments=[]
    for i in range(count):
        env=make_env(backend,config);obs=env.reset(seed+i,'train');hist=History(obs)
        seg=collect_segment(env,hist,env.cfg.horizon)
        records.append(env.summary()|{'expert_actions':len(seg.actions)})
        # Keep complete unsuccessful trajectories too; no free success rejection.
        segments.append(seg)
    return segments,records

def examples(segments,horizon=8):
    xs=[];ys=[];ms=[];ps=[]
    for seg in segments:
        seg.validate()
        for t in range(len(seg.actions)):
            length=min(horizon,len(seg.actions)-t)
            y=np.zeros((horizon,4),np.float32);mask=np.zeros(horizon,np.float32)
            y[:length]=seg.actions[t:t+length];mask[:length]=1
            xs.append(seg.histories[t]);ys.append(y);ms.append(mask)
            # Observed object-to-goal displacement over the valid segment only.
            ps.append(seg.next_histories[t+length-1,-1,3:6]-seg.histories[t,-1,3:6])
    if not xs:raise ValueError('empty training set')
    return {k:np.asarray(v,np.float32) for k,v in [('x',xs),('y',ys),('mask',ms),('progress',ps)]}

def save_segments(path,segments):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={};meta=[]
    for i,s in enumerate(segments):
        for key in ('histories','actions','next_histories','terminated','truncated'):
            arrays[f'{i}_{key}']=getattr(s,key)
        meta.append({'episode_id':s.episode_id,'split':s.split,'source':s.source})
    arrays['metadata']=np.array(json.dumps(meta));np.savez_compressed(path,**arrays)

def load_segments(path):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));out=[]
        for i,m in enumerate(meta):
            kw={k:z[f'{i}_{k}'].copy() for k in ('histories','actions','next_histories','terminated','truncated')}
            out.append(Segment(**kw,**m).validate())
    return out
