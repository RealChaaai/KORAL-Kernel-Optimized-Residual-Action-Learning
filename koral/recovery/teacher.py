"""Recovery teacher; private controller memory is never in the policy observation."""
import numpy as np
class RecoveryTeacher:
    def __init__(self):self.mode='approach';self.ticks=0
    def reset_from(self,env):
        self.ticks=0
        contacting=np.min(env.contact)>.15
        near_goal=np.linalg.norm(env.obj[:2]-env.goal[:2])<.025
        on_table=env.obj[2]<env.table+env.cfg.object_radius+.015
        if near_goal and on_table and not contacting:
            self.mode='withdraw'
        elif contacting and near_goal:
            self.mode='lower'
        elif contacting and env.obj[2]>env.table+env.cfg.object_radius+.06:
            self.mode='transfer'
        elif contacting:
            self.mode='lift'
        else:
            self.mode='approach'
    def action(self,env):
        obj,ee,goal=env.obj,env.ee,env.goal
        grasp=np.linalg.norm(ee-obj);xy=np.linalg.norm(ee[:2]-obj[:2]);contact=np.min(env.contact)>.12
        h=env.table+env.cfg.object_radius
        self.ticks+=1
        if self.mode in ('lift','transfer','lower') and not contact and grasp>.045:
            self.mode='approach';self.ticks=0
        if self.mode=='approach':
            target=obj+np.array([0.,0.,.055]);grip=1.
            if np.linalg.norm(ee-target)<.012:self.mode='descend';self.ticks=0
        elif self.mode=='descend':
            target=obj.copy();grip=1.
            if grasp<.009:self.mode='close';self.ticks=0
        elif self.mode=='close':
            target=obj.copy();grip=-1.
            if contact and env.opening<.013:self.mode='lift';self.ticks=0
            elif self.ticks>26 and not contact:self.mode='approach';self.ticks=0
        elif self.mode=='lift':
            target=np.r_[ee[:2],h+.075];grip=-1.
            if ee[2]>h+.065:self.mode='transfer';self.ticks=0
        elif self.mode=='transfer':
            target=np.r_[goal[:2],h+.075];grip=-1.
            if np.linalg.norm(ee-target)<.013:self.mode='lower';self.ticks=0
        elif self.mode=='lower':
            target=goal+np.array([0.,0.,.003]);grip=-1.
            if np.linalg.norm(ee-target)<.009:self.mode='release';self.ticks=0
        elif self.mode=='release':
            target=ee.copy();grip=1.
            if env.opening>.035:self.mode='withdraw';self.ticks=0
        else:
            target=goal+np.array([0.,0.,.075]);grip=1.
        return np.r_[np.clip(.7*(target-ee)/env.cfg.action_scale,-1,1),grip].astype(np.float32)
