"""Task interface, read-only policy observations and exact replay snapshots."""
from dataclasses import dataclass, asdict
import copy
import numpy as np
from ..robot import HOME, LIMITS, forward, chain, JointServo
from .contact import integrate_ball

ACTION_DIM=4
OBS_DIM=32
HISTORY=4

@dataclass(frozen=True)
class TaskConfig:
    horizon: int=160
    control_dt: float=0.04
    action_scale: float=0.012
    physics_substeps: int=80
    object_radius: float=0.017
    history: int=HISTORY
    def __post_init__(self):
        if self.horizon<1 or self.control_dt<=0 or self.physics_substeps<1:
            raise ValueError('positive horizon and time discretization required')
        if self.action_scale<=0 or self.object_radius<=0 or self.history!=HISTORY:
            raise ValueError('positive geometry/action scale and supported history required')

class ContactEnv:
    """7R kinematic arm + inertial sphere / Coulomb contact (explicit proxy).

    The policy receives positions, velocities, gripper measurements and joint
    state. It NEVER receives task phases, teacher goals or disturbance parameters.
    """
    backend='contact_proxy'
    def __init__(self, config=None):
        self.cfg=config or TaskConfig();self.servo=JointServo(damping=.035,max_step=.16)
        self.q=HOME.copy();self.ee=forward(self.q);self.done=False
    def reset(self,seed=0,split='train',disturbance='none'):
        if split not in ('train','acquire','development','test','stress'):
            raise ValueError('Unknown split')
        if disturbance not in ('none','impulse','low_friction'):
            raise ValueError('Unknown disturbance')
        self.rng=np.random.default_rng(int(seed));self.seed=int(seed);self.split=split
        self.disturbance=disturbance;self.q=HOME.copy();self.ee=forward(self.q)
        self.ee_vel=np.zeros(3);self.qvel=np.zeros(7);self.opening=.038
        self.contact=np.zeros(2);self.peak_force=0.;self.steps=0;self.time=0.
        self.done=False;self.success=False;self.lifted=False;self.stable=0
        spread=.02 if split=='train' else .035
        self.table=float(self.ee[2]-.12)
        self.obj=np.r_[self.ee[:2]+self.rng.uniform(-spread,spread,2),self.table+self.cfg.object_radius]
        self.initial_obj=self.obj.copy()
        self.goal=np.r_[self.ee[:2]+np.array([.075,-.045])+self.rng.uniform(-.018,.018,2),self.table+self.cfg.object_radius]
        self.vel=np.zeros(3);self.omega=np.zeros(3);self.last_action=np.array([0.,0.,0.,1.])
        self.mass=float(self.rng.uniform(.045,.075));self.mu=float(self.rng.uniform(.75,1.1))
        if disturbance=='low_friction': self.mu*=.45
        self.impulse_step=int(self.rng.integers(65,90));self.impulse=self.rng.uniform(-.002,.002,3);self.impulse[2]=0
        self.trace=[]
        self._record()
        return self.observe()
    def pads(self,ee=None,opening=None):
        ee=self.ee if ee is None else ee;w=self.opening if opening is None else opening
        # Proxy gripper remains aligned with the table; the MuJoCo backend holds
        # orientation with a 6D joint servo instead of this boundary condition.
        return np.array([ee+np.array([0.,w+.009,0.]),ee-np.array([0.,w+.009,0.])])
    def observe(self):
        # Absolute object poses are simulated state sensing, not images.
        o=np.r_[self.obj-self.ee,self.goal-self.obj,self.ee[2]-self.table,
                self.vel,self.ee_vel,self.q/np.pi,self.qvel*.05,
                self.opening,self.contact/15.,self.last_action[3],self.steps/self.cfg.horizon]
        assert o.shape==(OBS_DIM,),o.shape
        return o.astype(np.float32)
    def _record(self):
        self.trace.append({'q':self.q.copy(),'ee':self.ee.copy(),'object':self.obj.copy(),
            'velocity':self.vel.copy(),'opening':self.opening,'contacts':self.contact.copy(),
            'time':self.time})
    def step(self,action):
        if self.done: raise RuntimeError('reset required after episode end')
        action=np.asarray(action,float)
        if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('finite four-dimensional action required')
        a=np.clip(action,-1,1);old_ee=self.ee.copy();old_q=self.q.copy();pads0=self.pads()
        self.q,_=self.servo.resolved_step(self.q,a[:3]*self.cfg.action_scale)
        self.ee=forward(self.q);self.qvel=(self.q-old_q)/self.cfg.control_dt
        self.ee_vel=(self.ee-old_ee)/self.cfg.control_dt
        target=.012+(a[3]+1)*.5*(.038-.012)
        self.opening+=float(np.clip(target-self.opening,-.003,.003))
        impulse=self.impulse if self.disturbance=='impulse' and self.steps==self.impulse_step else np.zeros(3)
        self.obj,self.vel,self.omega,self.contact,peak=integrate_ball(self.obj,self.vel,self.omega,
            pads0,self.pads(),self.cfg.object_radius,.009,self.table,self.mass,self.mu,
            self.cfg.control_dt/self.cfg.physics_substeps,self.cfg.physics_substeps,impulse)
        self.peak_force=max(self.peak_force,peak);self.last_action=a.copy()
        self.steps+=1;self.time=self.steps*self.cfg.control_dt
        self.lifted |= bool(self.obj[2]>self.table+self.cfg.object_radius+.035)
        resting=(np.linalg.norm(self.obj[:2]-self.goal[:2])<.027 and
          abs(self.obj[2]-self.goal[2])<.007 and np.linalg.norm(self.vel)<.10 and
          self.opening>.03 and np.linalg.norm(self.ee-self.obj)>.045 and self.lifted)
        self.stable=self.stable+1 if resting else 0;self.success=self.stable>=4
        failure=(not np.isfinite(self.obj).all() or np.linalg.norm(self.obj-self.goal)>.6)
        terminated=bool(self.success or failure);truncated=self.steps>=self.cfg.horizon and not terminated
        self.done=terminated or truncated;self._record()
        score=self.progress()
        return self.observe(),score,terminated,truncated,{'success':self.success,
            'failure':bool(failure),'score':score,'peak_force':self.peak_force}
    def progress(self):
        # Fixed continuous development utility; NOT a success rate.
        approach=np.exp(-np.linalg.norm(self.ee-self.obj)/.05)
        lift=np.clip((self.obj[2]-self.table-self.cfg.object_radius)/.055,0.,1.)
        transport=np.exp(-np.linalg.norm(self.obj[:2]-self.goal[:2])/.05)
        return float(.15*approach+.2*lift+.35*transport+.3*self.success)
    def snapshot(self):return copy.deepcopy(self.__dict__)
    def restore(self,state):self.__dict__=copy.deepcopy(state);return self.observe()
    def fork(self):
        other=type(self)(self.cfg);other.restore(self.snapshot());return other
    def summary(self):return {'seed':self.seed,'split':self.split,'disturbance':self.disturbance,
        'backend':self.backend,'success':bool(self.success),'steps':self.steps,
        'score':self.progress(),'peak_force':float(self.peak_force),'lifted':bool(self.lifted)}

class History:
    def __init__(self,obs,length=HISTORY):
        self.length=int(length);self.values=[np.asarray(obs,np.float32).copy() for _ in range(length)]
    def append(self,obs):self.values=(self.values+[np.asarray(obs,np.float32).copy()])[-self.length:]
    def array(self):return np.stack(self.values)
    def copy(self):return copy.deepcopy(self)

def make_env(backend='contact_proxy',config=None):
    if backend=='contact_proxy':return ContactEnv(config)
    if backend=='mujoco':
        from .mujoco_env import MuJoCoEnv
        return MuJoCoEnv(config)
    raise ValueError('backend must be contact_proxy or mujoco')
