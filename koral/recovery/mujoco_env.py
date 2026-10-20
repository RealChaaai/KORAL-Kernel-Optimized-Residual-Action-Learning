"""Independent MJCF task with actuated 7R arm, parallel gripper and free sphere.

No external robot/task repository or robot mesh is imported. MuJoCo is a physics
engine dependency. Import failure is explicit; there is no proxy substitution.
"""
from pathlib import Path
import copy
import numpy as np
from ..robot import A,D,ALPHA,HOME,LIMITS,chain
from .environment import ContactEnv

def _quat(R):
    # Stable eigenvector construction of scalar-first unit quaternion.
    r=R
    K=np.array([[r[0,0]-r[1,1]-r[2,2],r[1,0]+r[0,1],r[2,0]+r[0,2],r[2,1]-r[1,2]],
                [r[1,0]+r[0,1],r[1,1]-r[0,0]-r[2,2],r[2,1]+r[1,2],r[0,2]-r[2,0]],
                [r[2,0]+r[0,2],r[2,1]+r[1,2],r[2,2]-r[0,0]-r[1,1],r[1,0]-r[0,1]],
                [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],np.trace(r)]])/3
    _,v=np.linalg.eigh(K);xyzw=v[:,-1];q=xyzw[[3,0,1,2]]
    return q if q[0]>=0 else -q

def build_xml(timestep=.002):
    ee=chain(HOME)[0][:3,3];R=chain(HOME)[0][:3,:3];q=_quat(R.T)
    table=ee[2]-.12
    lines=[f'''<mujoco model="koral_recovery">
    <compiler angle="radian" autolimits="true"/>
    <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 -9.81"/>
    <default><joint damping="3" armature="0.08"/>
      <geom condim="4" friction="0.9 0.005 0.0001" solref="0.006 1"/>
    </default><worldbody>
    <light pos="0 0 2"/>
    <geom name="table" type="box" pos="{ee[0]+.03} {ee[1]} {table-.02}" size=".24 .20 .02" contype="4" conaffinity="3"/>
    <body name="base"><geom type="cylinder" pos="0 0 .04" size=".06 .04" contype="0" conaffinity="0"/>''']
    for i,(a,d,alpha,bounds) in enumerate(zip(A,D,ALPHA,LIMITS)):
        lines.append(f'<body name="link{i+1}" gravcomp="1"><joint name="j{i+1}" type="hinge" axis="0 0 1" range="{bounds[0]} {bounds[1]}"/>')
        if np.hypot(a,d)<1e-5:
            lines.append('<geom type="sphere" size=".023" mass=".35" contype="1" conaffinity="6"/>')
        else:
            lines.append(f'<geom type="capsule" fromto="0 0 0 {a} 0 {d}" size=".020" mass=".8" contype="1" conaffinity="6"/>')
        lines.append(f'<body name="frame{i+1}" pos="{a} 0 {d}" quat="{np.cos(alpha/2)} {np.sin(alpha/2)} 0 0">')
    lines.append(f'<body name="hand" quat="{" ".join(map(str,q))}" gravcomp="1"><site name="tcp" pos="0 0 0" size=".005"/>')
    lines.append('<geom name="palm" type="box" pos="0 0 .055" size=".035 .045 .015" mass=".15" contype="1" conaffinity="6"/>')
    for sign in (1,-1):
        name='left' if sign==1 else 'right'
        lines.append(f'''<body name="{name}_finger" gravcomp="1">
            <joint name="{name}_slide" type="slide" axis="0 {sign} 0" range=".008 .04" damping="1" armature=".002"/>
            <geom name="{name}_pad" type="box" pos="0 {sign*.009} 0" size=".025 .009 .026" mass=".045" contype="1" conaffinity="6"/>
            </body>''')
    lines.append('</body>'+'</body></body>'*7+'</body>')
    lines.append(f'''<body name="object" pos="{ee[0]} {ee[1]} {table+.017}">
        <freejoint name="object_free"/><geom name="object_geom" type="sphere" size=".017" mass=".06" contype="2" conaffinity="5"/></body>
        <site name="goal" pos="{ee[0]+.075} {ee[1]-.045} {table+.017}" size=".025"/>
        </worldbody><actuator>''')
    for i in range(7):
        lines.append(f'<position name="servo{i+1}" joint="j{i+1}" kp="350" kv="35" forcerange="-87 87"/>')
    for name in ('left','right'):
        lines.append(f'<position joint="{name}_slide" kp="400" kv="8" forcerange="-15 15"/>')
    lines.append('</actuator></mujoco>');return '\n'.join(lines)

class MuJoCoEnv(ContactEnv):
    backend='mujoco'
    def __init__(self,config=None):
        try:import mujoco
        except ImportError as exc:
            raise ImportError('MuJoCo backend requires pip install -e ".[physics]". No fallback backend was used.') from exc
        self.mj=mujoco;super().__init__(config)
        if abs(self.cfg.object_radius-.017)>1e-12:
            raise ValueError('This MJCF asset uses a 0.017 m object radius')
        self.model=mujoco.MjModel.from_xml_string(build_xml(self.cfg.control_dt/20))
        self.data=mujoco.MjData(self.model)
        self.tcp_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'tcp')
        self.obj_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'object')
        self.ball_q=int(self.model.jnt_qposadr[self.model.joint('object_free').id])
        self.ball_v=int(self.model.jnt_dofadr[self.model.joint('object_free').id])
        self.pad_ids=[self.model.geom(n).id for n in ('left_pad','right_pad')]
        self.obj_geom=self.model.geom('object_geom').id
        self._spec=mujoco.mjtState.mjSTATE_INTEGRATION
    def reset(self,seed=0,split='train',disturbance='none'):
        super().reset(seed,split,disturbance);m,d=self.model,self.data;self.mj.mj_resetData(m,d)
        m.body_mass[self.obj_id]=self.mass;m.body_inertia[self.obj_id]=.4*self.mass*self.cfg.object_radius**2
        for gid in self.pad_ids+[self.obj_geom]:m.geom_friction[gid,0]=self.mu
        m.site_pos[m.site('goal').id]=self.goal
        self.mj.mj_setConst(m,d)
        d.qpos[:7]=HOME;d.qpos[7:9]=.038
        d.qpos[self.ball_q:self.ball_q+3]=self.obj;d.qpos[self.ball_q+3:self.ball_q+7]=[1,0,0,0]
        d.ctrl[:7]=HOME;d.ctrl[7:9]=.038;self.mj.mj_forward(m,d)
        self._sync();self.trace=[];self._record();return self.observe()
    def _sync(self):
        m,d=self.model,self.data;self.q=d.qpos[:7].copy();self.qvel=d.qvel[:7].copy()
        self.ee=d.site_xpos[self.tcp_id].copy();self.obj=d.xpos[self.obj_id].copy()
        self.vel=d.qvel[self.ball_v:self.ball_v+3].copy();self.omega=d.qvel[self.ball_v+3:self.ball_v+6].copy()
        self.opening=float(d.qpos[7:9].mean());jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
        self.mj.mj_jacSite(m,d,jp,jr,self.tcp_id);self.ee_vel=jp@d.qvel
        self.contact=np.zeros(2)
        force=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i]
            if self.obj_geom not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.obj_geom else c.geom1
            if other in self.pad_ids:
                self.mj.mj_contactForce(m,d,i,force);j=self.pad_ids.index(other)
                self.contact[j]+=max(0,float(force[0]))
    def step(self,action):
        if self.done:raise RuntimeError('reset required')
        a=np.asarray(action,float)
        if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('finite four-dimensional action required')
        a=np.clip(a,-1,1);m,d=self.model,self.data
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);self.mj.mj_jacSite(m,d,jp,jr,self.tcp_id)
        R=d.site_xmat[self.tcp_id].reshape(3,3);target_R=np.eye(3)
        er=.5*sum((np.cross(R[:,i],target_R[:,i]) for i in range(3)),start=np.zeros(3))
        J=np.vstack([jp[:,:7],jr[:,:7]])
        err=np.r_[a[:3]*self.cfg.action_scale,np.clip(er,-.08,.08)]
        dq=J.T@np.linalg.solve(J@J.T+.025**2*np.eye(6),err)
        d.ctrl[:7]=np.clip(self.q+np.clip(dq,-.13,.13),LIMITS[:,0]+.01,LIMITS[:,1]-.01)
        d.ctrl[7:9]=.012+(a[3]+1)*.5*(.038-.012)
        if self.disturbance=='impulse' and self.steps==self.impulse_step:
            d.qvel[self.ball_v:self.ball_v+3]+=self.impulse/self.mass
        self.mj.mj_step(m,d,nstep=20);self._sync()
        self.peak_force=max(self.peak_force,float(self.contact.max()));self.last_action=a.copy()
        self.steps+=1;self.time=float(d.time);self.lifted|=bool(self.obj[2]>self.table+self.cfg.object_radius+.035)
        resting=(np.linalg.norm(self.obj[:2]-self.goal[:2])<.027 and abs(self.obj[2]-self.goal[2])<.007 and
                 np.linalg.norm(self.vel)<.10 and self.opening>.03 and np.linalg.norm(self.ee-self.obj)>.045 and self.lifted)
        self.stable=self.stable+1 if resting else 0;self.success=self.stable>=4
        failure=bool(not np.isfinite(d.qpos).all() or np.linalg.norm(self.obj-self.goal)>.6)
        te=bool(self.success or failure);tr=self.steps>=self.cfg.horizon and not te;self.done=te or tr;self._record()
        return self.observe(),self.progress(),te,tr,{'success':self.success,'failure':failure,'score':self.progress()}
    def snapshot(self):
        state=np.empty(self.mj.mj_stateSize(self.model,self._spec));self.mj.mj_getState(self.model,self.data,state,self._spec)
        fields={k:copy.deepcopy(v) for k,v in self.__dict__.items() if k not in ('mj','model','data','_spec')}
        fields['_integration_state']=state;return fields
    def restore(self,state):
        self.reset(state['seed'],state['split'],state['disturbance'])
        fields={k:copy.deepcopy(v) for k,v in state.items() if k!='_integration_state'};self.__dict__.update(fields)
        self.mj.mj_setState(self.model,self.data,state['_integration_state'],self._spec);self.mj.mj_forward(self.model,self.data)
        self._sync();return self.observe()
