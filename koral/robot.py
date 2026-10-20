"""Analytic 7-revolute-joint robot geometry and resolved-rate motion control.

The transform chain is a simplified serial arm; parameters are declared here and
are not intended to reproduce any manufacturer's certified model.
"""
from dataclasses import dataclass
import numpy as np

A = np.array([0.,0.,0.,0.0825,-0.0825,0.,0.088],dtype=float)
D = np.array([0.333,0.,0.316,0.,0.384,0.,0.107],dtype=float)
ALPHA = np.array([0.,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2])
LIMITS = np.array([[-2.8973,2.8973],[-1.7628,1.7628],[-2.8973,2.8973],[-3.0718,-0.0698],[-2.8973,2.8973],[-0.0175,3.7525],[-2.8973,2.8973]])
HOME = np.array([0.,-0.65,0.,-1.9,0.,1.55,0.78])


def joint_matrix(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    """Standard DH transform: Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)."""
    c,s=np.cos(theta),np.sin(theta); ca,sa=np.cos(alpha),np.sin(alpha)
    return np.array([[c,-s*ca,s*sa,a*c], [s,c*ca,-c*sa,a*s], [0.,sa,ca,d], [0.,0.,0.,1.]])


def chain(q: np.ndarray):
    q=np.asarray(q,float)
    if q.shape!=(7,): raise ValueError('Expected seven revolute joints')
    T=np.eye(4); origins=[]; axes=[]; joints=[T[:3,3].copy()]
    for qi,ai,di,alpha in zip(q,A,D,ALPHA):
        origins.append(T[:3,3].copy()); axes.append(T[:3,2].copy())
        T=T@joint_matrix(qi,ai,di,alpha)
        joints.append(T[:3,3].copy())
    return T,np.asarray(origins),np.asarray(axes),np.asarray(joints)


def forward(q): return chain(q)[0][:3,3].copy()


def jacobian(q):
    T,origins,axes,_=chain(q); p=T[:3,3]
    return np.stack([np.cross(z,p-o) for z,o in zip(axes,origins)],axis=1)


@dataclass
class JointServo:
    damping: float = 0.065
    max_step: float = 0.16
    def resolved_step(self,q,cartesian_delta):
        """Damped least-squares IK with joint-rate and joint-position limits."""
        q=np.asarray(q,float); goal=np.asarray(cartesian_delta,float)
        J=jacobian(q)
        delta=J.T@np.linalg.solve(J@J.T+self.damping**2*np.eye(3),goal)
        delta=np.clip(delta,-self.max_step,self.max_step)
        next_q=np.clip(q+delta,LIMITS[:,0]+0.01,LIMITS[:,1]-0.01)
        return next_q, forward(next_q)-forward(q)

    def manipulability(self,q):
        s=np.linalg.svd(jacobian(q),compute_uv=False)
        return float(np.prod(s))
