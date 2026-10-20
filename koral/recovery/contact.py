"""Independent sphere--planar-pad friction dynamics for the contact-proxy backend.

The arm and fingers are prescribed kinematic boundary conditions. Only the free
sphere has inertial dynamics. This model is NOT substituted for MuJoCo results.
No object attachment, welding, pose copying, or prescribed transport is used.
"""
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(fn): return fn
        return decorate

@njit(cache=True)
def integrate_ball(pos, vel, omega, pads0, pads1, radius, pad_radius,
                   table, mass, mu, dt, substeps, impulse):
    p=pos.copy(); v=vel.copy(); w=omega.copy()
    inertia=0.4*mass*radius*radius
    contact=np.zeros(2); peak=0.; impulse_used=False
    pad_v=(pads1-pads0)/(dt*substeps)
    for k in range(substeps):
        if k==0:
            v+=impulse/mass
        F=np.array([0.,0.,-9.81*mass]); torque=np.zeros(3)
        pads=pads0+(pads1-pads0)*((k+1)/substeps)
        for j in range(2):
            side=1. if j==0 else -1.
            distance=side*(pads[j,1]-p[1])-pad_radius
            pen=radius-distance
            inside=(abs(p[0]-pads[j,0])<.025 and abs(p[2]-pads[j,2])<.026)
            if pen>0 and inside:
                n=np.array([0.,-side,0.]); r=-radius*n
                vr=v+np.cross(w,r)-pad_v[j]
                vn=np.dot(vr,n); vt=vr-vn*n
                # Bounded Kelvin--Voigt normal contact, regularized Coulomb friction.
                fn=min(15., max(0.,4500.*pen-2.5*vn))
                speed=np.sqrt(np.dot(vt,vt))
                ft=-mu*fn*vt/(speed+0.035)
                F+=fn*n+ft; torque+=np.cross(r,ft)
                contact[j]=max(contact[j],fn);peak=max(peak,fn)
        pen=table+radius-p[2]
        if pen>0:
            fn=max(0.,6500.*pen-4.*v[2]);r=np.array([0.,0.,-radius])
            vr=v+np.cross(w,r);vt=np.array([vr[0],vr[1],0.])
            ft=-0.6*fn*vt/(np.sqrt(np.dot(vt,vt))+0.025)
            F+=np.array([0.,0.,fn])+ft;torque+=np.cross(r,ft)
        v+=dt*F/mass;w+=dt*(torque-0.00002*w)/inertia
        p+=dt*v
    return p,v,w,contact,peak
