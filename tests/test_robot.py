import numpy as np
import pytest
from koral.robot import HOME,LIMITS,forward,jacobian,chain,JointServo


def test_fk_finite():
    assert forward(HOME).shape==(3,)
    assert np.isfinite(forward(HOME)).all()


def test_fk_repeated():
    assert np.allclose(forward(HOME),forward(HOME))


def test_jacobian_differences():
    for i in range(7):
        eps=1e-6;dq=np.zeros(7);dq[i]=eps
        actual=(forward(HOME+dq)-forward(HOME-dq))/(2*eps)
        assert np.allclose(jacobian(HOME)[:,i],actual,rtol=1e-5,atol=1e-6)


def test_jacobian_rank():
    assert np.linalg.matrix_rank(jacobian(HOME))==3


def test_chain_shapes():
    T,origins,axes,points=chain(HOME)
    assert T.shape==(4,4) and origins.shape==(7,3) and axes.shape==(7,3) and points.shape==(8,3)


def test_joint_limits_respected():
    s=JointServo();q=HOME.copy()
    for _ in range(40):
        q,_=s.resolved_step(q,[.02,-.02,.015])
        assert (q>=LIMITS[:,0]).all() and (q<=LIMITS[:,1]).all()


def test_servo_moves_toward_goal():
    q,d=JointServo().resolved_step(HOME,[.01,-.005,.005])
    assert np.linalg.norm(d-[.01,-.005,.005])<.01


def test_robot_rejects_nonseven():
    with pytest.raises(ValueError): forward(np.zeros(6))
