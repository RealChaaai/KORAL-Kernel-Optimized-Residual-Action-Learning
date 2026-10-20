import xml.etree.ElementTree as ET
import numpy as np
import pytest
from koral.recovery.mujoco_env import build_xml,_quat

def test_mjcf_independent_model_structure():
 root=ET.fromstring(build_xml())
 assert len(root.findall('.//worldbody//joint'))==9
 assert len(root.findall('.//freejoint'))==1
 assert len(root.find('actuator'))==9
 assert root.find('equality') is None
 assert root.find('asset') is None
 assert root.find('include') is None

def test_quaternion_identity():assert np.allclose(_quat(np.eye(3)),[1,0,0,0])

@pytest.mark.parametrize('theta',[.3,-.7,2.1])
def test_quaternion_roundtrip(theta):
 R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
 q=_quat(R)
 assert np.allclose(q,[np.cos(theta/2),0,0,np.sin(theta/2)])

@pytest.mark.physics
def test_mujoco_model_execution():
 pytest.importorskip('mujoco')
 from koral.recovery.mujoco_env import MuJoCoEnv
 e=MuJoCoEnv();o=e.reset(7);n=e.step([0,0,0,1])[0]
 assert e.model.nq==16 and e.model.nv==15 and np.isfinite(n).all()

@pytest.mark.physics
def test_mujoco_snapshot_execution():
 pytest.importorskip('mujoco')
 from koral.recovery.mujoco_env import MuJoCoEnv
 e=MuJoCoEnv();e.reset(7);s=e.snapshot();a=e.step([.1,0,0,1])[0]
 e.restore(s);b=e.step([.1,0,0,1])[0];assert np.allclose(a,b,atol=1e-8)

@pytest.mark.physics
def test_mujoco_free_object_falls():
 pytest.importorskip('mujoco')
 from koral.recovery.mujoco_env import MuJoCoEnv
 e=MuJoCoEnv();e.reset(7);e.data.qpos[e.ball_q+2]+=.06;e.mj.mj_forward(e.model,e.data);e._sync()
 z=e.obj[2];e.step([0,0,0,1]);assert e.obj[2]<z
