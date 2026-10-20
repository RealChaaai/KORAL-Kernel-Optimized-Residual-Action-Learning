import copy
import numpy as np
import pytest
import torch
from koral.recovery.environment import ContactEnv,TaskConfig,History,OBS_DIM,make_env
from koral.recovery.teacher import RecoveryTeacher
from koral.recovery.budget import Budget
from koral.recovery.data import Segment,collect_segment,examples,save_segments,load_segments
from koral.recovery.policy import ChunkPolicy,sequence_loss,initialize,update
from koral.recovery.acquisition import Candidate,GainSelector,features
from koral.recovery.experiment import development_utility,StudyConfig,summarize
from koral.recovery.residual_sac import Actor,ResidualAgent

@pytest.fixture
def segment():
 e=ContactEnv();o=e.reset(123,'train');return collect_segment(e,History(o),20)

@pytest.mark.parametrize('seed',[1,7,31])
def test_teacher_physical_pick_place(seed):
 e=ContactEnv();e.reset(seed);t=RecoveryTeacher()
 while not e.done:e.step(t.action(e))
 assert e.success and e.lifted
 assert e.peak_force>0
 assert np.linalg.norm(e.obj[:2]-e.goal[:2])<.027

@pytest.mark.parametrize('bad',[[0,0,0],[0,0,0,np.nan],[0,0,np.inf,0]])
def test_invalid_action(bad):
 e=ContactEnv();e.reset()
 with pytest.raises(ValueError):e.step(bad)

def test_observation_no_teacher_state():
 e=ContactEnv();a=e.reset(1);e.phase=6;e.teacher_subgoal=np.ones(3);e.mu=.01
 assert np.array_equal(a,e.observe()) and len(a)==OBS_DIM

def test_no_attach_gravity():
 e=ContactEnv();e.reset(2);e.obj[2]+=0.06;e.obj[0]+=.1
 start=e.obj.copy();e.step([0,0,0,1])
 assert e.obj[2]<start[2] and e.vel[2]<0
 assert not hasattr(e,'attached')

def test_open_fingers_do_not_lift():
 e=ContactEnv();e.reset(8);t=RecoveryTeacher()
 while not e.done:
  a=t.action(e);a[3]=1;e.step(a)
 assert not e.lifted and not e.success

def test_release_is_dynamic():
 e=ContactEnv();e.reset(7);t=RecoveryTeacher()
 while not e.done and e.obj[2]<e.table+.067:e.step(t.action(e))
 assert e.obj[2]>e.table+.06
 height=e.obj[2]
 for _ in range(16):
  if not e.done:e.step([0,0,0,1])
 assert e.obj[2]<height-.02

def test_table_contact():
 e=ContactEnv();e.reset(2)
 for _ in range(12):e.step([0,0,0,1])
 assert abs(e.obj[2]-e.table-e.cfg.object_radius)<.003

def test_exact_snapshot_replay():
 e=ContactEnv();e.reset(7,'acquire',disturbance='impulse');e.step([.1,.1,-.2,1]);s=e.snapshot()
 first=e.step([.2,0,0,-1]);p=e.obj.copy();e.restore(s);second=e.step([.2,0,0,-1])
 assert np.array_equal(first[0],second[0]) and np.array_equal(p,e.obj)

def test_fork_does_not_modify_parent():
 e=ContactEnv();e.reset(2);s=e.observe().copy();f=e.fork();f.step([1,0,0,1]);assert np.array_equal(s,e.observe())

def test_no_silent_backend_fallback():
 with pytest.raises(ValueError):make_env('missing')

def test_query_after_done_fails():
 e=ContactEnv(TaskConfig(horizon=1));e.reset();e.step([0,0,0,1])
 with pytest.raises(RuntimeError):e.step([0,0,0,1])

@pytest.mark.parametrize('split',['test','stress','development'])
def test_no_heldout_teacher_queries(split):
 e=ContactEnv();o=e.reset(8,split)
 with pytest.raises(ValueError):collect_segment(e,History(o),2)

def test_budget_exact():
 e=ContactEnv();o=e.reset(8,'acquire');b=Budget(8);s=collect_segment(e,History(o),8,b)
 assert b.expert_actions==b.environment_steps==len(s.actions)==8
 assert b.audit() and b.remaining==0
 with pytest.raises(RuntimeError):b.charge()

def test_budget_rejects_unaffordable():
 with pytest.raises(ValueError):Budget(-1)
 b=Budget(4)
 with pytest.raises(ValueError):b.begin(0,5)
 assert b.expert_actions==0

def test_failed_expert_still_charged(monkeypatch):
 e=ContactEnv();o=e.reset(8,'acquire');b=Budget(4)
 def fail(*args):raise RuntimeError('planner failure')
 monkeypatch.setattr(RecoveryTeacher,'action',fail)
 with pytest.raises(RuntimeError):collect_segment(e,History(o),4,b)
 assert b.expert_actions==1 and b.environment_steps==0

def test_segments_are_contiguous(segment):
 assert np.array_equal(segment.next_histories[:-1],segment.histories[1:])

def test_chunks_do_not_cross_segments(segment):
 ex=examples([segment],8)
 assert ex['mask'][-1].sum()==1 and ex['mask'][-2].sum()==2
 assert np.array_equal(ex['y'][-1,0],segment.actions[-1])

def test_segment_io_roundtrip(segment,tmp_path):
 p=tmp_path/'s.npz';save_segments(p,[segment]);s=load_segments(p)[0]
 assert np.array_equal(s.actions,segment.actions)
 assert np.array_equal(s.histories,segment.histories)

def test_padding_has_zero_gradient():
 p=torch.zeros(1,8,4,requires_grad=True);y=torch.ones_like(p);mask=torch.tensor([[1,1,0,0,0,0,0,0.]])
 loss=sequence_loss(p,y,mask,torch.zeros(1,3),torch.zeros(1,3));loss.backward()
 assert torch.count_nonzero(p.grad[:,2:])==0

def test_no_cross_episode_transition(segment):
 segment.terminated[3]=True
 with pytest.raises(ValueError):segment.validate()

def test_policy_bounded_and_checkpoint(tmp_path):
 torch.manual_seed(0);m=ChunkPolicy();h=np.zeros((4,32),np.float32);a=m.chunk(h)
 assert a.shape==(8,4) and np.max(np.abs(a))<=1
 p=tmp_path/'policy.pt';m.save(p);n=ChunkPolicy.load(p)
 assert np.array_equal(m.chunk(h),n.chunk(h))

def test_old_checkpoint_rejected(tmp_path):
 p=tmp_path/'old.pt';torch.save({'state_dict':{}},p)
 with pytest.raises(ValueError):ChunkPolicy.load(p)

def test_identical_shadow_updates_identical(segment):
 torch.manual_seed(2);m=ChunkPolicy(members=1,width=24)
 a,_=update(m,[segment],[],steps=3,seed=9)
 b,_=update(m,[segment],[],steps=3,seed=9)
 assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))

def test_update_does_not_mutate_input(segment):
 m=ChunkPolicy(members=1,width=24);state=copy.deepcopy(m.state_dict());update(m,[segment],[],steps=1)
 assert all(torch.equal(state[k],v) for k,v in m.state_dict().items())

def test_feature_no_private_parameters():
 e=ContactEnv();o=e.reset();c=Candidate(0,e.snapshot(),History(o),.01)
 x=features(c,16,32);c.state['mu']=999;c.state['phase']=10
 assert np.array_equal(x,features(c,16,32))

def test_selector_updates_from_actual_gain():
 e=ContactEnv();o=e.reset();c=Candidate(0,e.snapshot(),History(o),.01);x=features(c,16,32)
 s=GainSelector(seed=0);before=s.predict(x)[0];target=s.observe(x,.4,16)
 assert target==.025 and s.predict(x)[0]>before and s.n==1

def test_negative_gains_are_kept():
 e=ContactEnv();o=e.reset();c=Candidate(0,e.snapshot(),History(o),.01);x=features(c,16,32)
 s=GainSelector();assert s.observe(x,-.2,8)<0 and s.n==1
 assert s.predict(x)[0]<0

@pytest.mark.parametrize('mode',['gain','random','uncertainty','fixed_length','no_sequence'])
def test_selector_only_affordable(mode):
 e=ContactEnv();o=e.reset();c=Candidate(0,e.snapshot(),History(o),.01)
 s=GainSelector(seed=0);_,h,_,_=s.choose([c],[8,16,32],10,mode)
 assert h==8

def test_utility_rejects_test():
 with pytest.raises(ValueError):development_utility([{'split':'test','success':1,'score':1}])
 assert development_utility([{'split':'development','success':1,'score':.5}])==1.1

def test_bootstrap_matches_pairing():
 rows=[]
 for s in (0,1,2):
  for method in ('random','gain'):
   for ep in range(4):rows.append({'method':method,'split':'test','training_seed':s,'seed':ep,'success':ep<2,'score':.1})
 result=summarize(rows,[0,1,2],['random','gain'],bootstrap=20)
 assert result['comparisons']['gain_minus_random']['paired_crossed_bootstrap_95']==[0.,0.]

def test_sac_action_density_finite():
 a=Actor();x=torch.zeros(5,128);base=torch.ones(5,4)*.98
 action,lp=a.sample(x,base)
 assert torch.isfinite(lp).all() and torch.max(action.abs())<=1
 lp.mean().backward();assert all(torch.isfinite(p.grad).all() for p in a.parameters())

def test_sac_critic_update():
 prior=ChunkPolicy(members=1,width=24);s=ResidualAgent(prior)
 rng=np.random.default_rng(1)
 batch=(rng.normal(size=(4,4,32)),np.zeros((4,4)),np.zeros((4,1)),rng.normal(size=(4,4,32)),np.zeros((4,1)))
 out=s.learn(batch);assert all(np.isfinite(v) for v in out.values())
 assert all(not p.requires_grad for p in s.prior.parameters())

def test_bad_config():
 with pytest.raises(ValueError):StudyConfig(execute_prefix=10).validate()

def test_retention_rejects_degrading_update():
 from koral.recovery.retention import admit
 m=ChunkPolicy(members=1,width=24);trial=copy.deepcopy(m)
 with torch.no_grad():
  for p in trial.parameters():p.add_(1)
 ref=next(m.parameters()).detach().clone()
 def evaluator(model):
  deviation=float((next(model.parameters()).detach()-ref).abs().mean())
  return [{'split':'development','success':deviation<1e-7,'score':1-deviation}]
 accepted,log=admit(m,trial,evaluator,development_utility)
 assert log[-1]['weight']==0
 assert torch.equal(next(accepted.parameters()),next(m.parameters()))

def test_retention_accepts_equal_update():
 from koral.recovery.retention import admit
 m=ChunkPolicy(members=1,width=24)
 fn=lambda p:[{'split':'development','success':True,'score':.5}]
 _,log=admit(m,copy.deepcopy(m),fn,development_utility)
 assert len(log)==1 and log[0]['weight']==1

def test_interpolation_endpoints():
 from koral.recovery.retention import interpolate
 a=ChunkPolicy(members=1,width=24);b=ChunkPolicy(members=1,width=24)
 zero=interpolate(a,b,0);one=interpolate(a,b,1)
 assert all(torch.equal(x,y) for x,y in zip(zero.parameters(),a.parameters()))
 assert all(torch.equal(x,y) for x,y in zip(one.parameters(),b.parameters()))
 with pytest.raises(ValueError):interpolate(a,b,2)

def test_cli_teacher_flag():
 from koral.recovery.cli import parser
 args=parser().parse_args(['demo','--controller','teacher','--out','x'])
 assert args.backend=='contact_proxy' and args.controller=='teacher'

def test_migration_checkpoint_blocked_in_cli(tmp_path):
 from koral.recovery.cli import main
 with pytest.raises(SystemExit) as err:main(['demo','--out',str(tmp_path/'out')])
 assert err.value.code==2

def test_reject_nonempty_output(tmp_path):
 from koral.recovery.cli import fresh
 (tmp_path/'record.json').write_text('{}')
 with pytest.raises(FileExistsError):fresh(tmp_path)
