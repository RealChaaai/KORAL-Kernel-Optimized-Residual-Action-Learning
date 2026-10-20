"""Executable checks for the local residual model, provenance and exact fallback."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from koral.recovery.environment import ContactEnv, TaskConfig, History, HISTORY, OBS_DIM
from koral.recovery.data import collect_segment
from koral.recovery.policy import ChunkPolicy
from koral.recovery.local_policy import (LocalConfig, LocalResidualPolicy,
                                         contact_features, load_policy, FEATURE_DIM)
from koral.recovery.local_study import weight_digest, retention_accept

@pytest.fixture
def example():
    e=ContactEnv(TaskConfig(horizon=50));o=e.reset(12,'train')
    return collect_segment(e,History(o),24)

@pytest.fixture
def prior():
    torch.manual_seed(10)
    return ChunkPolicy(members=2,width=24,horizon=4)

@pytest.mark.parametrize('field,value',[
    ('neighbors',0),('ridge',0),('ridge',float('nan')),('slope_penalty',-1),
    ('support_radius',0),('max_residual',0),('strength',1.1),
    ('regression','bad'),('target','bad'),('bandwidth_floor',-1)])
def test_invalid_config(field,value):
    with pytest.raises(ValueError):replace(LocalConfig(),**{field:value}).validate()

def test_shapes(prior,example):
    model=LocalResidualPolicy.fit(prior,[example])
    assert model.features.shape==(len(example.actions),FEATURE_DIM)
    assert model.chunk(example.histories[0]).shape==(4,4)

def test_immutable_prior(prior,example):
    before=weight_digest(prior);m=LocalResidualPolicy.fit(prior,[example])
    n=m.extend([example]);n.chunk(example.histories[2])
    assert before==weight_digest(m.prior)==weight_digest(n.prior)==weight_digest(prior)
    assert all(not p.requires_grad for p in m.prior.parameters())

def test_source_memory_not_mutated(prior,example):
    m=LocalResidualPolicy.fit(prior,[example]);h=m.histories.copy()
    n=m.extend([example])
    assert np.array_equal(h,m.histories) and len(n.histories)==2*len(h)
    assert n.histories is not m.histories

def test_empty_memory_exact_prior(prior,example):
    m=LocalResidualPolicy.fit(prior,[])
    assert np.array_equal(m.chunk(example.histories[0]),prior.chunk(example.histories[0]))

def test_out_of_support_exact_prior(prior,example):
    m=LocalResidualPolicy.fit(prior,[example]);h=example.histories[0].copy();h[:,:3]+=5
    assert m.explain(h)['gate']==0
    assert np.array_equal(m.chunk(h),prior.chunk(h))

def test_zero_strength_exact_prior(prior,example):
    m=LocalResidualPolicy.fit(prior,[example],LocalConfig(strength=0))
    assert np.array_equal(m.chunk(example.histories[1]),prior.chunk(example.histories[1]))

def test_change_bounded_by_support(prior,example):
    m=LocalResidualPolicy.fit(prior,[example],LocalConfig(max_residual=.15,strength=.7))
    for h in example.histories[::5]:
        a=m.chunk(h);b=prior.chunk(h);gate=m.explain(h)['gate']
        assert np.max(np.abs(a[0]-b[0]))<=.7*gate*.15+1e-6
        assert np.array_equal(a[1:],b[1:])

def test_checkpoint_roundtrip(prior,example,tmp_path):
    m=LocalResidualPolicy.fit(prior,[example]);p=tmp_path/'local.pt';m.save(p);n=load_policy(p)
    assert isinstance(n,LocalResidualPolicy)
    assert m.provenance==n.provenance
    for h in example.histories[::7]:assert np.array_equal(m.chunk(h),n.chunk(h))

def test_neural_checkpoint_still_loads(prior,example,tmp_path):
    p=tmp_path/'prior.pt';prior.save(p);m=load_policy(p)
    assert np.array_equal(prior.chunk(example.histories[1]),m.chunk(example.histories[1]))

def test_duplicate_rows_numerically_stable(prior,example):
    hs=np.repeat(example.histories[:1],32,axis=0);ys=np.ones((32,4))*.7
    m=LocalResidualPolicy(prior,hs,ys,LocalConfig(neighbors=16))
    a=m.chunk(hs[0]);assert np.isfinite(a).all() and np.abs(a).max()<=1
    assert m.explain(hs[0])['neighbors']==list(range(16))

def test_metric_excludes_clock_joint_identity(example):
    a=example.histories[3].copy();b=a.copy();b[:,13:27]+=2;b[:,31]+=.8
    assert np.array_equal(contact_features(a),contact_features(b))

def test_metric_uses_history(example):
    a=example.histories[3].copy();b=a.copy();b[0,:3]+=.03
    assert not np.array_equal(contact_features(a),contact_features(b))

@pytest.mark.parametrize('shape',[(32,),(3,32),(4,31)])
def test_bad_history(shape):
    with pytest.raises(ValueError):contact_features(np.zeros(shape))

def test_nan_history():
    h=np.zeros((4,32));h[2,5]=np.nan
    with pytest.raises(ValueError):contact_features(h)

@pytest.mark.parametrize('split',['test','stress','development'])
def test_no_heldout_memory(prior,example,split):
    s=copy.deepcopy(example);s.split=split
    with pytest.raises(ValueError):LocalResidualPolicy.fit(prior,[s])

def test_provenance_has_every_row(prior,example):
    m=LocalResidualPolicy.fit(prior,[example]);assert sum(x['rows'] for x in m.provenance)==len(m.histories)
    assert all(x['split']=='train' for x in m.provenance)

def test_no_teacher_at_inference(prior,example,monkeypatch):
    from koral.recovery.teacher import RecoveryTeacher
    def forbidden(*args,**kwargs):raise RuntimeError('teacher unavailable')
    m=LocalResidualPolicy.fit(prior,[example]);monkeypatch.setattr(RecoveryTeacher,'action',forbidden)
    assert np.isfinite(m.chunk(example.histories[10])).all()

def test_acquisition_fit_does_not_need_environment_state(prior,example):
    m=LocalResidualPolicy.fit(prior,[example])
    assert not hasattr(m,'environment') and not hasattr(m,'phase')
    assert not hasattr(m,'teacher') and not hasattr(m,'seed')

@pytest.mark.parametrize('config',[LocalConfig(regression='constant'),LocalConfig(target='action'),LocalConfig(support_gate=False)])
def test_ablation_interfaces(prior,example,config):
    m=LocalResidualPolicy.fit(prior,[example],config)
    for h in example.histories[::6]:assert np.isfinite(m.chunk(h)).all() and np.abs(m.chunk(h)).max()<=1

def test_retention_checks_both_conditions():
    a=[dict(seed=1,split='development',disturbance='none',success=True,score=.6),
       dict(seed=2,split='development',disturbance='low_friction',success=False,score=.2)]
    b=copy.deepcopy(a);b[0]['success']=False;b[1]['success']=True
    assert not retention_accept(a,b)
    assert retention_accept(a,a)

def test_retention_rejects_test_data():
    a=[dict(seed=1,split='test',disturbance='none',success=True,score=.6)]
    with pytest.raises(ValueError):retention_accept(a,a)
