"""Command-line workflows with explicit backends and no-overwrite outputs."""
import argparse,json,importlib.util,platform
from pathlib import Path
import numpy as np
from .environment import TaskConfig,History,make_env
from .policy import ChunkPolicy,initialize
from .local_policy import load_policy
from .teacher import RecoveryTeacher
from .data import collect_demos,load_segments,save_segments
from .experiment import StudyConfig,run_study,fit_recovery,evaluate,write_json,audit_study,rollout

def fresh(path):
    p=Path(path)
    if p.exists() and (not p.is_dir() or any(p.iterdir())):raise FileExistsError(f'Nonempty output: {p}')
    p.mkdir(parents=True,exist_ok=True);return p

def config(path):
    with open(path,encoding='utf-8') as f:return StudyConfig(**json.load(f)).validate()

def make_demo(out,checkpoint=None,controller='policy',backend='contact_proxy',seed=99000,gif=False,disturbance='none'):
    out=fresh(out);cfg=TaskConfig(horizon=130)
    if controller=='policy':
        if checkpoint is None:raise ValueError('policy replay requires --checkpoint')
        m=load_policy(checkpoint);record=rollout(m,seed,'development',backend,cfg,disturbance,save=True)
        e=make_env(backend,cfg);e.reset(seed,'development',disturbance);record['goal']=e.goal.copy()
    else:
        e=make_env(backend,cfg);e.reset(seed,'development',disturbance);teacher=RecoveryTeacher();actions=[]
        while not e.done:
            a=teacher.action(e);e.step(a);actions.append(a)
        record=e.summary();record['trajectory']={k:np.stack([r[k] for r in e.trace]) for k in e.trace[0]}
        record['trajectory']['action']=np.asarray(actions);record['goal']=e.goal.copy()
    record['controller']=controller
    if checkpoint is not None:
        import hashlib
        record['checkpoint_sha256']=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    np.savez_compressed(out/'trajectory.npz',**record['trajectory'])
    write_json(out/'metrics.json',{k:v for k,v in record.items() if k!='trajectory'})
    if gif:
        from .visualization import render_gif
        render_gif(record,out/'demo.gif')
    return record

def parser():
    p=argparse.ArgumentParser(prog='koral',description='Support-gated local residual learning for robotic manipulation')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('doctor');a.add_argument('--backend',choices=['contact_proxy','mujoco'],default='contact_proxy')
    a=s.add_parser('reproduce');a.add_argument('--config',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('collect');a.add_argument('--episodes',type=int,default=30);a.add_argument('--backend',default='contact_proxy',choices=['contact_proxy','mujoco']);a.add_argument('--out',required=True)
    a=s.add_parser('train');a.add_argument('--data',required=True);a.add_argument('--steps',type=int,default=500);a.add_argument('--seed',type=int,default=0);a.add_argument('--out',required=True)
    a=s.add_parser('adapt');a.add_argument('--prior',required=True);a.add_argument('--data',required=True);a.add_argument('--config',required=True);a.add_argument('--method',default='gain');a.add_argument('--seed',type=int,default=0);a.add_argument('--out',required=True)
    a=s.add_parser('evaluate');a.add_argument('--checkpoint',required=True);a.add_argument('--backend',choices=['contact_proxy','mujoco'],default='contact_proxy');a.add_argument('--episodes',type=int,default=20);a.add_argument('--seed',type=int,default=171000);a.add_argument('--out',required=True)
    a=s.add_parser('demo');a.add_argument('--checkpoint');a.add_argument('--controller',choices=['policy','teacher'],default='policy');a.add_argument('--backend',choices=['contact_proxy','mujoco'],default='contact_proxy');a.add_argument('--seed',type=int,default=99000);a.add_argument('--out',required=True);a.add_argument('--gif',action='store_true');a.add_argument('--disturbance',default='none',choices=['none','impulse','low_friction'])
    a=s.add_parser('audit');a.add_argument('--results',required=True)
    a=s.add_parser('verify');a.add_argument('--results',required=True);a.add_argument('--methods',nargs='+',default=['bc','koral']);a.add_argument('--banks',nargs='+',default=['reference'])
    a=s.add_parser('report');a.add_argument('--results',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('export-model');a.add_argument('--out',required=True)
    a=s.add_parser('sac');a.add_argument('--prior',required=True);a.add_argument('--backend',choices=['contact_proxy','mujoco'],default='contact_proxy');a.add_argument('--steps',type=int,default=5000);a.add_argument('--seed',type=int,default=0);a.add_argument('--out',required=True)
    return p

def main(argv=None):
    import torch
    torch.set_num_threads(1)
    p=parser();a=p.parse_args(argv)
    try:
        if a.command=='doctor':
            d={'python':platform.python_version(),'backend_requested':a.backend,
                'mujoco_installed':importlib.util.find_spec('mujoco') is not None}
            e=make_env(a.backend);e.reset(7);e.step([0,0,0,1]);d['backend_step_passed']=True
            print(json.dumps(d,indent=2));return
        if a.command=='reproduce':
            from .workflow import reproduce
            print(json.dumps(reproduce(a.config,a.out),indent=2));return
        if a.command=='collect':
            out=fresh(a.out);seg,rec=collect_demos(a.episodes,backend=a.backend)
            save_segments(out/'demos.npz',seg);write_json(out/'episodes.json',rec);return
        if a.command=='train':
            out=fresh(a.out);d=load_segments(a.data);nval=max(1,len(d)//5)
            initialize(d[:-nval],a.seed,a.steps).save(out/'policy.pt')
            write_json(out/'split.json',{'train':[s.episode_id for s in d[:-nval]],'held_out':[s.episode_id for s in d[-nval:]]});return
        if a.command=='adapt':
            out=fresh(a.out);d=load_segments(a.data);nval=max(1,len(d)//5)
            from .workflow import read_config
            from .local_study import fit_local
            fit_local(ChunkPolicy.load(a.prior),d[:-nval],a.seed,read_config(a.config),out,mode={'koral':'gain'}.get(a.method,a.method));return
        if a.command=='evaluate':
            out=fresh(a.out);records=evaluate(load_policy(a.checkpoint),range(a.seed,a.seed+a.episodes),'test',a.backend,TaskConfig())
            write_json(out/'episodes.json',records);return
        if a.command=='demo':
            if a.controller=='policy' and not a.checkpoint:p.error('--checkpoint required for policy demo')
            r=make_demo(a.out,a.checkpoint,a.controller,a.backend,a.seed,a.gif,a.disturbance)
            print(json.dumps({k:v for k,v in r.items() if k not in ('trajectory','goal')},indent=2));return
        if a.command=='audit':
            from .verification import audit_results
            print(json.dumps(audit_results(a.results),indent=2));return
        if a.command=='verify':
            from .verification import verify_replay
            print(json.dumps(verify_replay(a.results,methods=tuple(a.methods),banks=tuple(a.banks)),indent=2));return
        if a.command=='report':
            from .reporting import make_report
            make_report(a.results,a.out);return
        if a.command=='export-model':
            from .mujoco_env import build_xml
            out=Path(a.out)
            if out.exists():raise FileExistsError(out)
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(build_xml(),encoding='utf-8');return
        if a.command=='sac':
            from .residual_sac import train
            train(ChunkPolicy.load(a.prior),a.out,a.backend,a.steps,a.seed);return
    except (ValueError,FileNotFoundError,FileExistsError,ImportError) as exc:
        p.exit(2,f'koral: {exc}\n')

if __name__=='__main__':main()
