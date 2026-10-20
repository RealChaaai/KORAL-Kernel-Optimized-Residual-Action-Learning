"""One command: demonstrations, baselines, local recovery, evaluation and audit."""
from pathlib import Path
from dataclasses import asdict,replace
import json,platform,time
import numpy as np
import torch
from .data import collect_demos,save_segments
from .environment import TaskConfig
from .policy import initialize
from .experiment import write_json,fit_recovery
from .local_policy import LocalResidualPolicy,LocalConfig
from .local_study import LocalStudyConfig,fit_local,freeze_run,evaluate_frozen
from .verification import audit_results

METHODS=['bc','random','uncertainty','gain','no_sequence','koral','no_recovery',
         'no_affine','no_support','retrieval','local_action']


def read_config(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    raw.pop('methods',None)
    return LocalStudyConfig(**raw).validate()


def reproduce(path,out):
    cfg=read_config(path);out=Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Output is not empty: {out}')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    started=time.perf_counter()
    write_json(out/'config.json',asdict(cfg)|{'methods':METHODS})
    write_json(out/'runtime.json',{'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'platform':platform.platform(),'torch_threads':1})
    demos,records=collect_demos(cfg.demos,backend=cfg.backend,config=TaskConfig(horizon=cfg.horizon))
    nval=max(1,len(demos)//5);train=demos[:-nval]
    save_segments(out/'demos.npz',demos);write_json(out/'teacher_records.json',records)
    write_json(out/'data_split.json',{'train_episode_ids':[s.episode_id for s in train],
        'validation_episode_ids':[s.episode_id for s in demos[-nval:]],
        'acquisition_scene_start':10000,'credit_scene_start':34000,'retention_scene_start':64000})
    for seed in cfg.train_seeds:
        print(f'Training seed {seed}: action-chunk prior',flush=True)
        folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        prior=initialize(train,seed,cfg.initial_steps,horizon=cfg.chunk_horizon)
        prior.save(folder/'prior.pt')
        for method in cfg.legacy().methods:
            print(f'Training seed {seed}: {method}',flush=True)
            fit_recovery(prior,train,seed,method,cfg.legacy(),folder/method)
        base=LocalResidualPolicy.fit(prior,train,LocalConfig(**cfg.local))
        base.save(folder/'no_recovery/policy.pt')
        model,_=fit_local(prior,train,seed,cfg,folder/'koral')
        model.with_config(replace(model.config,regression='constant')).save(folder/'no_affine/policy.pt')
        model.with_config(replace(model.config,support_gate=False)).save(folder/'no_support/policy.pt')
        model.with_config(replace(model.config,regression='constant',target='action',support_gate=False)).save(folder/'retrieval/policy.pt')
        model.with_config(replace(model.config,target='action')).save(folder/'local_action/policy.pt')
    freeze_run(out,cfg,METHODS);evaluate_frozen(out,cfg,METHODS)
    audit=audit_results(out)
    from .reporting import make_report
    make_report(out,out/'REPORT.md')
    write_json(out/'COMPLETE.json',{'wall_seconds':time.perf_counter()-started,'audit':audit})
    return audit
