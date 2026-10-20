"""Paired-shadow acquisition, autonomous evaluation, and split-aware provenance."""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy,json,time,platform,hashlib
import numpy as np
import torch
from .environment import make_env,History,TaskConfig
from .teacher import RecoveryTeacher
from .data import collect_demos,collect_segment,save_segments
from .policy import initialize,update,ChunkPolicy
from .acquisition import Candidate,GainSelector
from .budget import Budget

@dataclass(frozen=True)
class StudyConfig:
    backend:str='contact_proxy'
    demos:int=24
    initial_steps:int=400
    label_budget:int=64
    updates_per_action:int=2
    lengths:tuple=(8,16,32)
    pool_episodes:int=2
    candidate_stride:int=6
    development_episodes:int=3
    test_episodes:int=16
    train_seeds:tuple=(0,1,2)
    methods:tuple=('random','uncertainty','gain','fixed_length','no_sequence')
    horizon:int=130
    chunk_horizon:int=8
    execute_prefix:int=1
    retention_gate:bool=True
    def validate(self):
        if self.backend not in ('contact_proxy','mujoco'):raise ValueError('backend')
        if self.execute_prefix<1 or self.execute_prefix>self.chunk_horizon:raise ValueError('execute_prefix')
        if min(self.lengths)<1 or self.label_budget<min(self.lengths):raise ValueError('budget/lengths')
        if self.demos<2 or self.updates_per_action<1:raise ValueError('training configuration')
        if not self.train_seeds or len(set(self.train_seeds))!=len(self.train_seeds):raise ValueError('training seeds')
        return self

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,(np.integer,np.floating)):return x.item()
        if isinstance(x,np.bool_):return bool(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=encode,allow_nan=False)+'\n',encoding='utf-8')

def rollout(model,seed,split,backend,config,disturbance='none',prefix=1,save=False):
    env=make_env(backend,config);obs=env.reset(seed,split,disturbance);history=History(obs)
    actions=[];observations=[obs.copy()];queued=[]
    while not env.done:
        if not queued:queued=list(model.chunk(history.array())[:prefix].copy())
        action=queued.pop(0);obs,r,te,tr,info=env.step(action);history.append(obs)
        actions.append(action.copy());observations.append(obs.copy())
    record=env.summary();record['teacher_actions_at_evaluation']=0
    if save:
        record['trajectory']={k:np.stack([row[k] for row in env.trace]) for k in env.trace[0]}
        record['trajectory']['action']=np.asarray(actions);record['trajectory']['observation']=np.asarray(observations)
    return record

def evaluate(model,seeds,split,backend,config,disturbance='none',prefix=1,budget=None):
    if split not in ('development','test','stress'):raise ValueError('invalid evaluation split')
    out=[rollout(model,int(s),split,backend,config,disturbance,prefix) for s in seeds]
    if budget is not None:
        if split!='development':raise ValueError('test results cannot be charged as training-development feedback')
        budget.development_steps+=sum(r['steps'] for r in out)
    return out

def development_utility(records):
    if not records or any(r['split']!='development' for r in records):
        raise ValueError('learning gains must use training-development rollouts, never test')
    return float(np.mean([float(r['success'])+.2*r['score'] for r in records]))

def candidate_pool(model,round_id,seed,cfg,ledger):
    candidates=[];task=TaskConfig(horizon=cfg.horizon)
    for ep in range(cfg.pool_episodes):
        env=make_env(cfg.backend,task)
        # Matched scene schedule across acquisition methods, not across policy states.
        sid=10000+seed*1000+round_id*cfg.pool_episodes+ep
        obs=env.reset(sid,'acquire',disturbance='impulse' if ep%2 else 'none');hist=History(obs)
        while not env.done:
            if env.steps%cfg.candidate_stride==0 and env.steps<cfg.horizon-min(cfg.lengths):
                candidates.append(Candidate(len(candidates),env.snapshot(),hist.copy(),model.disagreement(hist.array())))
            obs,_,_,_,_=env.step(model.chunk(hist.array())[0]);hist.append(obs);ledger.acquisition_steps+=1
    return candidates

def fit_recovery(prior,rehearsal,seed,method,cfg,out):
    if method not in cfg.methods:raise ValueError('method absent from config')
    model=copy.deepcopy(prior);ledger=Budget(cfg.label_budget);selector=GainSelector(seed=seed+89)
    selected=[];rounds=[];round_id=0;task=TaskConfig(horizon=cfg.horizon)
    start=time.perf_counter()
    # Frozen initial-policy acquisition bank: same student interaction budget across methods.
    pool=candidate_pool(model,0,seed,cfg,ledger)
    while ledger.remaining>=min(cfg.lengths):
        for candidate in pool:
            if not candidate.used:candidate.disagreement=model.disagreement(candidate.history.array())
        if not pool:raise RuntimeError('no nonterminal acquisition candidates')
        c,length,x,selection=selector.choose(pool,cfg.lengths,ledger.remaining,method)
        c.used=True
        env=make_env(cfg.backend,task);env.restore(c.state)
        segment=collect_segment(env,c.history,length,ledger,candidate_id=c.id)
        charged=len(segment.actions);steps=cfg.updates_per_action*charged
        # Same starting weights, optimization count, and rehearsal RNG.
        update_seed=seed*10000+round_id+101
        before=time.perf_counter()
        control,control_loss=update(model,rehearsal,selected,steps,update_seed,method!='no_sequence')
        trial,trial_loss=update(model,rehearsal,selected+[segment],steps,update_seed,method!='no_sequence')
        update_seconds=time.perf_counter()-before;ledger.optimizer_steps+=2*steps
        retention_log={}
        if cfg.retention_gate:
            from .retention import admit
            # Distinct from both gain-credit scenes and final test scenes.
            retention_seeds=range(60000,60000+cfg.development_episodes)
            def retention_eval(candidate):
                return evaluate(candidate,retention_seeds,'development',cfg.backend,task,
                    prefix=cfg.execute_prefix,budget=ledger)
            control,retention_log['control']=admit(model,control,retention_eval,development_utility)
            trial,retention_log['trial']=admit(model,trial,retention_eval,development_utility)
        dev_seeds=range(30000+round_id*cfg.development_episodes,30000+(round_id+1)*cfg.development_episodes)
        ctrl_eval=evaluate(control,dev_seeds,'development',cfg.backend,task,prefix=cfg.execute_prefix,budget=ledger)
        trial_eval=evaluate(trial,dev_seeds,'development',cfg.backend,task,prefix=cfg.execute_prefix,budget=ledger)
        gain=development_utility(trial_eval)-development_utility(ctrl_eval)
        target=selector.observe(x,gain,charged)
        rounds.append({'round':round_id,'candidate_scene':env.seed,'candidate_step':int(c.state['steps']),
            'length_requested':length,'labels_charged':charged,'features':x,'selection':selection,
            'control_development':ctrl_eval,'trial_development':trial_eval,'paired_gain':gain,
            'gain_per_action':target,'optimizer_steps_per_branch':steps,'update_seconds':update_seconds,
            'control_loss':control_loss,'trial_loss':trial_loss,
            'retention_gate':retention_log,'terminal_recovery_success':bool(env.success),'action_count':len(segment.actions)})
        # Keep every charged segment, including negative-gain data. Weight updates
        # may be attenuated by the separately logged retention gate.
        model=trial;selected.append(segment);round_id+=1
    ledger.audit();Path(out).mkdir(parents=True,exist_ok=True)
    model.save(Path(out)/'policy.pt');save_segments(Path(out)/'recovery_segments.npz',selected)
    ledger_data=asdict(ledger);ledger_data['wall_seconds']=time.perf_counter()-start
    write_json(Path(out)/'learning_gains.json',rounds);write_json(Path(out)/'budget.json',ledger_data)
    write_json(Path(out)/'selector.json',selector.state())
    return model,ledger_data

def summarize(records,seed_ids,methods,bootstrap=4000):
    result={'training_seeds':list(seed_ids),'methods':{},'comparisons':{}}
    for method in methods:
        group=[r for r in records if r['method']==method and r['split']=='test']
        seed_rates=[float(np.mean([r['success'] for r in group if r['training_seed']==s])) for s in seed_ids]
        result['methods'][method]={'successes':sum(r['success'] for r in group),'episodes':len(group),
            'mean_success_rate':float(np.mean(seed_rates)),'per_training_seed_rates':seed_rates,
            'mean_score':float(np.mean([r['score'] for r in group]))}
    if 'gain' in methods:
        for other in methods:
            if other=='gain':continue
            diffs=[]
            for s in seed_ids:
                a={r['seed']:r['success'] for r in records if r['method']=='gain' and r['training_seed']==s and r['split']=='test'}
                b={r['seed']:r['success'] for r in records if r['method']==other and r['training_seed']==s and r['split']=='test'}
                if a.keys()!=b.keys():raise ValueError('unpaired scene results')
                diffs.append([float(a[k])-float(b[k]) for k in sorted(a)])
            d=np.asarray(diffs);rng=np.random.default_rng(901);values=[]
            # Crossed resampling: training seed and scene IDs shared by methods.
            for _ in range(bootstrap):
                si=rng.integers(0,d.shape[0],size=d.shape[0]);ei=rng.integers(0,d.shape[1],size=d.shape[1])
                values.append(float(d[np.ix_(si,ei)].mean()))
            result['comparisons']['gain_minus_'+other]={'difference':float(d.mean()),
                 'paired_crossed_bootstrap_95':np.quantile(values,[.025,.975]).tolist(),
                 'qualification':'exploratory; few training seeds and multiplicity not adjusted'}
    return result

def run_study(cfg,out):
    cfg.validate();out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Nonempty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',asdict(cfg))
    write_json(out/'runtime.json',{'python':platform.python_version(),'torch':torch.__version__,
           'numpy':np.__version__,'backend':cfg.backend,'platform':platform.platform()})
    task=TaskConfig(horizon=cfg.horizon);demos,teacher_records=collect_demos(cfg.demos,backend=cfg.backend,config=task)
    # The episode split is frozen before any training. Reserve 20% for diagnostic loss.
    nval=max(1,len(demos)//5);train=demos[:-nval];heldout=demos[-nval:]
    save_segments(out/'demos.npz',demos);write_json(out/'teacher_records.json',teacher_records)
    write_json(out/'data_split.json',{'train_episode_ids':[d.episode_id for d in train],
        'prior_validation_episode_ids':[d.episode_id for d in heldout],
        'development_range_start':30000,'test_range_start':171000,'stress_range_start':181000,'retention_range_start':60000})
    all_records=[]
    from .policy import examples
    for seed in cfg.train_seeds:
        print(f'train_seed={seed}: prior',flush=True)
        prior=initialize(train,seed,cfg.initial_steps,horizon=cfg.chunk_horizon)
        folder=out/f'seed_{seed}';folder.mkdir();prior.save(folder/'prior.pt')
        with torch.no_grad():
            ex=examples(heldout,prior.horizon);a,_=prior(torch.from_numpy(ex['x']))
            mse=float(((a.mean(0)-torch.from_numpy(ex['y'])).square()*torch.from_numpy(ex['mask'])[...,None]).sum()/max(1,4*ex['mask'].sum()))
        write_json(folder/'prior_validation.json',{'masked_action_mse':mse,'used_for_model_selection':False})
        models={'bc':prior}
        for method in cfg.methods:
            print(f'train_seed={seed}: {method}',flush=True)
            m,ledger=fit_recovery(prior,train,seed,method,cfg,folder/method);models[method]=m
        for method,model in models.items():
            for split,base,dist in [('test',171000,'none'),('stress',181000,'low_friction')]:
                rows=evaluate(model,range(base,base+cfg.test_episodes),split,cfg.backend,task,dist,cfg.execute_prefix)
                for row in rows:row.update({'method':method,'training_seed':seed})
                all_records.extend(rows)
        write_json(out/'episodes.json',all_records)
    summary=summarize(all_records,cfg.train_seeds,['bc',*cfg.methods]);write_json(out/'summary.json',summary)
    audit_study(out);return summary

def audit_study(out):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());rows=json.loads((out/'episodes.json').read_text())
    expected=len(cfg['train_seeds'])*(1+len(cfg['methods']))*cfg['test_episodes']*2
    assert len(rows)==expected,(len(rows),expected)
    identities={(r['training_seed'],r['method'],r['split'],r['seed']) for r in rows};assert len(identities)==len(rows)
    for r in rows:assert r['teacher_actions_at_evaluation']==0 and r['backend']==cfg['backend']
    gains=0
    for s in cfg['train_seeds']:
        for method in cfg['methods']:
            p=out/f'seed_{s}'/method
            budget=json.loads((p/'budget.json').read_text());events=json.loads((p/'learning_gains.json').read_text())
            assert sum(e['labels_charged'] for e in events)==budget['expert_actions']<=cfg['label_budget']
            assert budget['optimizer_steps']==2*cfg['updates_per_action']*budget['expert_actions']
            for e in events:
                delta=development_utility(e['trial_development'])-development_utility(e['control_development'])
                assert abs(delta-e['paired_gain'])<1e-10
                assert abs(delta/e['labels_charged']-e['gain_per_action'])<1e-10
                assert [r['seed'] for r in e['trial_development']]==[r['seed'] for r in e['control_development']]
                for branch in e.get('retention_gate',{}).values():
                    assert branch[-1]['accepted']
                    for check in branch:
                        baseline=check.get('baseline_records')
                        if baseline is None:continue  # archived diagnostic logs
                        assert abs(development_utility(baseline)-check['baseline_utility'])<1e-10
                        assert abs(development_utility(check['records'])-check['utility'])<1e-10
                        assert not ({r['seed'] for r in baseline}&{r['seed'] for r in e['trial_development']})
                        if check['accepted'] and check['weight']>0:
                            assert check['utility']>=check['baseline_utility']
                            assert sum(r['success'] for r in check['records'])>=sum(r['success'] for r in baseline)
            gains+=len(events)
    audit={'passed':True,'autonomous_episodes':len(rows),'paired_learning_gain_records':gains,
        'checks':['unique episode keys','no evaluation expert actions','expert-action budget caps',
                  'optimizer updates per charged expert action','development-only gains','gain recomputation','matched backend']}
    write_json(out/'AUDIT.json',audit);return audit
