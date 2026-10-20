"""Release audit: recorded physics, immutable priors, budgets and frozen models."""
from pathlib import Path
import hashlib,json
import numpy as np
from .environment import TaskConfig,make_env
from .local_policy import load_policy,LocalResidualPolicy
from .local_study import weight_digest,checkpoint_path,retention_accept
from .experiment import development_utility,write_json,rollout


def audit_results(results, *, verify_sources=True):
    root=Path(results);cfg=json.loads((root/'config.json').read_text())
    protocol=json.loads((root/'FROZEN_PROTOCOL.json').read_text())
    rows=json.loads((root/'episodes.json').read_text())
    methods=protocol['methods'];seeds=protocol['training_seeds']
    expected=len(seeds)*2*(len(methods)*cfg['test_episodes'] +
                          len(protocol['confirmation_methods'])*cfg['confirmation_episodes'])
    if len(rows)!=expected:raise ValueError(f'Expected {expected} evaluation episodes, found {len(rows)}')
    identities={(r['training_seed'],r['method'],r['bank'],r['split'],r['seed']) for r in rows}
    if len(identities)!=len(rows):raise ValueError('duplicate episode identities')
    for rel,digest in protocol['checkpoint_sha256'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Changed frozen checkpoint: {rel}')
    source=Path(__file__).resolve().parents[1]
    if verify_sources:
        for rel,digest in protocol['source_sha256'].items():
            if hashlib.sha256((source/rel).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Changed frozen evaluation code: {rel}')
    task=TaskConfig(horizon=cfg['horizon']);charged=0;credit_records=0
    for s in seeds:
        prior=load_policy(root/f'seed_{s}/prior.pt');model=load_policy(root/f'seed_{s}/koral/policy.pt')
        if not isinstance(model,LocalResidualPolicy):raise ValueError('KORAL checkpoint missing local model')
        if weight_digest(prior)!=weight_digest(model.prior):raise ValueError('base network was modified')
        if any(r['split'] not in ('train','acquire') for r in model.provenance):
            raise ValueError('held-out data in policy memory')
        p=root/f'seed_{s}/koral';events=json.loads((p/'learning_gains.json').read_text());b=json.loads((p/'budget.json').read_text())
        if sum(e['labels_charged'] for e in events)!=b['expert_actions'] or b['expert_actions']>cfg['label_budget']:
            raise ValueError('expert-action accounting failed')
        for e in events:
            c,t=e['control_development'],e['trial_development']
            if [(r['seed'],r['disturbance']) for r in c]!=[(r['seed'],r['disturbance']) for r in t]:
                raise ValueError('unpaired credit estimates')
            gain=development_utility(t)-development_utility(c)
            if not np.isclose(gain,e['paired_gain'],atol=1e-12) or not np.isclose(gain/e['labels_charged'],e['gain_per_action'],atol=1e-12):
                raise ValueError('learning-credit mismatch')
            if e['accepted']!=retention_accept(e['retention_control'],e['retention_trial']):
                raise ValueError('retention decision mismatch')
            if {r['seed'] for r in c}&{r['seed'] for r in e['retention_control']}:
                raise ValueError('credit/retention scene overlap')
            if e['bank_rows_after']-e['bank_rows_before'] != (e['labels_charged'] if e['accepted'] else 0):
                raise ValueError('unaccounted memory rows')
        charged+=b['expert_actions'];credit_records+=len(events)
    successes=0
    for row in rows:
        if row['teacher_actions_at_evaluation']!=0 or row['backend']!=cfg['backend']:
            raise ValueError('evaluation protocol mismatch')
        split=row['split'];bank=row['bank'];rkey='normal_seeds' if split=='test' else 'stress_seeds'
        lo,hi=protocol[bank][rkey]
        if not lo<=row['seed']<=hi:raise ValueError('episode outside frozen seed range')
        file=root/row['trajectory_file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=row['trajectory_sha256']:
            raise ValueError('modified trajectory')
        env=make_env(cfg['backend'],task);env.reset(row['seed'],split,row['disturbance'])
        with np.load(file,allow_pickle=False) as z:
            obj=z['object'];ee=z['ee'];vel=z['velocity'];opening=z['opening'];a=z['action']
            if len(a)!=row['steps'] or len(obj)!=len(a)+1 or len(z['observation'])!=len(obj):
                raise ValueError('trajectory length mismatch')
            if not all(np.isfinite(z[k]).all() for k in z.files):raise ValueError('nonfinite trajectory')
            if np.max(np.abs(a))>1.00001:raise ValueError('unbounded executed action')
            lifted=False;stable=0;terminal=False
            for i in range(1,len(obj)):
                lifted |= bool(obj[i,2]>env.table+task.object_radius+.035)
                resting=(np.linalg.norm(obj[i,:2]-env.goal[:2])<.027 and
                    abs(obj[i,2]-env.goal[2])<.007 and np.linalg.norm(vel[i])<.10 and
                    opening[i]>.03 and np.linalg.norm(ee[i]-obj[i])>.045 and lifted)
                stable=stable+1 if resting else 0
                terminal=stable>=4
            if terminal!=row['success'] or lifted!=row['lifted']:
                raise ValueError('success flag disagrees with recorded physics')
            if row['steps']>task.horizon:raise ValueError('extended task horizon')
            if np.max(z['contacts'])>row['peak_force']+1e-6:raise ValueError('force metric mismatch')
        successes+=int(row['success'])
    result={'passed':True,'evaluation_episodes':len(rows),'successful_episodes':successes,
        'paired_local_credit_records':credit_records,'charged_local_expert_actions':charged,
        'checks':['frozen code and checkpoint hashes','unique paired episode keys',
          'unchanged base neural weights','training-only memory provenance','exact expert-action accounting',
          'disjoint development credit and retention','zero test-time expert actions',
          'finite bounded actions','success recomputed from recorded physical states','same task horizon']}
    write_json(root/'AUDIT.json',result);return result


def verify_replay(results,methods=('bc','koral'),banks=('reference',)):
    """Independently regenerate trajectories from saved checkpoints and seeds."""
    root=Path(results);cfg=json.loads((root/'config.json').read_text())
    rows=json.loads((root/'episodes.json').read_text());cache={};n=0
    for row in rows:
        if row['method'] not in methods or row['bank'] not in banks:continue
        key=(row['training_seed'],row['method'])
        if key not in cache:cache[key]=load_policy(checkpoint_path(root,*key))
        new=rollout(cache[key],row['seed'],row['split'],cfg['backend'],TaskConfig(horizon=cfg['horizon']),row['disturbance'],save=True)
        if new['success']!=row['success'] or new['steps']!=row['steps']:
            raise ValueError('regenerated outcome mismatch')
        with np.load(root/row['trajectory_file'],allow_pickle=False) as old:
            for name,values in new['trajectory'].items():
                if not np.allclose(old[name],values,rtol=0,atol=1e-7):
                    raise ValueError(f'trajectory replay mismatch: {name}')
        n+=1
    result={'passed':True,'independently_regenerated_episodes':n,'methods':list(methods),'banks':list(banks)}
    write_json(root/'REPLAY_VERIFICATION.json',result);return result
