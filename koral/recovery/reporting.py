"""Generate all benchmark tables from frozen episode, trajectory and cost logs."""
from pathlib import Path
import json
import numpy as np
from .experiment import write_json

NAMES={'bc':'Behavior cloning','random':'Random segment fine-tuning',
       'uncertainty':'Disagreement segment fine-tuning','gain':'Gain-guided fine-tuning',
       'no_sequence':'Fine-tuning without temporal loss','koral':'KORAL',
       'no_recovery':'Local residual / initial demonstrations only',
       'no_affine':'Constant local residual','no_support':'Residual without support gate',
       'retrieval':'Kernel action retrieval','local_action':'Local-action blend'}


def comparison(rows,methods,seeds,bank,split,repeats=6000):
    estimates={};keys={}
    for m in methods:
        r=[x for x in rows if x['method']==m and x['bank']==bank and x['split']==split]
        if not r:continue
        keys[m]={(x['training_seed'],x['seed']):int(x['success']) for x in r}
        estimates[m]={'successes':sum(x['success'] for x in r),'episodes':len(r),
            'success_rate':float(np.mean([x['success'] for x in r])),
            'per_training_seed':{str(s):{'successes':sum(x['success'] for x in r if x['training_seed']==s),
                'episodes':sum(x['training_seed']==s for x in r)} for s in seeds},
            'mean_steps_all_episodes':float(np.mean([x['steps'] for x in r])),
            'mean_success_seconds':float(np.mean([x['steps']*.04 for x in r if x['success']])) if any(x['success'] for x in r) else None}
    contrasts={}
    if 'koral' in keys:
        for other in keys:
            if other=='koral':continue
            if set(keys['koral'])!=set(keys[other]):raise ValueError('unpaired methods')
            scenes=sorted({sid for _,sid in keys['koral']})
            d=np.array([[keys['koral'][s,sid]-keys[other][s,sid] for sid in scenes] for s in seeds])
            rng=np.random.default_rng(20261020);draws=np.empty(repeats)
            for i in range(repeats):
                si=rng.integers(len(seeds),size=len(seeds));ei=rng.integers(len(scenes),size=len(scenes))
                draws[i]=d[np.ix_(si,ei)].mean()
            contrasts[other]={'difference':float(d.mean()),
                'crossed_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),
                'paired_wins':int((d==1).sum()),'paired_losses':int((d==-1).sum()),
                'per_seed_difference':d.mean(1).tolist()}
    return {'methods':estimates,'koral_minus':contrasts}


def make_report(results,out):
    root=Path(results);cfg=json.loads((root/'config.json').read_text())
    rows=json.loads((root/'episodes.json').read_text());protocol=json.loads((root/'FROZEN_PROTOCOL.json').read_text())
    summary={};methods=protocol['methods'];seeds=protocol['training_seeds']
    for bank in ('reference','confirmation'):
        summary[bank]={}
        for split in ('test','stress'):
            summary[bank][split]=comparison(rows,methods,seeds,bank,split)
    write_json(root/'summary.json',summary)
    lines=['# Experiments','',
       f"Backend: `{cfg['backend']}`. Training seeds: `{seeds}`. Horizon: {cfg['horizon']} control steps (0.04 s per step).",'',
       'All policies share the same robot servo, contact dynamics, initial states, control limits and terminal-success rule. '
       'A successful episode must include a lift, stable placement within 27 mm in the table plane, '
       'an open withdrawn gripper, low object speed, and four consecutive stable steps.', '',
       'The neural BC checkpoint for each training seed is shared unchanged by all KORAL variants. '
       'Only the 24 designated training demonstrations enter the memory bank; six complete episodes are held out. '
       'The reference scene IDs provide the fixed regression benchmark. Confirmation scene IDs were frozen with '
       'the weights before their first evaluation. Every inference call uses observation history only.', '']
    for bank,title in [('reference','Fixed reference benchmark'),('confirmation','Independent confirmation scenes')]:
        lines += ['## '+title,'','| Method | Normal success | Normal rate | Low-friction success | Low-friction rate |',
                  '|---|---:|---:|---:|---:|']
        for m in methods:
            a=summary[bank]['test']['methods'].get(m);b=summary[bank]['stress']['methods'].get(m)
            if not a:continue
            lines.append(f"| {NAMES[m]} | {a['successes']}/{a['episodes']} | {100*a['success_rate']:.2f}% | "
                         f"{b['successes']}/{b['episodes']} | {100*b['success_rate']:.2f}% |")
        lines += ['','| Training seed | BC normal | KORAL normal | BC low friction | KORAL low friction |',
                  '|---|---:|---:|---:|---:|']
        for seed in seeds:
            counts=[]
            for split,m in [('test','bc'),('test','koral'),('stress','bc'),('stress','koral')]:
                x=summary[bank][split]['methods'][m]['per_training_seed'][str(seed)]
                counts.append(f"{x['successes']}/{x['episodes']}")
            lines.append('| '+str(seed)+' | '+' | '.join(counts)+' |')
        lines += ['','### Paired differences','',
            '| Condition | KORAL minus comparator | Difference (pp) | Crossed bootstrap 95% interval (pp) | Wins / losses |',
            '|---|---|---:|---:|---:|']
        for split in ('test','stress'):
            for other,v in summary[bank][split]['koral_minus'].items():
                lo,hi=np.array(v['crossed_bootstrap_95'])*100
                lines.append(f"| {split} | {NAMES[other]} | {100*v['difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                             f"{v['paired_wins']} / {v['paired_losses']} |")
        lines.append('')
    lines += ['## Estimation protocol','',
        'Intervals resample training seeds and paired scene IDs as crossed factors (6,000 draws). '
        'Methods always use the same resampled indices. The three training seeds are reported separately above. '
        'These are pointwise intervals; comparisons are not multiplicity-adjusted. The reference and confirmation '
        'banks remain separate in all tables. Normal and low-friction conditions have distinct scene ID ranges; method contrasts are paired within each condition.','',
        '## Training costs','',
        f"Every recovery method has a {cfg['label_budget']}-expert-action cap per training seed. "
        'A queried action is charged even on a failed recovery. A segment can terminate early, leaving unused '
        'budget when fewer than 16 actions remain. Development evaluation, source rollouts and optimization '
        'are counted separately. BC and the initial-memory ablation purchase no additional actions.','',
        '| Method | Expert actions by seed | Acquisition steps by seed | Development steps by seed | Accepted local segments |',
        '|---|---|---|---|---|']
    for m in ['random','uncertainty','gain','no_sequence','koral']:
        budgets=[json.loads((root/f'seed_{s}'/m/'budget.json').read_text()) for s in seeds]
        fields=['expert_actions','acquisition_steps','development_steps']
        columns=[', '.join(str(b[k]) for b in budgets) for k in fields]
        accepted=', '.join(str(b.get('accepted_segments','—')) for b in budgets)
        lines.append('| '+NAMES[m]+' | '+' | '.join(columns)+' | '+accepted+' |')
    lines += ['','KORAL performs no network-gradient updates after the BC prior is frozen. '
        'It stores demonstration transitions and solves a small query-local ridge system during action inference. '
        'The retrieval and local-regression ablations share the same accepted data bank. '
        'Their memory costs are therefore different from the compressed neural BC model.','',
        '## Component definitions','',
        '`no_recovery` uses only the initial demonstration bank. `no_affine` replaces the local linear fit '
        'with a weighted constant residual. `no_support` removes the compact support gate. '
        '`retrieval` directly averages neighboring recorded actions. `local_action` fits absolute actions '
        'and blends them with BC, instead of fitting the errors of BC. '
        'The temporal-loss baseline removes only adjacent-action-difference supervision.','',
        '## Stored evidence','',
        '`FROZEN_PROTOCOL.json` records model hashes, evaluation code hashes and scene ranges. '
        '`episodes.json` points to each recorded trajectory and its checksum. `learning_gains.json` stores '
        'paired development evaluations and memory-admission decisions. `AUDIT.json` recomputes task success '
        'from the recorded object and gripper states. `REPLAY_VERIFICATION.json` records independent checkpoint replays.','']
    profile=root/'INFERENCE_PROFILE.json'
    if profile.exists():
        measured=json.loads(profile.read_text())['records']
        extra=['## Policy inference profile','',
            'Single-thread CPU; same held-out demonstration histories for both policies. Thirty warmup queries per model are excluded. The measured region is the policy call; physics integration and file logging are outside the timed region.','',
            '| Method | Timed queries | Median (ms) | p95 (ms) | Stored local transitions by seed |',
            '|---|---:|---:|---:|---|']
        for method in ['bc','koral']:
            group=[r for r in measured if r['method']==method]
            lat=np.concatenate([r['latency_ms'] for r in group])
            memory=', '.join(str(r.get('memory_rows',0)) for r in group)
            extra.append(f'| {NAMES[method]} | {len(lat)} | {np.median(lat):.3f} | {np.quantile(lat,.95):.3f} | {memory} |')
        extra += ['', 'Per-query timings and checkpoint/data-bank byte counts are in `INFERENCE_PROFILE.json`.', '']
        at=lines.index('## Stored evidence')
        lines[at:at]=extra
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text('\n'.join(lines),encoding='utf-8')
    return summary
