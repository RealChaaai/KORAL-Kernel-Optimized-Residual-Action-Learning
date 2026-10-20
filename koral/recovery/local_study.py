"""Budgeted local recovery, immutable-prior verification, and frozen evaluation."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from .acquisition import GainSelector
from .budget import Budget
from .data import collect_segment, save_segments, load_segments
from .environment import TaskConfig, make_env
from .experiment import (StudyConfig, candidate_pool, evaluate, development_utility,
                         write_json, rollout, fit_recovery)
from .local_policy import LocalConfig, LocalResidualPolicy, load_policy
from .policy import ChunkPolicy


def weight_digest(prior: ChunkPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(prior.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalStudyConfig:
    backend: str = 'contact_proxy'
    demos: int = 30
    initial_steps: int = 500
    label_budget: int = 64
    pool_episodes: int = 3
    candidate_stride: int = 6
    lengths: tuple = (16,)
    development_episodes: int = 6
    horizon: int = 130
    chunk_horizon: int = 8
    train_seeds: tuple = (0, 1, 2)
    test_episodes: int = 20
    confirmation_episodes: int = 60
    local: dict = field(default_factory=lambda: asdict(LocalConfig()))
    confirmation_methods: tuple = ('bc', 'koral', 'no_recovery', 'retrieval', 'local_action')
    algorithm: str = 'local_recovery'

    def validate(self):
        if self.backend not in ('contact_proxy', 'mujoco'):
            raise ValueError('unsupported backend')
        if min(self.demos, self.initial_steps, self.pool_episodes, self.candidate_stride,
               self.development_episodes, self.horizon, self.test_episodes) < 1:
            raise ValueError('positive study sizes required')
        if not self.lengths or min(self.lengths) < 1 or self.label_budget < min(self.lengths):
            raise ValueError('invalid expert budget')
        if self.confirmation_episodes < 1 or not self.train_seeds or len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError('invalid confirmation size or training seeds')
        LocalConfig(**self.local).validate()
        return self

    def legacy(self):
        return StudyConfig(backend=self.backend, demos=self.demos, initial_steps=self.initial_steps,
            label_budget=self.label_budget, updates_per_action=3, lengths=self.lengths,
            pool_episodes=self.pool_episodes, candidate_stride=self.candidate_stride,
            development_episodes=3, test_episodes=self.test_episodes, train_seeds=self.train_seeds,
            methods=('random', 'uncertainty', 'gain', 'no_sequence'), horizon=self.horizon,
            chunk_horizon=self.chunk_horizon)


def dev_bank(model, base, cfg, ledger):
    records = []
    for offset, disturbance in ((0, 'none'), (1000, 'low_friction')):
        records.extend(evaluate(model, range(base + offset, base + offset + cfg.development_episodes),
            'development', cfg.backend, TaskConfig(horizon=cfg.horizon), disturbance, budget=ledger))
    return records


def retention_accept(control, trial):
    """Paired developmental acceptance, separately for each disturbance condition."""
    if [(r['seed'], r['disturbance']) for r in control] != [(r['seed'], r['disturbance']) for r in trial]:
        raise ValueError('unpaired retention evaluation')
    for condition in ('none', 'low_friction'):
        a = [r for r in control if r['disturbance'] == condition]
        b = [r for r in trial if r['disturbance'] == condition]
        if not a or not b or any(r['split'] != 'development' for r in a+b):
            raise ValueError('retention must use development-only paired conditions')
        if sum(r['success'] for r in b) < sum(r['success'] for r in a):
            return False
    return development_utility(trial) >= development_utility(control)


def fit_local(prior, rehearsal, seed, cfg: LocalStudyConfig, out, mode='gain'):
    """Buy segments sequentially; accept only a supported local memory extension.

    The paired learning credit compares the *same frozen model and memory* with
    and without the purchased segment. No network update affects old skills.
    Paired credit and retention use disjoint development scene banks.
    """
    cfg.validate()
    if mode not in ('gain', 'random', 'uncertainty'):
        raise ValueError('unsupported acquisition mode')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = LocalResidualPolicy.fit(prior, rehearsal, LocalConfig(**cfg.local))
    initial_digest = weight_digest(prior)
    ledger = Budget(cfg.label_budget)
    selector = GainSelector(seed=seed + 89)
    pool = candidate_pool(prior, 0, seed, cfg.legacy(), ledger)
    purchased, logs = [], []
    start = time.perf_counter()
    while ledger.remaining >= min(cfg.lengths):
        available = [c for c in pool if not c.used]
        if not available:
            break
        c, length, x, choice = selector.choose(available, cfg.lengths, ledger.remaining, mode)
        c.used = True
        env = make_env(cfg.backend, TaskConfig(horizon=cfg.horizon))
        env.restore(c.state)
        segment = collect_segment(env, c.history, length, ledger, candidate_id=c.id)
        charged = len(segment.actions)
        trial = model.extend([segment])
        round_id = len(logs)
        credit_base = 34000 + round_id * cfg.development_episodes
        control_rows = dev_bank(model, credit_base, cfg, ledger)
        trial_rows = dev_bank(trial, credit_base, cfg, ledger)
        gain = development_utility(trial_rows) - development_utility(control_rows)
        gain_per_action = selector.observe(x, gain, charged)
        retain_control = dev_bank(model, 64000, cfg, ledger)
        retain_trial = dev_bank(trial, 64000, cfg, ledger)
        accepted = retention_accept(retain_control, retain_trial)
        before_rows = len(model.histories)
        if accepted:
            model = trial
        purchased.append(segment)
        logs.append({'round': round_id, 'candidate_id': c.id, 'candidate_scene': int(env.seed),
            'candidate_step': int(c.state['steps']), 'length_requested': length,
            'labels_charged': charged, 'features': x, 'selection': choice,
            'control_development': control_rows, 'trial_development': trial_rows,
            'paired_gain': gain, 'gain_per_action': gain_per_action,
            'retention_control': retain_control, 'retention_trial': retain_trial,
            'accepted': accepted, 'bank_rows_before': before_rows, 'bank_rows_after': len(model.histories),
            'prior_weight_sha256': weight_digest(model.prior), 'expert_segment_success': bool(env.success)})
        assert weight_digest(model.prior) == initial_digest
    model.save(out/'policy.pt')
    save_segments(out/'recovery_segments.npz', purchased)
    ledger.audit()
    costs = asdict(ledger)
    costs.update({'wall_seconds': time.perf_counter()-start, 'network_optimizer_steps': 0,
                  'accepted_segments': sum(r['accepted'] for r in logs),
                  'memory_rows': len(model.histories), 'prior_weight_sha256': initial_digest,
                  'development_rollouts': sum(len(r[k]) for r in logs for k in
                      ('control_development','trial_development','retention_control','retention_trial'))})
    write_json(out/'budget.json', costs)
    write_json(out/'learning_gains.json', logs)
    write_json(out/'selector.json', selector.state())
    return model, costs


def freeze_run(root: Path, cfg: LocalStudyConfig, methods: list[str]) -> dict:
    """Write the evaluation protocol and hashes before any terminal-test rollout."""
    from datetime import datetime, timezone
    root = Path(root)
    checkpoint_hashes = {}
    for seed in cfg.train_seeds:
        for method in methods:
            p = checkpoint_path(root, seed, method)
            checkpoint_hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    source = Path(__file__).resolve().parents[1]
    core_paths = ['robot.py', 'recovery/environment.py', 'recovery/contact.py',
        'recovery/policy.py', 'recovery/local_policy.py', 'recovery/local_study.py',
        'recovery/experiment.py', 'recovery/teacher.py', 'recovery/data.py',
        'recovery/acquisition.py', 'recovery/retention.py', 'recovery/budget.py']
    source_hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                     for name in core_paths}
    protocol = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_seeds': list(cfg.train_seeds), 'methods': methods,
        'confirmation_methods': list(cfg.confirmation_methods),
        'task_horizon': cfg.horizon, 'backend': cfg.backend,
        'reference': {'normal_seeds': [171000,171000+cfg.test_episodes-1],
                      'stress_seeds': [181000,181000+cfg.test_episodes-1]},
        'confirmation': {'normal_seeds': [271000,271000+cfg.confirmation_episodes-1],
                         'stress_seeds': [281000,281000+cfg.confirmation_episodes-1]},
        'stress_condition': 'low_friction', 'checkpoint_sha256': checkpoint_hashes,
        'source_sha256': source_hashes,
        'selection_data': 'development scene banks only; no terminal-test feedback',
        'terminal_success_rule': 'unchanged ContactEnv.step predicate',
        'execute_prefix': 1}
    write_json(root/'FROZEN_PROTOCOL.json', protocol)
    return protocol


def checkpoint_path(root, seed, method):
    return Path(root)/f'seed_{seed}'/('prior.pt' if method == 'bc' else f'{method}/policy.pt')


def evaluate_frozen(root: Path, cfg: LocalStudyConfig, methods: list[str], *, confirmations=True):
    """All methods share scene IDs, contact dynamics, actuators, and termination."""
    root = Path(root)
    protocol = json.loads((root/'FROZEN_PROTOCOL.json').read_text())
    rows = []
    task = TaskConfig(horizon=cfg.horizon)
    trajectory_dir = root/'trajectories'
    trajectory_dir.mkdir(exist_ok=True)
    banks = [('reference', 'test',171000,'none',cfg.test_episodes),
             ('reference', 'stress',181000,'low_friction',cfg.test_episodes)]
    if confirmations:
        banks += [('confirmation','test',271000,'none',cfg.confirmation_episodes),
                  ('confirmation','stress',281000,'low_friction',cfg.confirmation_episodes)]
    for training_seed in cfg.train_seeds:
        for method in methods:
            ckpt = checkpoint_path(root, training_seed, method)
            digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            if protocol['checkpoint_sha256'][str(ckpt.relative_to(root))] != digest:
                raise ValueError('checkpoint changed after evaluation freeze')
            model = load_policy(ckpt)
            for bank, split, base, disturbance, n in banks:
                if bank == 'confirmation' and method not in cfg.confirmation_methods:
                    continue
                start = time.perf_counter()
                current = []
                for sid in range(base,base+n):
                    record = rollout(model,sid,split,cfg.backend,task,disturbance,save=True)
                    trajectory = record.pop('trajectory')
                    file = trajectory_dir/f'{training_seed}_{method}_{bank}_{split}_{sid}.npz'
                    if file.exists():
                        raise FileExistsError(f'Refusing to overwrite {file}')
                    np.savez_compressed(file, **trajectory)
                    record.update({'training_seed':training_seed,'method':method,'bank':bank,
                        'checkpoint_sha256':digest, 'trajectory_file':str(file.relative_to(root)),
                        'trajectory_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
                    current.append(record)
                rows.extend(current)
                print(f'seed={training_seed} {method} {bank}/{split}: '
                      f'{sum(r["success"] for r in current)}/{n}; '
                      f'{time.perf_counter()-start:.2f}s',flush=True)
                write_json(root/'episodes.json',rows)
    return rows
