"""Support-gated local residual learning around an immutable action-chunk prior.

No environment, teacher, phase label, seed, or future observation is accessible
at inference. A query-local ridge solve fits *errors of the frozen prior*, not a
replacement policy. The compact support gate returns exactly to the prior away
from the data bank. Training and acquisition provenance travel with checkpoints.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import copy
import json
import numpy as np
import torch
from .environment import HISTORY, OBS_DIM
from .policy import ChunkPolicy
from .data import Segment

FEATURE_DIM = 20


def contact_features(history: np.ndarray) -> np.ndarray:
    """Dimensionless, history-aware features from the documented observation.

    Position/goal differences: 6; tool height: 1; measured velocities: 6;
    finger opening: 1; measured pad forces: 2; last gripper action: 1;
    four-frame relative-position change: 3. Global pose, joint identity and
    elapsed episode time are intentionally excluded from the retrieval metric.
    """
    h = np.asarray(history, dtype=np.float64)
    if h.shape[-2:] != (HISTORY, OBS_DIM) or not np.isfinite(h).all():
        raise ValueError(f'Expected finite (..., {HISTORY}, {OBS_DIM}) history')
    o = h[..., -1, :]
    return np.concatenate((o[..., :6] / .06, o[..., 6:7] / .12,
        np.clip(o[..., 7:13], -.3, .3) / .3, o[..., 27:28] / .03,
        np.sqrt(np.clip(o[..., 28:30], 0., 1.)), o[..., 30:31],
        (o[..., :3] - h[..., 0, :3]) / .04), axis=-1)


@dataclass(frozen=True)
class LocalConfig:
    neighbors: int = 24
    ridge: float = .02
    slope_penalty: float = 4.
    support_radius: float = 3.
    bandwidth_floor: float = .03
    strength: float = 1.
    max_residual: float = 2.
    regression: str = 'affine'
    target: str = 'residual'
    support_gate: bool = True

    def validate(self) -> 'LocalConfig':
        if not isinstance(self.neighbors, int) or self.neighbors < 1:
            raise ValueError('neighbors must be a positive integer')
        for key in ('ridge', 'slope_penalty', 'support_radius', 'bandwidth_floor', 'max_residual'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not np.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError('strength must be in [0, 1]')
        if self.regression not in ('affine', 'constant') or self.target not in ('residual', 'action'):
            raise ValueError('unsupported local estimator')
        return self


class LocalResidualPolicy:
    """Frozen neural action prior plus a compactly supported local correction.

    The action chunk API is retained. Only the first action is corrected; the
    deployment protocol recomputes a new chunk after every observation. Every
    stored row comes from a purchased or initial demonstration transition.
    """
    format_version = 3

    def __init__(self, prior: ChunkPolicy, histories: np.ndarray,
                 actions: np.ndarray, config: LocalConfig | None = None,
                 provenance: list[dict] | None = None):
        self.config = (config or LocalConfig()).validate()
        self.prior = copy.deepcopy(prior).eval()
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.horizon = self.prior.horizon
        self.histories = np.array(histories, dtype=np.float32, copy=True)
        self.actions = np.array(actions, dtype=np.float64, copy=True)
        n = len(self.histories)
        if self.histories.shape != (n, HISTORY, OBS_DIM) or self.actions.shape != (n, 4):
            raise ValueError('inconsistent memory-bank shapes')
        if not np.isfinite(self.actions).all() or np.max(np.abs(self.actions), initial=0) > 1.00001:
            raise ValueError('memory actions must be finite and bounded')
        self.features = contact_features(self.histories)
        self.provenance = copy.deepcopy(provenance or [])
        if self.provenance and sum(int(r['rows']) for r in self.provenance) != n:
            raise ValueError('provenance does not account for all memory rows')
        if any(r.get('split') not in ('train', 'acquire') for r in self.provenance):
            raise ValueError('held-out data cannot enter a local policy')
        values = []
        with torch.no_grad():
            for start in range(0, n, 256):
                x = torch.from_numpy(self.histories[start:start + 256])
                values.append(self.prior(x)[0].mean(0)[:, 0].cpu().numpy())
        self.residuals = self.actions - np.concatenate(values) if values else np.empty((0, 4))

    @classmethod
    def fit(cls, prior: ChunkPolicy, segments: Sequence[Segment],
            config: LocalConfig | None = None) -> 'LocalResidualPolicy':
        rows = list(segments)
        for seg in rows:
            seg.validate()
        histories = np.concatenate([s.histories for s in rows]) if rows else np.empty((0, HISTORY, OBS_DIM), np.float32)
        actions = np.concatenate([s.actions for s in rows]) if rows else np.empty((0, 4))
        provenance = [{'episode_id': int(s.episode_id), 'split': s.split,
                       'source': s.source, 'rows': len(s.actions)} for s in rows]
        return cls(prior, histories, actions, config, provenance)

    def extend(self, segments: Sequence[Segment]) -> 'LocalResidualPolicy':
        """Return a new memory, keeping both the source policy and prior intact."""
        rows = list(segments)
        for seg in rows:
            seg.validate()
        histories = np.concatenate([self.histories, *[s.histories for s in rows]])
        actions = np.concatenate([self.actions, *[s.actions for s in rows]])
        provenance = self.provenance + [{'episode_id': int(s.episode_id), 'split': s.split,
            'source': s.source, 'rows': len(s.actions)} for s in rows]
        return type(self)(self.prior, histories, actions, self.config, provenance)

    def with_config(self, config: LocalConfig) -> 'LocalResidualPolicy':
        return type(self)(self.prior, self.histories, self.actions, config, self.provenance)

    def _predict(self, history: np.ndarray) -> tuple[np.ndarray, dict]:
        base = self.prior.chunk(history)
        cfg = self.config
        if not len(self.histories) or cfg.strength == 0:
            return base, {'gate': 0., 'nearest_distance': None, 'neighbors': [], 'local_solves': 0}
        x = contact_features(history)
        squared = np.sum((self.features - x)**2, axis=1)
        k = min(cfg.neighbors, len(squared))
        # Stable ordering also defines deterministic behavior for duplicate states.
        indices = np.argsort(squared, kind='stable')[:k]
        ds = squared[indices]
        distance = float(np.sqrt(ds[0]))
        gate = max(0., 1. - distance / cfg.support_radius)**2 if cfg.support_gate else 1.
        diagnostics = {'gate': gate, 'nearest_distance': distance,
                       'neighbors': indices.tolist(), 'local_solves': 0}
        if gate == 0:
            return base, diagnostics
        bandwidth = max(float(np.median(ds)), cfg.bandwidth_floor)
        weights = np.exp(-ds / (2. * bandwidth))
        targets = self.residuals[indices] if cfg.target == 'residual' else self.actions[indices]
        if cfg.regression == 'affine':
            design = np.column_stack((np.ones(k), self.features[indices] - x))
            lhs = design.T @ (weights[:, None] * design)
            lhs += np.diag(np.r_[cfg.ridge, np.full(FEATURE_DIM, cfg.ridge * cfg.slope_penalty)])
            value = np.linalg.solve(lhs, design.T @ (weights[:, None] * targets))[0]
        else:
            value = (weights[:, None] * targets).sum(0) / (weights.sum() + cfg.ridge)
        if cfg.target == 'action':
            value = value - base[0]
        correction = np.clip(value, -cfg.max_residual, cfg.max_residual)
        action = np.clip(base[0].astype(np.float64) + cfg.strength * gate * correction, -1., 1.)
        diagnostics.update({'local_solves': 1, 'raw_correction': value.tolist(),
                            'applied_correction': (action - base[0]).tolist()})
        base[0] = action
        return base, diagnostics

    def chunk(self, history: np.ndarray) -> np.ndarray:
        return self._predict(history)[0]

    def explain(self, history: np.ndarray) -> dict:
        return self._predict(history)[1]

    def disagreement(self, history: np.ndarray) -> float:
        return self.prior.disagreement(history)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format': self.format_version, 'kind': 'support_gated_local_residual',
            'prior_spec': self.prior.spec, 'prior_state_dict': self.prior.state_dict(),
            'histories': torch.from_numpy(self.histories),
            'actions': torch.from_numpy(self.actions), 'local_config': asdict(self.config),
            'provenance_json': json.dumps(self.provenance, ensure_ascii=False)}, path)

    @classmethod
    def from_state(cls, state: dict) -> 'LocalResidualPolicy':
        if state.get('format') != cls.format_version or state.get('kind') != 'support_gated_local_residual':
            raise ValueError('unsupported local policy checkpoint')
        prior = ChunkPolicy(**state['prior_spec'])
        prior.load_state_dict(state['prior_state_dict'])
        return cls(prior, state['histories'].cpu().numpy(), state['actions'].cpu().numpy(),
                   LocalConfig(**state['local_config']), json.loads(state['provenance_json']))

    @classmethod
    def load(cls, path: str | Path) -> 'LocalResidualPolicy':
        return cls.from_state(torch.load(path, map_location='cpu', weights_only=True))


def load_policy(path: str | Path) -> ChunkPolicy | LocalResidualPolicy:
    """Load a safe-tensor/dict checkpoint; never unpickle arbitrary model objects."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') == LocalResidualPolicy.format_version:
        return LocalResidualPolicy.from_state(state)
    if state.get('format') == 2:
        model = ChunkPolicy(**state['spec'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        return model
    raise ValueError('Unsupported checkpoint format')
