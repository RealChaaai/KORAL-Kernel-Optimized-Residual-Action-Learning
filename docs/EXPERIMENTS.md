# Experiments

Backend: `contact_proxy`. Training seeds: `[0, 1, 2]`. Horizon: 130 control steps (0.04 s per step).

All policies share the same robot servo, contact dynamics, initial states, control limits and terminal-success rule. A successful episode must include a lift, stable placement within 27 mm in the table plane, an open withdrawn gripper, low object speed, and four consecutive stable steps.

The neural BC checkpoint for each training seed is shared unchanged by all KORAL variants. Only the 24 designated training demonstrations enter the memory bank; six complete episodes are held out. The reference scene IDs provide the fixed regression benchmark. Confirmation scene IDs were frozen with the weights before their first evaluation. Every inference call uses observation history only.

## Fixed reference benchmark

| Method | Normal success | Normal rate | Low-friction success | Low-friction rate |
|---|---:|---:|---:|---:|
| Behavior cloning | 54/60 | 90.00% | 47/60 | 78.33% |
| Random segment fine-tuning | 47/60 | 78.33% | 38/60 | 63.33% |
| Disagreement segment fine-tuning | 53/60 | 88.33% | 42/60 | 70.00% |
| Gain-guided fine-tuning | 48/60 | 80.00% | 37/60 | 61.67% |
| Fine-tuning without temporal loss | 52/60 | 86.67% | 40/60 | 66.67% |
| KORAL | 58/60 | 96.67% | 58/60 | 96.67% |
| Local residual / initial demonstrations only | 59/60 | 98.33% | 58/60 | 96.67% |
| Constant local residual | 55/60 | 91.67% | 53/60 | 88.33% |
| Residual without support gate | 59/60 | 98.33% | 59/60 | 98.33% |
| Kernel action retrieval | 5/60 | 8.33% | 23/60 | 38.33% |
| Local-action blend | 53/60 | 88.33% | 51/60 | 85.00% |

| Training seed | BC normal | KORAL normal | BC low friction | KORAL low friction |
|---|---:|---:|---:|---:|
| 0 | 20/20 | 20/20 | 19/20 | 20/20 |
| 1 | 17/20 | 18/20 | 16/20 | 18/20 |
| 2 | 17/20 | 20/20 | 12/20 | 20/20 |

### Paired differences

| Condition | KORAL minus comparator | Difference (pp) | Crossed bootstrap 95% interval (pp) | Wins / losses |
|---|---|---:|---:|---:|
| test | Behavior cloning | +6.67 | [-3.33, +21.67] | 5 / 1 |
| test | Random segment fine-tuning | +18.33 | [+3.33, +36.67] | 12 / 1 |
| test | Disagreement segment fine-tuning | +8.33 | [-1.67, +21.67] | 6 / 1 |
| test | Gain-guided fine-tuning | +16.67 | [+3.33, +35.00] | 11 / 1 |
| test | Fine-tuning without temporal loss | +10.00 | [-1.67, +23.33] | 7 / 1 |
| test | Local residual / initial demonstrations only | -1.67 | [-10.00, +0.00] | 0 / 1 |
| test | Constant local residual | +5.00 | [-5.00, +18.33] | 4 / 1 |
| test | Residual without support gate | -1.67 | [-10.00, +0.00] | 0 / 1 |
| test | Kernel action retrieval | +88.33 | [+71.67, +100.00] | 53 / 0 |
| test | Local-action blend | +8.33 | [-5.00, +23.33] | 6 / 1 |
| stress | Behavior cloning | +18.33 | [+0.00, +45.00] | 12 / 1 |
| stress | Random segment fine-tuning | +33.33 | [+13.33, +55.00] | 21 / 1 |
| stress | Disagreement segment fine-tuning | +26.67 | [+6.67, +48.33] | 17 / 1 |
| stress | Gain-guided fine-tuning | +35.00 | [+15.00, +55.00] | 22 / 1 |
| stress | Fine-tuning without temporal loss | +30.00 | [+10.00, +51.67] | 19 / 1 |
| stress | Local residual / initial demonstrations only | +0.00 | [-6.67, +6.67] | 1 / 1 |
| stress | Constant local residual | +8.33 | [-1.67, +21.67] | 6 / 1 |
| stress | Residual without support gate | -1.67 | [-10.00, +0.00] | 0 / 1 |
| stress | Kernel action retrieval | +58.33 | [+38.33, +78.33] | 35 / 0 |
| stress | Local-action blend | +11.67 | [+0.00, +25.00] | 7 / 0 |

## Independent confirmation scenes

| Method | Normal success | Normal rate | Low-friction success | Low-friction rate |
|---|---:|---:|---:|---:|
| Behavior cloning | 154/180 | 85.56% | 163/180 | 90.56% |
| KORAL | 172/180 | 95.56% | 178/180 | 98.89% |
| Local residual / initial demonstrations only | 172/180 | 95.56% | 177/180 | 98.33% |
| Kernel action retrieval | 31/180 | 17.22% | 101/180 | 56.11% |
| Local-action blend | 162/180 | 90.00% | 172/180 | 95.56% |

| Training seed | BC normal | KORAL normal | BC low friction | KORAL low friction |
|---|---:|---:|---:|---:|
| 0 | 53/60 | 58/60 | 57/60 | 59/60 |
| 1 | 51/60 | 54/60 | 55/60 | 59/60 |
| 2 | 50/60 | 60/60 | 51/60 | 60/60 |

### Paired differences

| Condition | KORAL minus comparator | Difference (pp) | Crossed bootstrap 95% interval (pp) | Wins / losses |
|---|---|---:|---:|---:|
| test | Behavior cloning | +10.00 | [+2.22, +20.00] | 19 / 1 |
| test | Local residual / initial demonstrations only | +0.00 | [-6.11, +5.56] | 3 / 3 |
| test | Kernel action retrieval | +78.33 | [+68.33, +87.78] | 141 / 0 |
| test | Local-action blend | +5.56 | [-2.78, +15.00] | 15 / 5 |
| stress | Behavior cloning | +8.33 | [+1.67, +17.22] | 15 / 0 |
| stress | Local residual / initial demonstrations only | +0.56 | [+0.00, +3.33] | 1 / 0 |
| stress | Kernel action retrieval | +42.78 | [+30.54, +55.00] | 77 / 0 |
| stress | Local-action blend | +3.33 | [+0.00, +7.78] | 6 / 0 |

## Estimation protocol

Intervals resample training seeds and paired scene IDs as crossed factors (6,000 draws). Methods always use the same resampled indices. The three training seeds are reported separately above. These are pointwise intervals; comparisons are not multiplicity-adjusted. The reference and confirmation banks remain separate in all tables. Normal and low-friction conditions have distinct scene ID ranges; method contrasts are paired within each condition.

## Training costs

Every recovery method has a 64-expert-action cap per training seed. A queried action is charged even on a failed recovery. A segment can terminate early, leaving unused budget when fewer than 16 actions remain. Development evaluation, source rollouts and optimization are counted separately. BC and the initial-memory ablation purchase no additional actions.

| Method | Expert actions by seed | Acquisition steps by seed | Development steps by seed | Accepted local segments |
|---|---|---|---|---|
| Random segment fine-tuning | 64, 64, 55 | 156, 320, 161 | 7671, 8240, 8548 | —, —, — |
| Disagreement segment fine-tuning | 59, 64, 55 | 156, 320, 161 | 6947, 8547, 6900 | —, —, — |
| Gain-guided fine-tuning | 59, 57, 55 | 156, 320, 161 | 8040, 8218, 8337 | —, —, — |
| Fine-tuning without temporal loss | 59, 64, 55 | 156, 320, 161 | 7965, 8203, 8348 | —, —, — |
| KORAL | 59, 64, 55 | 156, 320, 161 | 11132, 11527, 10892 | 0, 2, 3 |

KORAL performs no network-gradient updates after the BC prior is frozen. It stores demonstration transitions and solves a small query-local ridge system during action inference. The retrieval and local-regression ablations share the same accepted data bank. Their memory costs are therefore different from the compressed neural BC model.

## Component definitions

`no_recovery` uses only the initial demonstration bank. `no_affine` replaces the local linear fit with a weighted constant residual. `no_support` removes the compact support gate. `retrieval` directly averages neighboring recorded actions. `local_action` fits absolute actions and blends them with BC, instead of fitting the errors of BC. The temporal-loss baseline removes only adjacent-action-difference supervision.

## Policy inference profile

Single-thread CPU; same held-out demonstration histories for both policies. Thirty warmup queries per model are excluded. The measured region is the policy call; physics integration and file logging are outside the timed region.

| Method | Timed queries | Median (ms) | p95 (ms) | Stored local transitions by seed |
|---|---:|---:|---:|---|
| Behavior cloning | 900 | 0.191 | 0.363 | 0, 0, 0 |
| KORAL | 900 | 0.572 | 0.823 | 1376, 1408, 1415 |

Per-query timings and checkpoint/data-bank byte counts are in `INFERENCE_PROFILE.json`.

## Stored evidence

`FROZEN_PROTOCOL.json` records model hashes, evaluation code hashes and scene ranges. `episodes.json` points to each recorded trajectory and its checksum. `learning_gains.json` stores paired development evaluations and memory-admission decisions. `AUDIT.json` recomputes task success from the recorded object and gripper states. `REPLAY_VERIFICATION.json` records independent checkpoint replays.
