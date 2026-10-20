# Validation record

## Executed checks

| Check | Result | Record |
|---|---|---|
| CPU / algorithm / contact tests | 99 passed | [pytest output](validation/pytest.txt) |
| Optional MuJoCo execution tests | 3 skipped; runtime not installed in this validation environment | [pytest output](validation/pytest.txt) |
| Frozen evaluation records | 3,120 episodes audited | [audit](../results/reference/AUDIT.json) |
| Independent policy replays | 960 BC/KORAL episodes regenerated and matched | [replay verification](../results/reference/REPLAY_VERIFICATION.json) |
| Demonstration regeneration | All 30 episodes matched exactly | [retraining verification](../results/reference/RETRAIN_VERIFICATION.json) |
| Neural prior retraining | All 3 initial policy weight digests matched exactly | [retraining verification](../results/reference/RETRAIN_VERIFICATION.json) |
| Complete small workflow | Collection, training, all methods, evaluation and audit executed | [workflow record](validation/workflow_smoke.json) |
| Isolated wheel installation | Imported installed code, loaded a local policy and executed an environment action | [wheel check](validation/wheel_install.json) |
| Offline HTML replay | JavaScript syntax and initial canvas draw passed | [replay check](validation/replay_html.json) |

The small workflow uses an intentionally short optimizer budget to check interfaces and output generation. Its outcomes are not pooled with the reference study.

## Recorded invariants

BC parameters and normalization buffers have identical hashes before and after local adaptation. Every local memory row comes from a designated training demonstration or charged recovery segment. Query and retention evaluations use separate development scene ranges. Terminal tests and confirmation scenes add no training labels.

All policies use a 130-step horizon, a 0.04-second control period, and the same contact and termination functions. The audit reconstructs success from lift, final object position and velocity, gripper opening, withdrawal distance and the consecutive-stability condition. Stored metric flags are checked against the recorded states.

## Re-execution

```bash
python -m pytest -q
python -m koral audit --results results/reference
python -m koral verify --results results/reference --methods bc koral --banks reference confirmation
```

The final distribution includes `SHA256SUMS.txt`. [Runtime versions](validation/runtime.json) record the exact CPU environment. Individual baseline and KORAL source/checkpoint hashes are stored in [the frozen protocol](../results/reference/FROZEN_PROTOCOL.json).
