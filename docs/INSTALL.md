# Installation and execution

## CPU reference environment

Python 3.10 or later is required. Python 3.11 is a convenient choice on Windows. Core packages are NumPy and PyTorch. Numba accelerates contact integration; Matplotlib and Pillow generate the recorded-state visualizations.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[fast,viz,test]"
.\.venv\Scripts\python.exe -m koral doctor
.\.venv\Scripts\python.exe -m pytest -q
```

Calling the virtual environment's interpreter directly avoids changing PowerShell execution policies.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[fast,viz,test]'
python -m koral doctor
python -m pytest -q
```

CLI runs set PyTorch to one CPU thread, matching the reference experiment. Numba compiles its contact kernel on the first execution. The exact measured environment is saved in `docs/validation/runtime.json`.

## Run a trained policy

```bash
python -m koral demo \
  --checkpoint results/reference/seed_2/koral/policy.pt \
  --backend contact_proxy \
  --seed 99011 --disturbance low_friction \
  --out runs/demo --gif
```

Outputs include `trajectory.npz`, `metrics.json`, `demo.gif`, and the final-frame PNG. The demonstration horizon is the same 130 control steps used by the reference benchmark.

## Inspect and reproduce results

```bash
python -m koral audit --results results/reference
python -m koral report --results results/reference --out runs/REPORT.md
python -m koral verify --results results/reference --methods bc koral --banks reference confirmation
```

The audit recomputes outcome conditions from the saved states and checks frozen model/code hashes. Verification runs the saved policies again and compares the regenerated trajectories.

## Full workflow

```bash
# Small execution check: collection, training, adaptation, evaluation and audit.
python -m koral reproduce --config configs/smoke.json --out runs/smoke

# Full reference and confirmation study, including all baselines and ablations.
python -m koral reproduce --config configs/local_recovery.json --out runs/study
```

The full workflow collects 30 demonstration episodes, partitions them by complete episode, trains three independent neural priors, fits all correction methods, freezes the model hashes and runs paired evaluation. Each output directory must be empty. Choose a new directory for another run.

## Separate joint-dynamics adapter

```bash
python -m pip install -e '.[physics,test]'
python -m koral doctor --backend mujoco
python -m pytest -m physics -q
```

MuJoCo is an optional backend. Its execution tests are reported separately in `docs/VALIDATION.md`. Requesting it without the dependency installed raises an explicit error.

The optional joint-dynamics job is available through the manual GitHub Actions workflow. Push and pull-request checks run the CPU reference suite.
