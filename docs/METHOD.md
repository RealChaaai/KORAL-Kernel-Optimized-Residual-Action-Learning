# Support-gated local residual learning

## 1. Policy decomposition

Let \(h_t=(o_{t-3},\ldots,o_t)\) denote four consecutive 32-dimensional observations. The behavior-cloning ensemble predicts an eight-action chunk. Its first action is the frozen prior \(b(h_t)\in[-1,1]^4\). The action consists of three Cartesian displacement commands and a gripper-opening command.

KORAL stores bounded training transitions \(\mathcal D=\{(h_i,a_i)\}\) and models the **errors of the prior**:

\[
r_i=a_i-b(h_i).
\]

The network parameters and normalization buffers are immutable after initialization. At runtime, a local model estimates a correction using nearby demonstration or accepted recovery transitions. Only the first action of the chunk is corrected; the next observation starts a new query.

```text
Observation history ──> frozen BC ───────────────────────────┐
        │                                                   │
        └─> contact-state features ─> local residual fit ─> support gate
                                                            │
                                            bounded action sum
                                                            │
                                         robot / object execution
```

## 2. Contact-state coordinates

The retrieval metric uses a 20-dimensional vector \(x=\phi(h)\). Every component is computed from the policy observation, without querying the environment or expert.

| Components | Dimension | Normalization |
|---|---:|---|
| Object relative to tool; goal relative to object | 6 | 0.06 m |
| Tool height above table | 1 | 0.12 m |
| Object velocity; tool velocity | 6 | Clip to ±0.3 m/s, divide by 0.3 m/s |
| Measured finger opening | 1 | 0.03 m |
| Two measured pad-contact signals | 2 | Square root of force / 15 N, clipped to [0,1] |
| Previous gripper command | 1 | Already normalized |
| Change in object/tool relative position across the history | 3 | 0.04 m |

The metric omits absolute world coordinates, joint identity and episode clock. The frozen neural prior retains its original complete observation. History differentiates a moving approach from a locally similar stopped or slipping state. `contact_features()` implements the metric and its shape/finite-value checks.

## 3. Query-centered ridge regression

For query \(x\), take the \(k=24\) nearest stored feature vectors using squared Euclidean distance \(d_i^2=\|x_i-x\|^2\). Ties follow stored-row order. Define

\[
\sigma_x^2=\max\left\{\operatorname{median}_{i\in\mathcal N_k(x)}d_i^2,\ 0.03\right\},
\qquad w_i=\exp\left(-\frac{d_i^2}{2\sigma_x^2}\right).
\]

A local affine residual is \(r_i\simeq c+B(x_i-x)\). With rows \(z_i^\top=[1,(x_i-x)^\top]\), its regularized fit is

\[
\widehat\Theta=(Z^\top WZ+\Lambda)^{-1}Z^\top WR,
\qquad
\Lambda=\operatorname{diag}(0.02,0.08,\ldots,0.08).
\]

The first row \(\widehat\delta=e_0^\top\widehat\Theta\) estimates the residual at the query. All diagonal entries of \(\Lambda\) are positive, so the system is positive definite even for repeated or rank-deficient samples. The implementation solves the system directly rather than explicitly computing its inverse.

This differs from averaging stored actions: the local model is trained on errors of the exact frozen prior and accounts for the geometric difference between the stored state and current observation.

## 4. Compact support and bounded correction

Let \(d_{\min}(x)=\min_i\|x_i-x\|\). The support gate is

\[
g(x)=\left[\max\left(0,1-\frac{d_{\min}(x)}{R}\right)\right]^2,
\qquad R=3.
\]

The action is

\[
a=\operatorname{clip}_{[-1,1]^4}
\left[b(h)+\eta g(x)\operatorname{clip}_{[-\Delta,\Delta]^4}(\widehat\delta)\right],
\qquad \eta=1,\quad\Delta=2.
\]

**Exact fallback.** At a fixed history with \(d_{\min}\geq R\), \(g=0\), hence \(a=b(h)\). An empty memory bank or zero correction strength has the same result.

**Bounded action change.** Coordinate-wise projection onto the action interval is nonexpansive, and \(b\) is already inside that interval. Therefore

\[
|a_j-b_j|\leq\eta g(x)\Delta.
\]

**Unchanged local neighborhood.** If an appended recovery segment contributes no row to a query's nearest-neighbor set and leaves its nearest distance unchanged, the local solve and action at that query remain unchanged. Neural parameters remain unchanged for every query regardless of which data are appended.

`test_local_policy.py` checks these properties, duplicate-state stability, serialization, provenance, finite outputs and teacher-free inference.

## 5. Purchased recovery segments

Initial data are 24 complete training demonstrations. Six additional demonstration episodes are held out. A frozen student rollout produces the acquisition candidate bank; all source-environment steps are counted.

A selected candidate is restored only during training. An expert executes up to 16 consecutive actions, with each queried action charged before execution. Termination or truncation ends the segment. The per-seed additional expert budget is 64 actions. Segment storage includes every transition, end flag and history window.

A proposal appends the purchased segment to the accepted memory. Its paired learning credit is

\[
\ell_i=\frac{U(\pi_{\mathcal D\cup\tau_i})-U(\pi_{\mathcal D})}{|\tau_i|},
\qquad U(\pi)=\mathbb E[\mathbf 1_{\mathrm{success}}+0.2\,\mathrm{progress}].
\]

Both policies use identical development scene IDs under nominal and low-friction conditions. The credit regressor receives only past measured credits and observable candidate features. Discounted ridge regression with an exploration bonus selects subsequent affordable candidates.

A separate retention bank determines whether to keep the memory extension. Its nominal and low-friction success counts must each be at least those of the current policy, and aggregate utility must not decrease. Rejected segments remain in the purchased-data log and cost ledger. No gradient update changes the frozen neural prior.

## 6. Complete inference and training costs

For \(N\) stored transitions and fixed feature dimension \(d=20\), the reference implementation performs an \(O(Nd)\) distance scan, a stable \(O(N\log N)\) sort and a 21-by-21 ridge solve. Both state-memory size and inference time are reported alongside success rates.

The ledger separates initial demonstrations, acquisition steps, purchased expert actions, development rollouts and network optimizer steps. Local-model reconstruction and development rollouts are included in measured adaptation wall time. The inference bank holds initial and accepted training data; final evaluation adds no labels.

## 7. Source mapping

| Component | Source |
|---|---|
| Action chunks and BC ensemble | `koral/recovery/policy.py` |
| Feature metric, local solve, support gate | `koral/recovery/local_policy.py` |
| Budgeted recovery and immutable-prior hashes | `koral/recovery/local_study.py` |
| Gain-based candidate selection | `koral/recovery/acquisition.py` |
| Contact task and action execution | `koral/recovery/environment.py`, `contact.py` |
| End-to-end study | `koral/recovery/workflow.py` |
| Trajectory and cost audit | `koral/recovery/verification.py` |

[Experimental protocol and results](EXPERIMENTS.md) · [Model specification](MODEL.md) · [Related work](RELATED_WORK.md)
