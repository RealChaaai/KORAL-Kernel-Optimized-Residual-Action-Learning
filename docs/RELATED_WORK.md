# Related work

KORAL combines an immutable action prior, query-local residual regression, compact support, and a measured-credit recovery data loop. The references below supply the relevant methodological context.

| Reference | Connection |
|---|---|
| Ross, Gordon and Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, 2011. [Paper](https://proceedings.mlr.press/v15/ross11a.html) | Interactive imitation on learner-induced states. |
| Hoque et al. *ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning*. CoRL, 2021. [Paper](https://arxiv.org/abs/2109.08273) | Explicit supervision budgets and intervention selection. |
| Johannink et al. *Residual Reinforcement Learning for Robot Control*. 2018. [Paper](https://arxiv.org/abs/1812.03201) | Residual corrections composed with an existing controller. |
| Pari et al. *The Surprising Effectiveness of Representation Learning for Visual Imitation*. [Paper](https://arxiv.org/abs/2112.01511) | Locally weighted action regression and nonparametric imitation. |
| Ankile et al. *Residual Off-Policy RL for Finetuning Behavior Cloning Policies*. 2025. [Paper](https://arxiv.org/abs/2509.19301) | Frozen BC policies with learned residual improvements. |

The reference benchmark includes state-based kernel action retrieval, absolute-action local regression, neural correction baselines, and ablations of the residual fit, support gate and additional recovery memory. These names refer to the implementations in this repository; the comparison protocol and data budgets are specified in [EXPERIMENTS.md](EXPERIMENTS.md).
