# KORAL

**Kernel-Optimized Residual Action Learning**

KORAL studies how to improve a robot manipulation policy while retaining the behavior learned from its initial demonstrations. A frozen neural action-chunk prior supplies nominal actions. A contact-state memory estimates locally affine action residuals; a compact support gate controls where those corrections are applied. Budgeted recovery segments are appended according to measured learning credit and a separate retention check.

## Research question

Can local, data-supported corrections improve autonomous manipulation without globally fine-tuning a small demonstration-trained policy?

## Implementation

The system includes a seven-joint arm model, dynamic object contact, demonstration generation, history-conditioned neural policies, local residual inference, budgeted recovery acquisition, immutable-prior checks, paired evaluations, and trace-based data audits. The deployed policy receives only current and historical observations.

## Measurement

The fixed reference study compares eleven implementations on three independent training seeds. A separate confirmation bank evaluates five predeclared methods on additional scene IDs. Both banks use the same task, controller limits and success conditions. Each saved episode links its checkpoint and physical-state trajectory by SHA-256.

[Method](docs/METHOD.md) · [Experiment results](docs/EXPERIMENTS.md) · [Execution model](docs/MODEL.md) · [Installation](docs/INSTALL.md)
