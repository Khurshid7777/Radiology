# D249: P-wave detector independent of the QRS — prototypes (not in the pipeline)

- `pchain_v1.py` — conducted P from the beat template (R−300…+550 ms) + non-conducted P on the residual.
- `pchain_v2.py` — full-cycle template; P-shaped waves inside it (locked P incl. blocked 2:1 P) + residual P.
- `pchain_v3.py` — P searched in diastole (T end … next QRS onset) per lead (II, III, aVF, I, V1, V2); best lead by
  regularity × coverage.
- `probe.py <out> <items.json> k/n <dir with pchain module>` — runs the pipeline and the detector per record.
- `evaluate.py <scratch dir> [parity]` — AV block II (Chapman 2AVB / MOBITZ1 / AVB called AF) vs AF (Chapman 400,
  PTB-XL) vs controls.

Licensed data (Chapman, PTB-XL) stay in `data/` (git-ignored); items lists are built from `tools/af_train/dump.py`.
Results and the next step: DECISIONS.md, D249.
