# Run ledger

One row per experiment run. Appended by the worker, audited by the planner.
`verdict` is measured against the run's **own stated hypothesis** in HANDOFF.md §7 —
a factual call about the number, not a judgement about the project.

See HANDOFF.md §0 for the output contract and the audit checklist.

| run_id | geometry | img | mean AU-PRO@5% | mean I-AUROC | code hash | verdict |
|---|---|---|---|---|---|---|
| (pre-E1 baseline) | crop | 448 | 0.1306 | 0.663 | — | void — coordinate-frame bug, see §6 |
| E1-squash-448 | squash | 448 | **0.3006** | 0.718 | `5168721b` | supports |
| E0-registration | crop vs squash | — | 0.019 vs 1.000 | — | `9d7536e3` | supports |
| E2-letterbox-448 | letterbox | 448 | 0.2792 | 0.670 | `bf3917d6` | refutes |
| E3-aspect-448 | aspect | 448 | 0.3225 | 0.724 | `07dcb26e` | ~~supports~~ **inconclusive** — region-set confound, see §7 E4a |
| E4a-regionset | native 77px | — | 1530 regs | — | `59257b20` | supports — bit-identical 1530 regions across all geometries |
| E1R-squash-448 | squash | 448 | 0.3139 | 0.697 | `59257b20` | re-scored fixed regions |
| E2R-letterbox-448 | letterbox | 448 | 0.2932 | 0.670 | `59257b20` | re-scored fixed regions |
| E3R-aspect-448 | aspect | 448 | **0.3429** | **0.724** | `59257b20` | refutes — aspect margin is +0.029 (>0.01 threshold), aspect definitively wins |
| E4-evalside-512 | aspect | 448 (eval 512) | 0.3444 | 0.724 | `5d43304b` | 1530/1530 active regions |
| E4-evalside-1024 | aspect | 448 (eval 1024) | 0.3444 | 0.724 | `5d43304b` | 1530/1530 active regions |
| E4-evalside-2048 | aspect | 448 (eval 2048) | 0.3442 | 0.724 | `5d43304b` | 1530/1530 active regions |
| E4-evalside-trend | aspect | 448 (sweep) | 0.3444 -> 0.3442 | 0.724 | `5d43304b` | **refutes** — flat across eval resolutions (range 0.0002); input resolution ceiling binds |
| E5-inputres | | | | | | _pending_ |
| E6-density | | | | | | _pending_ |
| E7-fusion-val | | | | | | _pending_ |
| E8-triton-bank | | | | | | _pending_ |
