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
| E3-aspect-448 | aspect | 448 | **0.3225** | **0.724** | `07dcb26e` | supports |
| E4-evalside | | | | | | _pending_ |
| E5-inputres | | | | | | _pending_ |
| E6-density | | | | | | _pending_ |
| E7-fusion-val | | | | | | _pending_ |
| E8-triton-bank | | | | | | _pending_ |
