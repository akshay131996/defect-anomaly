# Run ledger

One row per experiment run. Appended by the worker, audited by the planner.
`verdict` is measured against the run's **own stated hypothesis** in HANDOFF.md §7 —
a factual call about the number, not a judgement about the project.

See HANDOFF.md §0 for the output contract and the audit checklist.

| run_id | geometry | img | mean AU-PRO@5% | mean I-AUROC | code hash | verdict |
|---|---|---|---|---|---|---|
| (pre-E1 baseline) | crop | 448 | 0.1306 | 0.663 | — | void — coordinate-frame bug, see §6 |
| E1-squash-448 | squash | 448 | **0.3006** | 0.718 | `5168721b` | supports |
