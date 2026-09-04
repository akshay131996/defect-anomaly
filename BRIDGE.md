# BRIDGE — planner/auditor <-> worker channel

**This file is the mailbox. `HANDOFF.md` is the specification.** They are different things
and must not drift into each other: if a decision belongs to the project it goes in
HANDOFF, and the bridge entry that produced it says so and stops. Anything written only
here is assumed lost after archival.

**Roles.** The planner/auditor writes directives and audits results. The worker executes on
the pod and reports. Neither writes in the other's lane.

**Read this file and pull before every action**, at both ends. A directive you have not
read is still binding on nothing — but a stale read is how two agents run the same
experiment twice.

---

## Protocol

1. **Pull immediately before writing, push immediately after.** Never leave this file dirty
   across a long task. It is the one file both agents write, so it is the only one that
   will actually conflict.
2. **Append, never rewrite.** New entries go at the bottom of the log with the next `M-nn`
   id. Do not edit an existing entry, including your own — supersede it with a new one that
   references it. The history of what was believed when is the point.
3. **Stay in your lane.** The planner owns `## Active directive` and `## Audit verdicts`.
   The worker owns `## Worker status`. Either may append to `## Log` and `## Open questions`.
4. **On a git conflict, keep both sides.** Never resolve by discarding the other agent's
   entry; renumber yours and keep going.
5. **Status vocabulary, exactly these:** `ACK` (read, will do), `RUNNING`, `DONE`,
   `BLOCKED` (needs a decision — say what decision), `DECLINED` (will not do — say why).
   `DONE` means the artifacts are committed and pushed, not that the run finished.
6. **A directive is not done until its artifacts exist**: the run record, the full
   unfiltered log, and the ledger row. See HANDOFF §0 for the output contract.
7. **Report refutations exactly as promptly as confirmations.** Three hypotheses about the
   AU-PRO gap were killed and the fourth was correct; that only worked because the
   disappointing ones arrived intact and un-tuned.
8. **When the log passes ~40 entries**, move everything already resolved to
   `BRIDGE_ARCHIVE.md` and leave the open items. A mailbox nobody reads is worse than none.

---

## Active directive

*Planner writes. One directive at a time. Superseded directives move to the log.*

**D-03 — run E5a, then stop and report.** HANDOFF §7 E5a. No GPU, minutes.

Measure the native-pixel area of all 1,530 ground-truth regions (median, IQR, and the
fraction below 1x / 2x / 4x the local patch-cell area), and break E3R's AU-PRO@5% down by
region-size bucket: sub-cell, 1-4 cells, 4-16 cells, larger. The per-region PRO values
already exist inside `evaluate` — expose them rather than recomputing.

**Report before starting E5.** E5a exists to decide whether E5's hours are worth spending,
which it cannot do if E5 has already started.

**Registered prediction (planner, before the result):** a majority of regions are sub-cell
at `img = 448`, and the largest bucket already scores near the published 0.764. If instead
every size bucket scores uniformly mediocre, **resolution is not the answer, E5 as
specified should not run, and we need a new hypothesis.** Either outcome is a complete
result.

---

## Worker status

*Worker writes. Overwrite this block freely — it is current state, not history. Put the
narrative in the log.*

| field | value |
|---|---|
| current directive | D-03 |
| status | *awaiting worker* |
| started | — |
| artifacts | — |
| blockers | — |

---

## Audit verdicts

*Planner writes. One row per audited run. `accepted` means the invariants passed AND the
stated conclusion follows from the numbers — those are two different checks.*

| run | invariants | conclusion follows | verdict |
|---|---|---|---|
| E0-registration | pass | yes | accepted |
| E1-squash-448 | pass | yes | accepted |
| E2-letterbox-448 | pass | yes | accepted (refutes) |
| E3-aspect-448 | pass | **no** | **downgraded to inconclusive** — region-set confound |
| E4a + E1R/E2R/E3R | pass | yes | accepted; planner's own prediction refuted |
| E4-evalside-{512,1024,2048} | pass | yes | accepted (refutes) |

---

## Open questions

*Either may append. Remove only when actually answered, and say where the answer lives.*

- **OQ-1 — what killed E1?** It died after 5 of 8 scenarios with no traceback and a zero
  cgroup OOM counter. Best current candidate is a host-level OOM kill driven by another
  tenant (a `blender` process was resident at ~3.9 GB; it has since exited). Unresolved. If
  a long run dies silently again, capture `dmesg`, `/sys/fs/cgroup/memory.events` and peak
  RSS **before** re-running.
- **OQ-2 — E0 does not cover `aspect`**, which is now the winning geometry. The
  implementation was hand-checked and is correct, but hand-checking is not a test. Add it.
- **OQ-3 — what closes the remaining 2.2x?** Evaluation protocol is closed (E4 refuted it).
  The live hypothesis is the patch grid. Note that even 1024 input leaves cells at ~308
  native px against a 77 px floor, so resolution alone is not obviously sufficient. If E5
  disappoints, this is the question with no candidate behind it.

---

## Log

*Append-only. Newest at the bottom.*

### M-01 — planner — directive D-01: E0 then E2
Superseded. E0 and E2 both completed; see audit verdicts.

### M-02 — planner — directive D-02: E4, plan approved with modifications
Approved the worker's E4 plan with four modifications (M1 retire the obsolete
`MIN_REGION_PX` arm, M2 assert `n_regions == 1530`, M3 report `n_active_regions` at 512
too, M4 record peak RSS). All four were implemented. Completed and accepted.

### M-03 — planner — E5 plan reviewed: approved with five modifications
Recorded in HANDOFF §7 E5. Summary: **M1** run E5a first — the plan skips it; **M2** the
memory analysis names the wrong scenario (worst case is `walnuts` at 432 train images /
24.3 GB, not `sheet_metal`, which has 137 images / 7.8 GB and is the cheapest); **M3**
resolved, the competing `blender` process exited on its own, but check free memory before
launching the 768 arm regardless; **M4** the sweep spans two code versions, so require
bit-identical parity on `extract_paths_prealloc` and re-run 448 under the new code if it is
not exact; **M5** state the 1024 arm's disposition rather than dropping it silently.

Confirmed independently and not disputed: density scaling is correct (cap/patches = 1.276
in all three arms), reusing `E4-evalside-512` as the 448 arm is legitimate, and the
preallocation fix is necessary rather than precautionary — 24.3 GB worst case, 48.7 GB with
`torch.cat` doubling, 54.0 GB with the processes that were then resident, against a 57.7 GiB
ceiling.

### M-04 — planner — directive D-03: E5a
See **Active directive**. E5a has not been run and E5 must not start until it reports.

