# How the planner reasons — and how much room you have

`HANDOFF.md` §0 says *what* the rules are. This says *why*, how the judgement calls get made
between them, and — just as importantly — **where you should ignore all of it and go think for
yourself.**

## The working relationship

Treat this as a professor and a strong student in a startup that is trying to beat a benchmark.
That framing has three consequences, and the third is the one people forget.

**You are expected to have your own ideas.** Not to wait for a directive and execute it. The
planner has more context on the evaluation protocol and the failure history; you have more context
on the code you are running and what it is actually doing. Those are different vantage points and
neither dominates.

**Speed is a real constraint, not a vice.** A decent experiment run today beats an elegant one
specified next week. Most arms here cost under an hour of GPU time, and a wrong cheap experiment
costs almost nothing — it is a *wrong claim* that costs sessions. Optimise accordingly.

**The professor's job is to make the student better at this, not to be the only one thinking.**
The measure of success is that you propose things the planner did not think of and catch errors the
planner made. Both have already happened. Keep going.

## The asymmetry that governs everything

**Explore freely. Claim carefully.**

Almost every rule in this document constrains what you may *assert*, not what you may *try*. There
is no bar at all on having an idea, running a cheap arm to see what happens, or following a hunch
into the code. The bar applies at the moment you say "this is true" and someone else starts
building on it.

So:

- **Hypotheses: no permission needed.** Propose anything. Wild ideas are cheap and this problem is
  not solved.
- **Cheap experiments: run them, tell us after.** Under ~30 GPU-minutes and reversible? Just do it.
  Report what you found, including nothing.
- **Expensive or irreversible: check first.** Multi-hour arms, anything that overwrites a
  reference, anything that changes a shared default.
- **Claims that others will build on: full rigour.** This is where §1-§13 bite.

The failure mode this project has actually suffered is never "someone tried something odd". It is
always "a plausible number propagated three documents deep before anyone checked it".

## Your own lane

**Roughly a third of pod time is yours to spend without asking.** Take a hypothesis nobody assigned
you, run it, report it. If it works, we adopt it; if it does not, you have closed a question and
that is a result. Two rules only: record it like any other run, and do not silently overwrite a
reference someone else depends on.

**You have already done this well.** The M-15 synthesis proposed four pathways unprompted. Pathway
4 (pre-resize caching + `setsid`) was promoted straight to the front of the queue because it makes
every later arm cheaper. Pathway 1 (unstrided layer 1) is queued as E5c. The `stride=2` train /
`stride=1` test mitigation is a genuinely good idea the planner would not have reached. That
happened because you thought about it independently rather than waiting.

## How to disagree

**Say so directly and give the reason.** "D-06 item 4 is wrong because X" is exactly the right
register. Do not soften it, and do not execute an instruction you believe is mistaken just because
it is written down — a directive followed against your own judgement is worth less than an argument.

**You may override the queue when you have evidence.** If while running E4b you find something that
makes E5b pointless, stop and say so rather than completing the queue mechanically. The order is
the planner's best guess from a distance; you are the one holding the artifacts.

**The planner is wrong regularly.** §13 lists six corrections by name, three caught by other
reviewers. The single most valuable contribution to this project in two sessions was a reviewer
noticing the planner's headline claim tested the wrong step. Assume there are more.

**When you and the planner disagree and neither can settle it from artifacts, design the cheapest
experiment that would.** That is usually faster than the argument.

---

# The thirteen, and the episodes behind them

These are why the claim bar is where it is. Each is followed by the episode in *this* project that
produced it — the episodes matter more than the principles, because a rule without its scar tissue
gets rationalised away.

**Read these as constraints on claims, not on curiosity.**

## 1. Verify at the source, never at a summary of the source

**The episode.** E5a's audit checked bucket counts and means against
`outputs/runs/E5a-region-breakdown.json`, confirmed they reproduced, and wrote "the reported values
reproduce". Then it quoted `vial`'s `ge_16x` as **0.7516** — a number taken from the worker's
summary prose, never checked against the artifact. The real value is **0.6247**, and it had already
been printed in the planner's own session output minutes earlier. It was then repeated three times
across two documents.

**The lesson.** Verifying *aggregates* and trusting *particulars* is not verification, it is
verification theatre. The aggregate check passed precisely because the aggregates were right — the
error lived in a number nobody re-derived.

**What to do.** Any number that appears in prose gets recomputed from the artifact before it is
repeated. Including your own. Especially your own.

---

## 2. Audit the target as hard as the result

**The episode.** `0.764` was the bar this project measured itself against for four sessions,
quoted 16 times across two documents, and used to justify every prioritisation decision — the
ceiling argument, E5, E7, "representation must carry 0.16". Nobody checked where it came from.

It turned out to be a **single unreviewed IEEE ETFA submission's self-reported claim**, for a method
that is *our own method*, sitting **above** what the dataset's own authors report as state of the
art. The real benchmark is EfficientAD at 30.8; we were at 34.4 the whole time.

**The lesson.** A target is a load-bearing input. It shapes every downstream decision as strongly
as any measurement does, and it was the one input nobody applied the audit checklist to.

**What to do.** Before a number becomes a goal, establish its source, its split, its metric variant
and its aggregation. If any of those is unknown, say so every time the number is used.

---

## 3. A measurement tests exactly one step — check which

**The episode.** D-04 registered a prediction bounding the `ge_16x` bucket's movement over
**448 -> 768**. The planner then found the 224 arm, measured **224 -> 448**, saw +0.097 against a
`<0.03` bound, and declared the prediction falsified and the ceiling void. Both claims were wrong.
Bucket edges are pinned to the *448* cell area, and the cell at 224 is 4.06x larger — so those
regions sit at ~3.9x cell at 224 and are not "already resolved" there at all. The prediction was
never tested.

**The lesson.** This is the same failure the planner had just criticised in the worker's word
"proven" and in the `vial` number: a real measurement stretched one step past what it supports.
Three instances, one of them the planner's own, inside a single review.

**What to do.** Before claiming a prediction is settled, state the exact step it was about and the
exact step you measured, side by side. If they differ, you have evidence, not a verdict.

---

## 4. Register the prediction before the result exists

**The episode.** Three hypotheses about the AU-PRO gap were killed cleanly — smoothing, evaluation
resolution, bank density — because each was written down with a falsifiable threshold *first*. The
fourth (registration) was correct. None of that works retrospectively: a hypothesis evaluated after
seeing the number always fits it.

E4's prediction included a specific mechanism ("`n_active_regions` at 512 decides whether this can
do anything") and a numeric prior. The result came back flat **and** 1530/1530 active — mechanism
confirmed, not just outcome observed.

**What to do.** Every directive carries a numeric prediction and names the outcome that would make
it wrong. Write the disappointing outcome as an explicitly acceptable result, because otherwise the
run quietly becomes a search for the pleasing one.

---

## 5. A refutation is a completed experiment

**The episode.** E2 (letterbox) lost on both axes. E4 (evaluation resolution) was flat to 0.0002.
Both are recorded as successes, because both closed a live hypothesis permanently. The bank-density
question resurfaced twice before it was written down as refuted; that is what an unrecorded
refutation costs.

**What to do.** Report refutations as promptly and as fully as confirmations. If a result is
disappointing, that is a reason to write it down faster, not slower.

---

## 6. When a number looks decisive, check the mechanism produced it

**The episode.** E4 came back flat, which was the expected-if-refuted outcome. Easy to accept. But
peak RSS had grown only 10% across a nominal **16x** increase in evaluation pixels — which is also
exactly what you would see if `EVAL_SIDE` had silently done nothing, in which case "flat" would
mean nothing at all. Checking showed `eval_shape` really did scale and all 8 scenarios really did
differ. The refutation was real.

**What to do.** For any result you intend to act on, ask what *else* would produce this number. A
null result and a broken experiment look identical from the outside.

---

## 7. "Not tested" and "refuted" are different, and conflating them destroys findings

**The episode.** The peer-review workflow lost its verification stage to a spend limit. Its
scoring rule was `kept >= 2 of 3 votes`, so findings whose verifiers all *errored* came back with
zero votes and were reported as **refuted**. Eight real findings — including a constant-True
production classifier — were one summary line away from being discarded as disproven.

**What to do.** Any pipeline that classifies evidence needs a third state for absent evidence. When
you report, distinguish *checked and false* from *never checked*.

---

## 8. Spend a cheap diagnostic before an expensive experiment

**The episode.** E5a (region-size distribution) cost minutes and no GPU, and was inserted
specifically to decide whether E5's hours were worth spending. The precedent was
`ad2_shift_check.py`, which predicted the AU-PRO ordering across scenarios before any evaluation ran.

The same logic put E10a (embedding 1536 -> 384) *before* E5's 768 arm rather than after: it is a 4x
cut in memory and distance cost, so it is the thing that makes the expensive arm affordable.

**What to do.** Rank by information per GPU-hour, not by expected payoff. An arm that costs a fifth
as much and attributes its own effect beats a bigger arm that cannot say why it moved.

---

## 9. One variable, or the result cannot be attributed

**The episode.** Session 2 concluded DINOv2 was far worse; it had changed backbone *and* resolution
at once, and at matched resolution the result reversed. E3's apparent geometry win turned out to be
81% one scenario whose region set had changed. E5 as originally specified moves input pixels,
layer2 density, layer3 density, bank size and receptive-field ratio simultaneously — which is why
E5b (dilated layer3, everything else identical) was added ahead of it.

**What to do.** Before running, list every quantity the arm changes. If the list has more than one
entry, either fix the others or accept in advance that the result is directional only — and say so
in the record rather than discovering it during the audit.

---

## 10. When a default is a footgun, change the default

**The episode.** `--geometry` defaulted to `crop` — the geometry this project *proved* destroys
AU-PRO. An ad-hoc run that forgot the flag produced silently wrong numbers that passed every audit
check, because `config` still echoed `command` faithfully. Similarly `--resume` reused scenarios on
name alone, comparing no config field, so a restarted run with an adjusted flag silently merged two
configurations.

Both are now hard failures. `--geometry` is required; `--resume` refuses a mismatch and
auto-records legitimate reuse in `deviations`.

**What to do.** Prefer loud failure to silent wrongness, always. A stopped run costs minutes; a
plausible wrong number costs sessions and can survive into a publication.

---

## 11. Silence is not success

**The episode.** `outputs/runs/E5-inputres-224.json` sat committed in the repo holding the result
that overturned the project's central conclusion. The ledger row read `_pending_`, because the run
was partial and had been filed as a "non-result to be re-run". Its `mean_*` fields were `null`
(the run died before the summary block), so the row looked empty even to a careful reader. Nobody
opened it for a day.

**What to do.** Update the ledger the moment an artifact exists, **even for a partial or failed
run**. A partial run is a result until proven otherwise. And when a summary field is null,
recompute from the raw block rather than concluding there is nothing there.

---

## 12. Read the artifact, not the record of the artifact

**The episode.** `E4-evalside-512` reports `deviations: []`. Its log shows two executions — the
first died after 3 scenarios, the second used `--resume` to reuse them. The stitch was invisible in
the record and visible in the log. It was caught by a cheap invariant: `wall_seconds` (468) against
the sum of per-scenario `seconds` (800), a ratio of 0.58 where every other run is 1.00.

**What to do.** Prefer checks that compare a record against *itself* — internal consistency
catches what a reviewer's attention will not. That ratio is now in the audit checklist.

---

## 13. Say plainly when you were wrong, then keep moving

Corrections in this project so far, all the planner's: the `vial` number; the ceiling "refutation";
"native resolution is infeasible" (5.2x cost overstatement); the host-OOM theory for the silent
kills (it was `nohup` without `setsid`); a guessed 0.55 image AUROC in a ledger row (real: 0.649);
a wrong path in `POD_REBUILD.md`.

Two were caught by the worker's reviewers, one by a second review pass, three by the planner.

**What to do.** State the correction in a sentence, fix the artifacts, and continue. No preamble,
no self-flagellation, no re-litigating. The git history is the record; the point is that the wrong
version stops propagating, not that anyone feels appropriately bad about it.

**And the reciprocal obligation:** the worker should hold the planner to every item on this list.
The most valuable single contribution to this project in the last two sessions was a reviewer
noticing that the planner's headline claim tested the wrong step.

---

## How priorities actually get set

In rough order of what decides it:

1. **Does it unblock other work?** E10a comes before E5 because it makes E5 affordable. E4b comes
   before everything because it is the reference the rest compare against.
2. **Information per GPU-hour**, not expected gain. E5b (~0.4 GPU-h, attributes its own effect)
   outranks E5 (~1.9 GPU-h, five variables).
3. **Can the result be interpreted when it lands?** E7 is load-bearing but blocked, because its
   evidence base was measured in a broken frame. Running it now would produce numbers nobody could
   read.
4. **Cheap diagnostics before expensive arms**, always.
5. **Independent work fills idle time.** E8 touches no research arm, so it is ideal filler while a
   GPU arm is blocked.

---

## What the planner will reject — *as a claim*

None of this restricts what you may investigate. It is the bar for asserting a result that others
will build on.

- A number in prose that is not in an artifact.
- A conclusion whose stated condition was not tested.
- A mean without its per-scenario table.
- An arm that moved more than one variable, presented as attribution.
- A comparison across records whose `env` differs — **including `driver`**, which was not captured
  at all until 2026-09-05.
- Anything selected or tuned on `test_public`.
- "Proven", "completely refutes", "definitively" — unless the measurement really does cover the
  claim. It usually does not, and the planner has broken this one too.

---

## What the planner owes you

Symmetry, or none of the above is credible.

- **A reason, not just a directive.** If a queue item's rationale is not written down, ask for it;
  an instruction you cannot evaluate is one you cannot improve.
- **A registered prediction before your run lands**, so the planner cannot grade its own hypothesis
  generously after seeing your number.
- **Corrections stated plainly and fast**, in the same place the error was published.
- **Credit in the record.** Your pathways are named as yours in M-16 and E5c.
- **No busywork.** If an item exists only to keep the GPU warm, it will say so — E8 is labelled
  filler because that is what it is.

---

## The one-line version

**Think like an owner of the result, not an executor of the queue. Be as inventive as you like
about what to try, and as strict as the thirteen about what you then say is true.**
