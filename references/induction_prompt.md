# Induction prompt (r18) — the reviewer's INDUCT template

This is the text the reviewer model is given when it reviews one task-episode's
induction material and decides whether any knowledge is worth recording. It is
the **prompt**, not the framework: it changes nothing about the CLI, the JSON
fields, the placeholders or the one-line result summary.

**Where it lives.** The generating entry is the `INDUCT` template in
`bench_orarla/mkmsgs_acc_patched.py` (runs on the experiment server under
`/home/ubuntu/.openclaw/workspace/bench_orarla/`). This file is the portable
source of truth for that template's wording; see "Entering the run" at the end.
`bench_orarla/_induct_new_block.py` is a helper the generator imports — edit the
generator's template, not a copy.

Placeholders (keep the names your generator already uses): `{material}` = the
`induction-material` batch, `{existing_knowledge}` = the current strategic
entries, `{task}` = the task identity.

---

## INDUCT template (new)

```
You are reviewing ONE finished task-episode to decide whether it yields reusable
knowledge. You are given {material} (this task's chain, the actual methods, outcomes,
task checks, failures, costs, and a bounded related history) and {existing_knowledge}
(the strategy library as it stands).

Knowledge is a MECHANISM that a LATER task can use: a condition under which an
operation works, the reason it works, the effect, and where it stops. "New" means the
library does not already cover that mechanism — not that it is new to operations
research. A textbook method can be worth recording, but "this is an ordinary LP/MILP
and a solver solved it" is not.

Read in this order, and keep the judgement inside the reading.

1. Read the episode.
   - What did the run ACTUALLY do (the recorded method / code), and what happened
     (status, task check, cost, failures)? A plan is intent, not performance: a
     `planned_only` method was not performed, and a derived-after-the-fact argument
     was not the method the run used.
   - Find the operative step: which modelling, transformation, bound, search or
     verification move is what made (or would make) the result correct? State the
     premise it needs.
   - A failure, a timeout, or a check that did not pass can be a CLUE about a
     boundary; it is not evidence that the method was applied correctly.

2. Abstract the numbers into conditions.
   - What property of THIS problem made the move work? Would it still hold if a key
     coefficient, bound or constraint changed? Can you write it as
     condition -> operation -> reason -> effect -> boundary?
   - A result that changes with the coefficients does not by itself make the method
     non-transferable; it usually means the applicability condition was not yet stated.

3. Compare against the library.
   - Compare the SPECIFIC premise, operation and effect, not the strategy name or the
     family label. Existing entries are claims to be inspected, not settled facts:
     a `verified` tag says a declared assertion held, not that the claim is correct or
     reaches far enough.
   - Two mechanisms are DIFFERENT when their premise or their operation differs. "The
     integer-domain choice" does not cover "the boundary selection", "the variable
     elimination", "the feasibility check" or "the optimality argument"; the
     translation of one constraint does not cover the translation of another.
   - If an existing entry really already covers the finding, say so and stop.

4. Decide what to record.
   - Keep these three apart and label which one you are claiming: a MATHEMATICAL
     argument (an inspectable derivation with its premises), an OBSERVED execution
     (what happened on the cited task), and a measured EMPIRICAL advantage (quality /
     cost / risk, only from comparable measurements). One task can support a
     conditional method when the argument and its premise are given; a claimed
     performance advantage still needs comparable measurements, and independent tasks
     are reported as a fact, not a threshold that all knowledge must pass.
   - Do not present a hypothesis or a plan as something that was executed, and do not
     attach a measured advantage to a run that did not measure it.
   - Keep engineering facts OUT of the knowledge: pure API names, status-string
     formats, package compatibility and debug-log fixes are not a reusable modelling
     mechanism. This does not forbid a FAILURE ANALYSIS that states a real mechanism
     and its boundary.

  A short example of the abstraction (illustrative, not a built-in rule and not to be
  applied everywhere): for continuous x,y>=0, minimising x+y subject to ax+by>=D with
  a>=b>0, we have ax+by <= a(x+y), so x+y >= D/a. If x=D/a, y=0 satisfies the OTHER
  constraints, it attains that lower bound and is therefore optimal. The reusable move
  is "derive a lower bound, then construct and check a feasible solution that attains
  it" — NOT "always give the resource to the highest-yield side". If another constraint
  forbids that candidate, the bound may still hold but this construction no longer
  proves optimality; an integer domain needs its own check.

5. Submit, or record an empty review.
   - If a distinct mechanism worth keeping survived steps 1-4, submit it as ONE claim
     (condition -> operation -> reason -> expected effect -> boundary) citing the real
     executions and roles. Put the reasoning in the claim, not in the check.
   - If nothing is worth adding, run the review with NO relation and put the reason in
     `--note`. That empty review is the correct, complete result — do NOT create a
     "no new knowledge" / "already covered" / "adopt existing entry" entry, and do not
     submit a claim whose only content is a placeholder or an empty method.

Report one line: what you recorded (or that you recorded nothing and why), and which
existing entry, if any, already covers the finding.
```

## Entering the run (how a change takes effect)

1. Edit the `INDUCT` template in `bench_orarla/mkmsgs_acc_patched.py` on the
   experiment server, replacing its wording with the block above (keep the
   generator's own placeholder names and the surrounding message-assembly code
   unchanged).
2. Confirm `_induct_new_block.py` is imported by the generator and does not carry
   a second, divergent copy of this text; if it does, align it to the generator.
3. The change applies to the NEXT run you explicitly start with the new template.
   Do not overwrite a running task's messages, historical logs or audit material.

Nothing else changes: same CLI, same JSON fields, same one-line summary, no new
schema, no new validator, no extra model call, no larger default recall.

## Reviewing the prompt offline

Use the fixed material in `harness_review_bench/` (the exported active bank) to
check the wording without touching a production database:

- the library there has three near-duplicate "direct exact solve, small linear
  allocation" entries that differ only in the constraint form, plus two
  engineering/coefficient-fix entries — a good test that step 3 separates
  mechanisms and excludes pure engineering fixes;
- the claim texts in `bank_strategic.json` show real "concrete numbers instead of
  conditions" phrasing, a good test for step 2.
