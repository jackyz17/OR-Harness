# Historical-execution retrieval: cosine threshold calibration on the local-hashing backend

Status: **local analysis note, not a commit** (`docs/` is gitignored).
Date: 2026-11-24. Repo working tree revision: `81f8bec`, plus unrelated uncommitted
work from other agents in the shared tree (this note was produced read-only against the
snapshot; the only file it wrote is this one).
DB snapshot: `harness_review_bench/or_harness.db`,
sha256 `065332f98e23975c5b88df643c2e3e1f57b4e6af08dd910691465be74bea16f5`.
Python 3.12.9, run with `PYTHONPATH=src`.

Scope: this note calibrates a cosine threshold **for the deterministic local backend
`LocalHashEmbeddingBackend` ("local-hashing-embedding-v1", 384-dim)** on the executions
actually present in this repo. No real embedding endpoint is configured in this
environment, so **nothing here is evidence about an OpenAI-compatible embedding model**.
Thresholds from different embedding backends are not comparable (see §9).

---

## 0. TL;DR — recommended values

| Key | Recommended value | Applies to | Confidence |
|---|---|---|---|
| `similarity_threshold` (overall document similarity) | **0.95** | `local-hashing-embedding-v1` only | in-sample A/C separation exists; corpus-specific, re-derive for any other backend |
| `method_similarity_threshold` (candidate-method-only query vs execution document) | **0.65** | `local-hashing-embedding-v1` only | no clean separation exists; keep + mandatory "low-confidence / lexical-overlap-only" labelling |

Both values are the **midpoint of the empirically defensible interval on this snapshot,
rounded to 2 decimals** (0.9483 → 0.95; 0.6546 → 0.65), which is why neither sits at an
extreme boundary. The intervals and their fragility are given in §5 and §6.

**Do not copy these numbers to a real embedding backend.** No such backend exists here;
the same experiment with the same class labels must be re-run first (§8).

---

## 1. Data and method

### 1.1 Corpus (tiny — say so up front)

| Quantity | Value |
|---|---|
| executions in `executions` | 12 (all `source="executed"`, none excluded/pending) |
| distinct `task_id` | 8 |
| task family labels | `linear_programming` 5 exec / 4 tasks; `resource_allocation` 4/2; `production_allocation` 2/1; `planning` 1/1 |
| tasks with >1 execution | 3 (`marketing_resource_allocation_20261008` ×3, `energy_investment_001` ×2, `marketing_campaign_20261008_v2` ×2) |
| rows that failed to decode / lost their task text | 0 / 0 |
| task-text versions available | 10; every used execution found its `(task_id, text_digest)` |

This is a review-bank snapshot of 8 two-variable linear-allocation tasks (7 marketing
budget/resource-allocation word problems plus 1 energy-investment problem), i.e. near-
template texts. That is
enough to *see* a distribution and to expose the failure modes, but **not** enough to
estimate a percentile reliably. A-class has n=5 pairs — its "p95" is an interpolation
over 5 numbers, not a population estimate. Treat every number below as a
description of this snapshot, not as a sampling result.

### 1.2 Documents and vectors (exactly the production builders)

For each `executions` row, in deterministic `execution_id` order:

```python
record = ExecutionRecord.from_dict(json.loads(row["payload"]))   # no store needed:
                                                                 # payloads are inline
text   = texts[(record.task_id, record.task_text_digest)]         # task_texts lookup,
                                                                 # keyed by version digest
doc    = document_execution(record, text)                         # index document
vector = LocalHashEmbeddingBackend().embed_query(doc)             # model_id
                                                                  # "local-hashing-embedding-v1"
score  = cosine(vector_a, vector_b)                              # the same cosine as recall
```

`document_execution` puts the task text first (dominant), then `strategy`, `family`,
`outcome`, `failures`, and the `planned`/`performed` method lines — so the document is
task text **plus** execution boilerplate; identical task text does not by itself give
score 1.0.

### 1.3 Pair classes (document ↔ document, the production recall shape)

| Class | Definition | n pairs available | source |
|---|---|---|---|
| **A** | same `task_id`, different `execution_id` | **5** | 3 tasks: `marketing_resource_allocation_20261008` (3 pairs), `energy_investment_001` (1), `marketing_campaign_20261008_v2` (1) |
| **B** | different `task_id`, same `family` label | **12** | `linear_programming` 9, `resource_allocation` 3; `production_allocation` and `planning` contribute 0 |
| **C** | different `task_id`, different `family` label | **49** | 21 cross-family task pairs × executions |
| total | | **66** = C(12,2) | |

At task granularity the whole experiment contains only **28 distinct task pairs** (8
choose 2): 7 same-family, 21 cross-family. Every A pair is "same task text" — in this
snapshot the same `task_id` always carries the same `text_digest`, so A is effectively
*identical task text + differing execution fields*, not "rephrased same task". That is
a limitation for interpreting A (§8).

Percentiles below use linear interpolation (the numpy/Excel `linear`/type-7 definition)
on the pair scores; state is explicit because p25–p95 on n=5 are nearly meaningless.

### 1.4 Method query (candidate-method-only)

For each execution the *candidate method* proxy is its recorded method, preferring the
plan (`method_planned`; 11 of 12) and falling back to the performed method
(`method_actual`; `ex_baaa99bcdf35` is the only actual-only record). The query text is
built with exactly the wording `document_execution` emits, so the experiment measures
the same lexical channel the index would see:

```
"{planned|performed} method: <name> — <step 1>; <step 2>; ..."
```

Method scores are then, per query: **(i) against its own document** ("self"),
**(ii) against documents of the other executions of the same task**, **(iii) against
documents of executions of other tasks** (split further by same/different family).

Note the built-in bias: this query string is a **literal substring of its own document**
(the doc includes the same method line), so "self" scores are an artificial upper bound.
The production query (a candidate method of a *new* task) is closer to the
cross-task column.

---

## 2. Document similarity: distribution per class

| Class | n | min | p25 | p50 | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| **A** same task | 5 | 0.9924 | 0.9966 | 0.9966 | 0.9972 | 0.9994 | 1.0000 | 0.9966 |
| **B** diff task, same family | 12 | 0.5494 | 0.5590 | 0.6442 | 0.7486 | 0.9707 | 0.9722 | 0.6895 |
| **C** diff task, diff family | 49 | 0.4773 | 0.5345 | 0.5900 | 0.6504 | 0.7640 | 0.9043 | 0.6019 |

**A vs C separate in this sample.** There is a real gap:

* `max(C) = 0.9043`  <  `min(A) = 0.9924`  (gap 0.0881).
* 0/49 C pairs reach `min(A)`; 5/5 A pairs exceed `max(C)`.
* Recommended interval: **(0.9043, 0.9924)**; midpoint 0.9483.

**B does not separate from C, and must not be used as a positive class.** B's mass
(0.5494–0.9722, p50 0.6442) is smeared across C's range (0.4773–0.9043, p50 0.5900).
The reason is visible in the data: B's two 0.97 outliers are the two **identical task
texts** carried under different `task_id`s (`marketing_campaign_20261008_v2` and
`..._v3`, 817 chars, byte-identical). B is "sometimes the same text, sometimes a
different template" — it is not a usable binary label.

### 2.1 Audit table — every A pair (5)

| score | execution A | execution B | note |
|---|---|---|---|
| 0.9924 | `ex_4afba484e51e` | `ex_f92538e0d400` | same task, error vs optimal, 1 vs 0 failures |
| 0.9966 | `ex_829b925ba0e6` | `ex_fc4d9e22486e` | same task, same strategy, error vs optimal |
| 0.9966 | `ex_f4b21b645261` | `ex_fc4d9e22486e` | same task, same strategy |
| 0.9972 | `ex_1fe734d7532c` | `ex_ede4dd28e26d` | same task, same strategy, optimal vs error |
| 1.0000 | `ex_829b925ba0e6` | `ex_f4b21b645261` | same task, same strategy, identical document |

### 2.2 Audit table — B pairs (12, all)

```
0.9722 ex_17b06e0f8965 ~ ex_1fe734d7532c   (v3 vs v2 — identical task text)
0.9694 ex_17b06e0f8965 ~ ex_ede4dd28e26d   (v3 vs v2 — identical task text)
0.8164 ex_a0d2552998f7 ~ ex_baaa99bcdf35
0.7260 ex_59a44b58cb24 ~ ex_fc4d9e22486e
0.7252 ex_59a44b58cb24 ~ ex_829b925ba0e6
0.7252 ex_59a44b58cb24 ~ ex_f4b21b645261
0.5633 ex_17b06e0f8965 ~ ex_a0d2552998f7
0.5617 ex_1fe734d7532c ~ ex_a0d2552998f7
0.5611 ex_a0d2552998f7 ~ ex_ede4dd28e26d
0.5527 ex_17b06e0f8965 ~ ex_baaa99bcdf35
0.5509 ex_1fe734d7532c ~ ex_baaa99bcdf35
0.5494 ex_baaa99bcdf35 ~ ex_ede4dd28e26d
```

### 2.3 Audit table — the 8 highest C pairs (the hard negatives)

```
0.9043 ex_134ddf044098 (planning)              ~ ex_a0d2552998f7 (linear_programming)
0.7940 ex_134ddf044098 (planning)              ~ ex_baaa99bcdf35 (linear_programming)
0.7650 ex_4afba484e51e (production_allocation) ~ ex_59a44b58cb24 (resource_allocation)
0.7625 ex_59a44b58cb24 (resource_allocation)   ~ ex_f92538e0d400 (production_allocation)
0.6927 ex_f92538e0d400 (production_allocation) ~ ex_fc4d9e22486e (resource_allocation)
0.6901 ex_4afba484e51e (production_allocation) ~ ex_829b925ba0e6 (resource_allocation)
0.6901 ex_4afba484e51e (production_allocation) ~ ex_f4b21b645261 (resource_allocation)
0.6893 ex_4afba484e51e (production_allocation) ~ ex_fc4d9e22486e (resource_allocation)
```

The top C pair (0.9043) is *lexically* two near-clone marketing budget tasks ("distribution
of budget between two advertising channels…", different constants), and *substantively*
the same mechanism (2-variable linear allocation) under two different `family` labels.
So the C upper tail is **label noise**: it is the ceiling of "different family", not the
ceiling of "unrelated". The true-unrelated ceiling is lower (C p95 = 0.7640), which makes
the 0.95 cut conservative in the right direction. C minimum is 0.4773.

---

## 3. Self vs cross-execution

### 3.1 Document query (= the execution's own document)

Querying with a document against the index obviously returns itself at **exactly 1.0000**.
The informative number is the **margin to the nearest other document**:

| nearest other document is… | count | margin `self - best_other` |
|---|---|---|
| same task (another attempt) | 7/12 | 0.0000–0.0076 |
| a **different task** | 5/12 | 0.0278–0.2350 |

Sorted margins (12): `0, 0, 0.0028, 0.0028, 0.0034, 0.0076, 0.0076, 0.0278, 0.0957,
0.0957, 0.1836, 0.2350`; p50 = **0.0076**.

**No — the self hit is not "clearly highest".** For 7 of 12 documents the nearest
neighbour is another execution whose score is within 0.008 of 1.0; in 2 cases
(`ex_829b925ba0e6`/`ex_f4b21b645261`) two executions produce an identical document and
the margin is **0.0000**. Near-duplicate executions are score-indistinguishable from
themselves. Consequences for the retrieval layer: *never* rely on score to drop self
hits (the existing `exclude_execution_ids` / self-id exclusion in `vector_recall` is the
correct mechanism, not the threshold), and do not expect top-1 to identify "the same
execution" — expect it to identify a near-duplicate cluster.

For 5 of 12 documents the nearest *non-self* document belongs to a **different task**
(the 0.9043 pair above is one of them), so high document similarity alone does not mean
"same task" either.

### 3.2 Method query

| comparison | n | min | p50 | max | mean |
|---|---|---|---|---|---|
| self (own doc) | 12 | 0.6750 | 0.7281 | 0.8694 | 0.7438 |
| other executions, **same task** | 10 | 0.6978 | 0.7259 | 0.7526 | 0.7262 |
| `self − best same-task other` | 7 | −0.0050 | −0.0001 | +0.0112 | **+0.0003** |
| `self − best other doc (any)` | 12 | −0.0607 | +0.0078 | +0.1125 | +0.0320 |

**Self is rank 1 for only 7/12 method queries.** It loses to another execution in 5/12,
by up to 0.0607; against same-task other executions it is statistically
indistinguishable (mean margin 0.0003). The reason is structural: same-task attempts
share the same planned method, so the candidate-method query describes several documents
equally well, and the query string is a substring of its own doc (inflating self).

This matters for the layer's semantics: with the hashing backend, a method query
**cannot identify "the execution this method came from"**; it identifies a
method/template neighbourhood. Method hits therefore need to be presented as *candidate
related history*, never as *the source of this method*.

---

## 4. Can one threshold separate A from C? (document similarity)

**Yes, in this snapshot** — with the fragility caveats below.

| class | min | max | comment |
|---|---|---|---|
| A | 0.9924 | 1.0000 | accept band |
| C | 0.4773 | 0.9043 | reject band |
| separation interval | **0.9043 < t < 0.9924** (exact: 0.904269 < t ≤ 0.992359) | | width 0.0881 |
| midpoint | **0.9483** | | rounded recommendation **0.95** |

Pass/fail at candidate thresholds (documents):

| t | A pass | B pass | C pass |
|---|---|---|---|
| 0.50 | 5/5 | 12/12 | 45/49 |
| 0.60 | 5/5 | 6/12 | 21/49 |
| 0.70 | 5/5 | 6/12 | 4/49 |
| 0.75 | 5/5 | 3/12 | 4/49 |
| 0.80 | 5/5 | 3/12 | 1/49 |
| 0.85 | 5/5 | 2/12 | 1/49 |
| 0.90 | 5/5 | 2/12 | 1/49 |
| **0.95** | **5/5** | **2/12** | **0/49** |

At 0.95: all 5 A pairs pass; 0 C pairs pass; the only 2 passing B pairs are the
byte-identical cross-task texts (which *should* pass — same content, different task id).

### Fragility of the separation (important)

1. **The gap is a same-text gap, not a semantic gap.** A pairs have *identical task
   text*. What separates A from C is "is the document text near-identical", which is all
   a lexical hash can see. A genuine cross-task analogy with different wording scores
   0.55–0.82 in this corpus (B and C bands) and will be rejected by 0.95.
2. **One label flip breaks the interval.** The identical-text pair
   `campaign_v2`/`campaign_v3` scores 0.9722 under the B label. If those two tasks had
   been given different `family` labels they would be C pairs **above 0.95** and would
   fail the cut. The interval is (0.9043, 0.9924) only because family labels happen to
   correlate with text templates here; with 8 tasks, a single relabel of a near-clone
   task pair can move `max(C)` by 0.07.
3. Therefore 0.95 is a **precision-first initial value** for this backend, not a
   theorem. It will effectively only return lexical near-duplicates; for cross-task
   discovery it returns little or nothing — which is the honest behaviour of a
   non-semantic backend (the module documentation already says this).

---

## 5. Threshold #1 — `similarity_threshold = 0.95`

Basis:

* A vs C separation interval **(0.9043, 0.9924)** on 5 vs 49 labelled pairs;
  recommended value = interval midpoint **0.9483 → 0.95**.
* At 0.95: A 5/5 accepted, C 0/49 accepted; nearest accepted non-A content is two
  byte-identical cross-task texts (B, 0.9722/0.9694).
* Margin to the hardest observed negative: 0.95 − 0.9043 = **+0.0457**;
  margin to the weakest observed positive: 0.9924 − 0.95 = **+0.0424**. Balanced, no
  boundary-picking.

Why not a lower value: at 0.90 one C pair (0.9043) passes; 0.80/0.85 behave the same;
at 0.70/0.75 four C pairs pass. Why not 0.99: it sits at the observed A minimum (an
extreme), and one slightly rephrased same-task text would be dropped.

Usage condition: this is an **initial, per-backend, per-corpus** value. Store it keyed by
`model_id` (the index already invalidates across `model_id`s and dimensions). Re-derive it
whenever the execution bank grows beyond template families, whenever task texts become
paraphrases rather than copies, and — obligatorily — for any non-hashing backend.

---

## 6. Threshold #2 — `method_similarity_threshold = 0.65`

### 6.1 Method-query distribution

| comparison | n | min | p25 | p50 | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| self (own doc) | 12 | 0.6750 | 0.7069 | 0.7281 | 0.7509 | 0.8526 | 0.8694 | 0.7438 |
| same-task other | 10 | 0.6978 | 0.7259 | 0.7259 | 0.7304 | 0.7512 | 0.7526 | 0.7262 |
| other-task, all | 122 | 0.2655 | 0.3635 | 0.4201 | 0.5322 | 0.6066 | 0.7642 | 0.4519 |
| ├ other-task, same family | 24 | 0.3395 | 0.4020 | 0.4137 | 0.5447 | 0.7398 | 0.7585 | 0.4788 |
| └ other-task, diff family | 98 | 0.2655 | 0.3589 | 0.4283 | 0.5215 | 0.5848 | 0.7642 | 0.4453 |

### 6.2 Separation check: **no clean separation**

`min(self) = 0.6750` but `max(other-task) = 0.7642` — the related band and the
cross-task top overlap. 2 "unrelated-family" pairs score above the weakest self hit.
Irrespective of the threshold, "self" and "another execution whose recorded method
matches" are not distinguishable by this score.

The overlapping pairs are, however, evidently *method-related* (manual inspection):

| score | query (method) | target document | why it is related |
|---|---|---|---|
| 0.7642 | `ex_134ddf044098` (`cp_sat_integer`) | `ex_a0d2552998f7` | target's method is `cp_sat_direct_integer` (same CP-SAT family), different family label |
| 0.7598 | `ex_a0d2552998f7` (`cp_sat_direct_integer`) | `ex_134ddf044098` | inverse of the pair above |
| 0.7585 | `ex_1fe734d7532c` (`MILP with Highspy direct formulation`) | `ex_17b06e0f8965` | v2/v3 have **identical task text**; same solver family |
| 0.7585 | `ex_ede4dd28e26d` (`MILP with Highspy direct formulation`) | `ex_17b06e0f8965` | idem |

So the upper tail of the cross-task distribution is **related content**, not noise. The
next-highest cross-task scores are a related cluster at 0.6335/0.6341 (same method name,
identical task text), then 0.6058/0.6067 (cp_sat-family analogies), then a continuous
tail down to 0.2655. There is no gap that says "related above, unrelated below".

### 6.3 Choosing a value from the self-anchored interval

Requirement: keep every self hit (the only "definitely related" set available), i.e.
`t ≤ min(self) = 0.6750`. Highest cross-task score that would still be rejected by a
threshold in that band is **0.6341**, so the defensible interval is:

> **(0.6341, 0.6750]**  → midpoint **0.6546 → recommend 0.65**

Sensitivity (method query, non-self pairs in the denominator):

| t | self pass | same-task pass | other-task pass | other-task **diff-family** pass |
|---|---|---|---|---|
| 0.60 | 12/12 | 10/10 | 8/122 | 3/98 |
| **0.65** | **12/12** | **10/10** | **4/122 (3.3%)** | **2/98 (2.0%)** |
| 0.70 | 9/12 | 8/10 | 4/122 | 2/98 |
| 0.75 | 3/12 | 1/10 | 4/122 | 2/98 |
| 0.80 | 2/12 | 0/10 | 0/122 | 0/98 |

Versus 0.70, the 0.65 cut keeps 3 more self hits and 2 more same-task hits while passing
no additional cross-task pair; versus 0.75 it is far better on both recall and precision.
A lower value (0.60) recovers 4 more cross-task pairs — all
4 of which inspect as method analogies (0.6067, 0.6058, 0.6335, 0.6341) — at the cost of a
4× higher pass rate on cross-task material; 0.65 is the conservative choice that still
retains the full related band.

**Mandatory labelling:** because the self/cross-task bands overlap at 0.63–0.76, a
method hit above 0.65 is evidence of **lexical/method-token overlap**, never proof that
the same method was run. Tag method hits with a low-confidence label (e.g.
`basis="method_text_overlap_only"`) and let the reader compare `method.name` / `steps`
structurally; do not let a method-similarity hit raise support counts, predict quality
or gate promotion (the existing design rule already forbids that for similarity in
general).

Also reverify the proxy: the calibration "self" band is a substring-inflated upper bound,
so a real candidate method for a *new* task can be expected to score **at or below** the
0.65–0.76 band. The data here cannot measure that production-shape recall (there are no
labelled candidate-method queries) — §8.

---

## 7. What the data cannot justify

The snapshot supports exactly two claims:

1. For local-hashing documents, `0.95` is above every observed cross-family pair and
   below every observed same-task pair.
2. For local-hashing method queries, `0.65` keeps every self and same-task hit while
   passing only 4/122 cross-task hits, all of which are visibly method-related.

It does **not** support:

* any statement about *semantic* relatedness — corpus has no "different wording, same
  mechanism" positive pairs (all A pairs are byte-identical task texts), so recall on
  semantic matches is **unmeasured**;
* a production threshold for a real embedding model (none configured; scores are not
  comparable across backends — §9);
* a percentile-level estimate of the tails — n(A)=5, n(B)=12; A's "p95" is an
  interpolation over 5 points;
* a claim that 0.95 generalises across corpora — a single near-clone task pair with a
  different family label would land at ~0.97 as class C and break the interval (§4).

---

## 8. What would verify a production value

Run the **same experiment, same class labels, same pair-class definitions** against the
real backend, on a larger bank (target: **30 tasks / 37 executions**, per the plan) with
`model_id` recorded next to every number:

1. Build documents with `document_execution` and embed with the OpenAI-compatible
   backend; keep task text + `task_text_digest` linking identical to production.
2. Recompute A/B/C distributions with the same percentile definition. Also add a
   fourth, essential class: **D = different task, different text, same mathematical
   mechanism** (labelled by hand, e.g. "2-variable linear allocation MILP"). Without D
   the experiment cannot measure semantic recall at all — the hashing run has no D.
3. Pick `similarity_threshold` as the midpoint of `(max(C), min(A))` only if that
   interval is non-empty; otherwise pick the operating point from a precision/recall
   curve on A ∪ D vs C and record the trade-off.
4. Recompute the method-query self/same-task/other distributions and derive
   `method_similarity_threshold` the same way; the hashing result (0.65) is **not**
   transferable — its scale is a lexical token-overlap scale.
5. Store both thresholds keyed by `model_id`; a threshold from one backend must be
   unusable/unreadable for another (the index's `model_id` check is the pattern).

Until that exists, the two values in §0 are local-hashing-only and must be presented as
such wherever they are configured.

---

## 9. Backend non-comparability (do not mix scales)

Cosine similarity is not a universal quantity: it depends on the embedding space,
normalisation, dimension and training objective. The `LocalHashEmbeddingBackend` is a
**signed-hash lexical bag-of-features** (tokens + 3-grams hashed into 384 dims) — its
scale is *token/n-gram overlap*, it has no semantics (two paraphrases with no shared
tokens score near 0), and its score distribution is compressed high because the
document builders share boilerplate. Read literally: a 0.95 on the hashing backend and
0.95 on `text-embedding-3-small` are unrelated events. Thresholds must therefore be
stored per backend (`model_id`), never shared, never defaulted across models; an index
built by one backend is already correctly invisible to another in
`embedding_index.py`, and the new retrieval layer must extend that isolation to its
threshold values.

---

## 10. Reproduction

The experiment is a read-only script over the SQLite snapshot; it used no writes, no
network, and touched only `docs/` (gitignored). Minimal core (all 12 rows, 66 pairs):

```python
import json, math, sqlite3, itertools
from or_harness.core.schema import ExecutionRecord
from or_harness.strategy.embedding_index import (
    LocalHashEmbeddingBackend, cosine, document_execution)

db = "harness_review_bench/or_harness.db"
con = sqlite3.connect(db); con.row_factory = sqlite3.Row
texts = {(r["task_id"], r["text_digest"]): r["text"]
         for r in con.execute("select task_id,text_digest,text from task_texts")}
recs, docs = [], []
for row in con.execute("select * from executions order by execution_id"):
    r = ExecutionRecord.from_dict(json.loads(row["payload"]))
    t = texts[(r.task_id, r.task_text_digest)]
    recs.append(r); docs.append(document_execution(r, t))
backend = LocalHashEmbeddingBackend()
vecs = [backend.embed_query(d) for d in docs]
def cls(a, b):
    if a.task_id == b.task_id: return "A"
    return "B" if a.profile_snapshot.family == b.profile_snapshot.family else "C"
buckets = {"A": [], "B": [], "C": []}
for i, j in itertools.combinations(range(len(recs)), 2):
    buckets[cls(recs[i], recs[j])].append(cosine(vecs[i], vecs[j]))
# percentile: linear interpolation (type 7), numpy default
def pct(v, q):
    v = sorted(v)
    if len(v) == 1: return v[0]
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (pos - lo)
for k in "ABC":
    s = buckets[k]
    print(k, len(s), "min", round(min(s), 4),
          "p95", round(pct(s, .95), 4), "max", round(max(s), 4))
# method query: method line exactly as document_execution writes it
```

Expected output: `A 5`, `B 12`, `C 49`; A range 0.9924–1.0000, C range
0.4773–0.9043; `min(A) > max(C)`.

Runtime: under 1 s. The full run additionally covered the method-query distributions
(§6) and the self/nearest-neighbour margins (§3).

---

## 11. One-line summary for the caller

Encode **`similarity_threshold = 0.95`** and **`method_similarity_threshold = 0.65`**,
keyed to `local-hashing-embedding-v1` only, and mark method-channel hits as
low-confidence lexical overlap. On this 12-execution/8-task snapshot, 0.95 separates the
5 same-task pairs (0.9924–1.0000) from all 49 cross-family pairs (≤0.9043), and 0.65
keeps every self/method hit (≥0.6750) while passing only 4/122 cross-task pairs, all four
of which are visibly method-related. Both values are corpus-specific and must be
re-derived before being used with a real embedding backend.

---

## 12. Implementation note (stage E/1)

This report measured DOCUMENT-to-DOCUMENT and METHOD-QUERY-to-DOCUMENT
similarities. The shipped retrieval uses **per-view query-to-document floors**
and, for the problem view, a **task-text-only vector** stored in the index
(`task_vector`), so a task-text query is compared with task text rather than
with a document whose length depends on the method it happens to carry.

Consequently the encoded defaults are NOT the doc-doc 0.95:

| view | encoded floor (local-hashing) |
|---|---|
| problem (task-text query vs `task_vector`) | 0.62 |
| candidate method | 0.60 |
| conditions | 0.30 |
| `similar_method` relevance | 0.62 |

The task-vector measurement on the same 12 executions / 8 tasks gives:
same-task 1.000, same-family 0.507-1.000, different-family 0.401-0.890 — so
0.62 is a DISCOVERY floor that keeps meaningful text overlap and is honestly
labelled `task_similar` when the candidate-method view does not also reach its
floor. An index built before the auxiliary vector existed falls back to the
full document (recorded as `problem_view_channel: legacy_full_document`); run
`orx rebuild-index --layer execution` to use the task-text channel.
