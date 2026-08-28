# Bank Lifecycle: Utility Attribution, Soft Delete, Cold Archive

How the experience bank evolves without ever editing or deleting content.
Read this when you want to understand why citations matter, how records rise
and sink in ranking, and what happens when you deprecate a record.

## The utility feedback loop

```
`orx recall` returns [E1],[E2] in priors.json (retrieval_count +1 for each)
    │
    ▼
you cite [uses E1] inside the [THINK] block (you declare which prior helped)
    │
    ▼
`orx validate` parses your citations
    (only injected En tags map to ids; you cannot invent one)
    │
    ▼
`orx episode` (gold_matched=true)
    │
    ▼
the framework credits utility_count +1 for every cited prior
    │
    ▼
soft-delete scoring: retrieval_count >= 5 AND utility/retrieval < 0.1
                     → retrieval score × 0.3
    │
    ▼
low-utility records sink in ranking;
harmful ones can be deprecated via `orx deprecate`
```

**Your only job in this loop**: cite priors you actually used with `[uses En]`
inside the `[THINK]` block. The framework handles counting, scoring, and ranking.

If you use a prior but don't cite it, it accrues **no utility** and may be
wrongly retired — uncited use silently breaks the feedback loop.

The `orx episode` output reports attribution:
```json
{"recorded": true, "episode_id": "ep_...", "utility_credited": 2, "cited_priors": 2, "status": "SOLVE_FLOW_COMPLETE"}
```

## Two counters, two sidecars

Mutable state NEVER lives in the append-only fact lines. It lives in sidecars
next to the store:

| Sidecar | Contents |
|---|---|
| `bank/lifecycle.json` | `{experience_id: {state, deprecated_at, reason}}` |
| `bank/utility_stats.json` | `{experience_id: {retrieval_count, utility_count}}` |
| `bank/induction_trigger_log.jsonl` | watermark + cluster fingerprints for induction cooldown |

| Counter | Meaning | Incremented when |
|---|---|---|
| `retrieval_count` | "seen" | a record is returned by `orx recall` / `orx query` / solve hints |
| `utility_count` | "helped" | gold match + the record was cited with `[uses En]` |

## Soft delete (ranking demotion, not deletion)

A record that has been seen enough but rarely helps sinks in ranking:

```
retrieval_count >= 5  AND  utility_count / retrieval_count < 0.1
→ retrieval score × 0.3
```

- The record **stays in the hot bank** (append-only red line).
- The `>= 5` guard is a grace window: new experiences have no evidence either
  way and must not be judged yet.
- Soft-deleted records can recover: if their utility ratio rises above the
  threshold, the penalty stops applying.

## Cold archive (deprecated records)

When you run `orx deprecate --id <experience_id> --reason <reason>`:

1. Lifecycle flips to `deprecated` in `bank/lifecycle.json`.
2. The record is compressed into a **provenance card** appended to
   `archive/deprecated.jsonl`:
   - dropped: bulky `retrieval_text`, full `method` body, `validation`/`scoring` detail
   - kept: `experience_id`, `content_hash`, `layer`, `polarity`, `title`,
     one-line `summary`, `structural_signature`, `source_episodes`,
     timestamps, deprecate reason, and the **embedding vector**
     (so approximate dedup still works)
3. The record is excluded from retrieval, induction clustering, and rebuilt indexes.

## Anti-resurrection (two-layer dedup)

When `orx append` or `orx append-pattern` submits a new record,
the framework checks it against the cold archive:

| Layer | Check | Rejects |
|---|---|---|
| 1. Exact | `content_hash` identical to an archive entry | verbatim resurrection |
| 2. Approximate | cosine similarity of embedding vector ≥ 0.8 | reworded resurrection |

Rejected appends return `status: "rejected_deprecated"`. A retired harmful
experience cannot re-enter the bank in either form.

## Why immutability (the red line)

Record content is never edited or physically deleted because:

- **Content-hash dedup** depends on stable hashes.
- **Episode provenance** (`produced_realization_ids`) points at record ids.
- **Induction lineage** (`derived_from_experience_ids`) links patterns to sources.

Corrections are always NEW records linked through `related_experience_ids`,
`derived_from_experience_ids`, or `contradicts_experience_ids` — never edits.
