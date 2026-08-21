# Modeling Contract

## Contents

1. Output format (think + model)
2. GAMS-style DSL syntax (Option A)
3. Constraint label rules (critical)
4. Three verification layers
5. Repair loop
6. Pluggable syntax interface (Option B)

## 1. Output format (think + model)

The modeling stage produces exactly two blocks:

```
 <think> 
Free-text analysis: identify the objective, decision variables, constraints,
key structural insights, and which injected principles you applied.
Cite applied experiences with [uses E1] or [uses E1, E2].
<model>
...five GAMS-style blocks...
</model>
```

The framework parses these with a regex `<(think|model)>...</\1>`. Both tags are required. Missing either is an L1 format failure.

If planning priors were injected (Phase 4.1), the prompt will contain `[E1]...` past modeling experience references. You **must** cite any experience you actually apply using `[uses En]` inside ` <think> `. The framework parses these citations — only `En` tags that were actually injected map to real ids; you cannot invent a citation.

## 2. GAMS-style DSL syntax (Option A)

The `<model>` body uses a lightweight GAMS-style DSL. It borrows GAMS's five-block discipline (`SETS` / `PARAMETERS` / `VARIABLES` / `OBJECTIVE` / `CONSTRAINTS`) plus symbolic indexing `x[i,t]`, but carries no execution semantics and does not depend on a GAMS interpreter.

### Block headers

Each block starts with a header line — the block name followed by an optional colon:

```
SETS:
PARAMETERS:
VARIABLES:
OBJECTIVE:
CONSTRAINTS:
```

The parser matches headers case-insensitively via `^\s*(SETS|PARAMETERS|VARIABLES|OBJECTIVE|CONSTRAINTS)\s*:?\s*$`.

### SETS

Declare sets with inline members in braces:

```
SETS:
 a in Animals = {cow, sheep, chicken}
 t in Periods = {1, 2, 3, 4}
```

Set members are registered as declared symbols so that literal indices like `x[cow]` resolve correctly. Members are parsed from the brace content, split by comma, and stripped of quotes.

### PARAMETERS

Declare parameters with optional index brackets:

```
PARAMETERS:
 sell_price[a]
 feed_cost[a]
 manure_limit
 max_total
```

**Do NOT add inline `#` comments** on parameter lines. The symbol declaration regex `^\s*([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\]]*\])?)` matches the first identifier — a `#` comment after it is harmless for the match itself, but the block-splitter filters out lines starting with `#`. Keep comments in ` <think> ` instead.

### VARIABLES

Declare variables with type and bounds:

```
VARIABLES:
 x[a] integer >= 0
 y[i,j] binary
 f[i,j] continuous >= 0
```

The parser recognizes `binary`, `integer`, and `continuous` as variable types (via `\b(binary|integer|continuous)\b`). If no type is specified, it defaults to `continuous`.

### OBJECTIVE

One line stating the sense and the expression:

```
OBJECTIVE:
 maximize sum(a, (sell_price[a] - feed_cost[a]) * x[a])
```

```
OBJECTIVE:
 minimize sum(i, sum(j, c[i,j] * x[i,j]))
```

The keyword `sum(index, expression)` is recognized as summation notation and filtered from symbol cross-reference (it is not treated as an undeclared symbol).

### CONSTRAINTS

Each constraint is one line with a label, a colon, and the expression:

```
CONSTRAINTS:
 C1: sum(a, manure_rate[a] * x[a]) <= manure_limit
 C2: x[chicken] <= max_chickens
 C3: x[cow] >= min_cows
```

## 3. Constraint label rules (critical)

**Constraint labels MUST be `C1`, `C2`, `C3`, ...** — matching the regex `^C\d+$`.

This is the single most common L2 rejection. The structural validator treats any token in OBJECTIVE/CONSTRAINTS lines that is not a declared symbol, not a reserved word, not summation notation, and not a `C\d+` constraint label as an **undeclared symbol**.

| ❌ Wrong (rejected) | ✅ Correct (accepted) |
|---|---|
| `manure: sum(a, manure_rate[a] * x[a]) <= manure_limit` | `C1: sum(a, manure_rate[a] * x[a]) <= manure_limit` |
| `chicken_cap: x[chicken] <= max_chickens` | `C2: x[chicken] <= max_chickens` |
| `cow_min: x[cow] >= min_cows` | `C3: x[cow] >= min_cows` |

### Reserved tokens (not treated as symbols)

These tokens in OBJECTIVE/CONSTRAINTS are filtered out by the `_is_noise` check:

- Summation notation: `sum`, `sum_i`, `sum_{i,t}` (matched by `^sum_?\{?.*$`)
- Constraint labels: `C1`, `C2`, ... (matched by `^C\d+$`)
- Math keywords: `minimize`, `maximize`, `subject`, `to`, `sum`, `sigma`, `forall`, `in`, `s`, `t`, `st`, `and`, `or`, `e`, `pi`, `le`, `ge`, `eq`, `leq`, `geq`

### Single-letter index tokens

A single-letter token that appears inside index brackets `[...]` is treated as an index variable (like `i`, `j`, `t`), not as an undeclared symbol.

## 4. Three verification layers

All three layers run **without executing any code**. The model is verified before any solver branch is created.

### L1 — FormatValidator (deterministic)

Checks:
1. ` <think> ` tag present and non-empty.
2. `<model>` tag present and non-empty.
3. All five required blocks present and non-empty: `SETS`, `PARAMETERS`, `VARIABLES`, `OBJECTIVE`, `CONSTRAINTS`.

L1 runs synchronously and never calls an LLM.

### L2 — StructuralValidator (deterministic)

Checks:
1. **Symbol cross-reference**: every symbol referenced in OBJECTIVE/CONSTRAINTS must be declared in SETS/PARAMETERS/VARIABLES (or be a reserved token or single-letter index).
2. **Signature consistency**: if a signature was extracted, checks that binary variables in the model correspond to `binary_assignment` in the signature's `decision` field, and that multi-index variables correspond to `multi_index_2d`/`multi_index_3d`.

L2 runs synchronously and never calls an LLM.

### L3 — SemanticValidator (LLM-as-a-Judge, optional)

When an LLM client is available, the framework asks it to compare the PROBLEM and the MODEL and report:
1. Constraints required by the problem but missing from the model.
2. Spurious or duplicated constraints.
3. Known problem-family pitfalls (TSP subtour elimination, VRP capacity, etc.).

Output: a JSON array of `{"type": "...", "detail": "..."}` objects. An empty array `[]` means the model is faithful.

L3 is a no-op (returns passed) when no LLM client is injected.

## 5. Repair loop

The `StructuredModelingStage.run()` method runs a generate→verify loop:

```
for round in 1..max_rounds (default 3):
 prompt = build_modeling_prompt(problem, issues, planning_priors)
 raw = llm.generate_text(prompt)
 signature = extract_signature(raw) # may ask LLM
 report = gate.check(problem, raw, signature) # L1 + L2 + L3
 if report.passed:
 return success(think, model, signature, rounds_used=round)
 issues = report.issues # feed back into next prompt
return failure(issues, rounds_used=max_rounds)
```

On failure, the next prompt includes:
```
The previous model FAILED verification. Fix these issues:
- [structural] undefined_symbol: 'manure' is referenced but not declared
- [format] missing_block: block SETS is empty or absent
```

**No solver branch is created until the model passes all layers.** This is the `verify-before-code` principle (D17).

## 6. Pluggable syntax interface (Option B)

The DSL syntax is defined behind a `ModelSyntax` protocol with three methods:
- `split_blocks(model_text) -> ParsedModel`
- `declared_symbols(parsed) -> Dict[str, Dict]`
- `referenced_symbols(parsed) -> List[str]`

The current implementation is `GamsStyleSyntax` (Option A). A full GAMS grammar (Option B) can be swapped in by implementing the same interface, without changing the gate/flow above it. The `ModelingGate` accepts a `syntax` parameter for this purpose.
