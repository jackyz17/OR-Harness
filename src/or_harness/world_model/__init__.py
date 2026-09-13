"""World-model substrate (M1): state, actions, and real transitions.

This package is NOT a world model. It is the data foundation one needs
before a world model can be built: frozen pre-action belief snapshots, a
unified record of the seven action classes (including the ones the OUTER
agent performs), budget accounting, and the links between them.

Design rules inherited from the research consensus:

- Facts vs inference vs unknown: every key field carries an origin label
  and, where applicable, an evidence reference. Unknown is never written
  as a measured zero.
- Recommendation != selection != execution: a recall result is never
  auto-interpreted as a chosen plan; a chosen plan never fabricates
  execution quality.
- Real vs hypothetical: hypothetical branches (future M2 rollouts) are
  labelled ``source="hypothetical"`` and can never enter execution
  statistics or the knowledge-verification gate.
- The verification gate is absolute: nothing recorded here promotes a
  candidate into verified knowledge — that remains ``induce --verify``.
- No hidden calls, no background loops: everything is an explicit,
  stateless invocation against an explicit home, like the rest of the
  harness.
"""
