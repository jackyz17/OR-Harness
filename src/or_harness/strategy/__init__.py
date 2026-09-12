"""Two-layer memory: Execution Evidence (facts) and Strategic Knowledge
(derived commitments).

The internal class names stay ``ExperienceBank`` / ``StrategicBank`` —
storage tables and serialized payloads never change. The aliases below are
paper/user-facing terminology ("Execution Evidence Bank" / "Strategic
Knowledge Bank") for code that speaks that vocabulary.
"""

from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.strategic_bank import StrategicBank

#: "Execution Evidence Bank" — paper terminology for the fact layer
#: (what actually happened: actual strategy/quality/cost, failures,
#: artifacts).
ExecutionEvidenceBank = ExperienceBank
#: "Strategic Knowledge Bank" — paper terminology for the derived layer
#: (what to do next time: expected quality/cost/failure risk; revisable
#: beliefs whose validity does not depend on the survival of the original
#: evidence rows). TARGET semantics (next Induction migration round):
#: admission validation completes in offline induction BEFORE entries enter
#: the bank; online execution only records new evidence. CURRENT status:
#: entries are born candidate and promoted/demoted online by forward
#: quality checks; cost feedback never alters entry state.
StrategicKnowledgeBank = StrategicBank

__all__ = [
    "ExperienceBank",
    "StrategicBank",
    "ExecutionEvidenceBank",
    "StrategicKnowledgeBank",
]
