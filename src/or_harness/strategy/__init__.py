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
#: (what to do next time: expected quality/cost/failure risk,
#: provenance-grounded, rebuildable).
StrategicKnowledgeBank = StrategicBank

__all__ = [
    "ExperienceBank",
    "StrategicBank",
    "ExecutionEvidenceBank",
    "StrategicKnowledgeBank",
]
