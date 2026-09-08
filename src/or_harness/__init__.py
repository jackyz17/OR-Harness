"""or_harness: a harness-friendly OR strategy-learning capability layer.

OR-Harness runs *inside* an outer harness agent (Hermes-style). It is not an
autonomous agent: no conversation loop, no runtime LLM calls, no hidden global
state. The outer agent orchestrates; this layer advises, executes, and remembers.

Dependency direction (strictly one-way):
    api/cli -> profiling -> strategy -> execution -> adapters -> core
"""

__version__ = "3.0.0"

__all__ = ["__version__"]
