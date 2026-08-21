"""End-to-end latency instrumentation."""

from szp.latency.instrument import STAGES, Instrument, percentiles

__all__ = ["Instrument", "percentiles", "STAGES"]
