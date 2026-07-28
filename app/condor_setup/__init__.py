"""Pre-market confluence-based iron condor setup subsystem.

Fully self-contained: computes daily NIFTY/SENSEX confluence support &
resistance levels (classic + Camarilla pivots, prior-day H/L, opening range,
round numbers) and derives suggested iron-condor legs from them.

This subsystem is INFORMATIONAL ONLY. It never places live orders — it
computes a suggested setup each morning and exposes it via a read-only API
/ UI panel so the operator can manually place the trade. It does not touch
the orchestrator or any live execution path.
"""
