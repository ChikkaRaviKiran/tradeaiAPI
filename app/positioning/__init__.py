"""Market Story (option positioning) feature.

Ported from the AgenticTrading project. Deliberately self-contained: it keeps
its own SQLite store, its own Dhan client and its own LLM guards so it can be
read, rewritten or deleted without touching the trading engine beside it.
"""
