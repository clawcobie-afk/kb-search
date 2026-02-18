# CLAUDE.md — kb-search

## Spuštění testů
```bash
source venv/bin/activate
pytest tests/ -v
```

## CLI
```bash
OPENAI_API_KEY=... python search.py "dotaz" \
  [--top 5] \
  [--collection kb] \
  [--qdrant-url http://localhost:6333] \
  [--channel lexfridman]
```

## Architektura
- `search.py` — Click CLI, formátování výstupu
- `kb/searcher.py` — embed dotaz, dotaz Qdrant, volitelný channel filter

## Klíčová funkce
```python
search(query, qdrant_client, openai_client, collection, top_k, channel_slug=None)
# → list[{"score": float, "payload": dict}]
```

## Env proměnné
- `OPENAI_API_KEY` — povinné

## Konvence
- TDD: testy jsou mockované (OpenAI + Qdrant)
- Channel filter se aplikuje přes Qdrant `must` filter na `channel_slug`
- `@` prefix v `--channel` se automaticky stripuje
