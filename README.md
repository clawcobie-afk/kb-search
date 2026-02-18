# kb-search

Sémantické vyhledávání ve znalostní bázi. Embedduje dotaz přes OpenAI a vyhledává v Qdrant.

## Instalace

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Použití

```bash
export OPENAI_API_KEY=sk-...

python search.py "jak funguje dopamin"
python search.py "tréninková periodizace" --top 10 --channel @SteveMagness
```

### Flagy

| Flag | Výchozí | Popis |
|------|---------|-------|
| `QUERY` | — | Hledaný dotaz (poziční argument) |
| `--top` | `5` | Počet výsledků |
| `--collection` | `kb` | Název Qdrant kolekce |
| `--qdrant-url` | `http://localhost:6333` | URL Qdrant serveru |
| `--channel` | — | Filtrovat podle kanálu (slug nebo @handle) |

## Env proměnné

| Proměnná | Popis |
|----------|-------|
| `OPENAI_API_KEY` | OpenAI API klíč (povinné) |

## Výstupní formát

```
[1] 0.87 | Název videa — lexfridman
    "První dvě stě znaků chunku..."
    https://youtube.com/watch?v=abc123&t=120s
```

## Testy

```bash
pytest tests/ -v
```

## Pipeline

```
dotaz → OpenAI embedding → Qdrant search → ranked výsledky s timestamp URL
```

Předchází: [kb-indexer](../kb-indexer)
