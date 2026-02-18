import os
import sys
import click
import openai
from openai import OpenAI
from qdrant_client import QdrantClient

from kb.searcher import search, SNIPPET_LENGTH


@click.group()
def cli():
    """kb-search — search and inspect the knowledge base."""


@cli.command()
@click.argument("query")
@click.option("--top", default=5, show_default=True, help="Number of results to return")
@click.option("--collection", default=os.environ.get("KB_SEARCH_COLLECTION", "kb"), show_default=True, help="Qdrant collection name (env: KB_SEARCH_COLLECTION)")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL")
@click.option("--channel", default=None, help="Filter by channel slug (e.g. @SteveMagness)")
@click.option("--model", default="text-embedding-3-small", show_default=True, help="OpenAI embedding model")
def run(query, top, collection, qdrant_url, channel, model):
    """Search the knowledge base for QUERY."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise click.ClickException("OPENAI_API_KEY environment variable is required")

    # Normalize channel slug
    channel_slug = channel.removeprefix("@") if channel else None

    qdrant_client = QdrantClient(url=qdrant_url, timeout=30)
    openai_client = OpenAI(api_key=openai_api_key, timeout=30)

    click.echo(f'Hledám: "{query}"\n')

    try:
        results = search(
            query=query,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            collection=collection,
            top_k=top,
            channel_slug=channel_slug,
            model=model,
        )
    except Exception as e:
        raise click.ClickException(str(e))

    if not results:
        click.echo("Žádné výsledky.")
        return

    for i, result in enumerate(results, start=1):
        score = result["score"]
        payload = result["payload"]
        title = payload.get("title", "?")
        source = payload.get("transcript_source", "?")
        text = payload.get("text", "")
        url = payload.get("timestamp_url", "")

        # Truncate text for display
        snippet = text[:SNIPPET_LENGTH].strip()
        if len(text) > SNIPPET_LENGTH:
            snippet += "..."

        click.echo(f"#{i} [{score:.2f}] \"{title}\"  ({source})")
        click.echo(f"   {snippet}")
        click.echo(f"   {url}")
        click.echo()


@cli.command()
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL")
@click.option("--openai-api-key", default=lambda: os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key (env: OPENAI_API_KEY)")
def check(qdrant_url, openai_api_key):
    """Check connectivity to OpenAI and Qdrant."""
    failures = 0

    # 1. OPENAI_API_KEY is set and non-empty
    if openai_api_key:
        click.echo("OK  OPENAI_API_KEY is set")
    else:
        click.echo("FAIL  OPENAI_API_KEY is not set")
        failures += 1

    # 2. OpenAI API key is valid
    try:
        OpenAI(api_key=openai_api_key).models.list()
        click.echo("OK  OpenAI API key is valid")
    except Exception as e:
        click.echo(f"FAIL  OpenAI API key is invalid: {e}")
        failures += 1

    # 3. Qdrant is reachable
    try:
        QdrantClient(url=qdrant_url).get_collections()
        click.echo("OK  Qdrant is reachable")
    except Exception as e:
        click.echo(f"FAIL  Qdrant is not reachable: {e}")
        failures += 1

    click.echo()
    if failures == 0:
        click.echo("All checks passed.")
    else:
        click.echo(f"{failures} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
