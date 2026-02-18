import os
import click
from openai import OpenAI
from qdrant_client import QdrantClient

from kb.searcher import search


@click.command()
@click.argument("query")
@click.option("--top", default=5, show_default=True, help="Number of results to return")
@click.option("--collection", default="kb", show_default=True, help="Qdrant collection name")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL")
@click.option("--channel", default=None, help="Filter by channel slug (e.g. @SteveMagness)")
def cli(query, top, collection, qdrant_url, channel):
    """Search the knowledge base for QUERY."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise click.ClickException("OPENAI_API_KEY environment variable is required")

    # Normalize channel slug
    channel_slug = channel.lstrip("@") if channel else None

    qdrant_client = QdrantClient(url=qdrant_url)
    openai_client = OpenAI(api_key=openai_api_key)

    click.echo(f'Hledám: "{query}"\n')

    results = search(
        query=query,
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        collection=collection,
        top_k=top,
        channel_slug=channel_slug,
    )

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
        snippet = text[:200].strip()
        if len(text) > 200:
            snippet += "..."

        click.echo(f"#{i} [{score:.2f}] \"{title}\"  ({source})")
        click.echo(f"   {snippet}")
        click.echo(f"   {url}")
        click.echo()


if __name__ == "__main__":
    cli()
