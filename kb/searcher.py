from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


def search(
    query: str,
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    collection: str,
    top_k: int,
    channel_slug: str | None = None,
) -> list[dict]:
    """
    Search for relevant chunks.

    1. Embeds query with text-embedding-3-small
    2. Searches Qdrant with optional channel_slug filter
    3. Returns list of {"score": float, "payload": dict}
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_vector = response.data[0].embedding

    query_filter = None
    if channel_slug is not None:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="channel_slug",
                    match=MatchValue(value=channel_slug),
                )
            ]
        )

    hits = qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )

    return [{"score": hit.score, "payload": hit.payload} for hit in hits]
