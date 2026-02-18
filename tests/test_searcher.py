import pytest
from unittest.mock import MagicMock, patch, call

from kb.searcher import search


# ── helpers ───────────────────────────────────────────────────────────────────

FAKE_EMBEDDING = [0.1] * 1536

SAMPLE_PAYLOAD = {
    "video_id": "abc123",
    "title": "Test Video",
    "channel_name": "Test Channel",
    "channel_slug": "testchannel",
    "upload_date": "20240101",
    "timestamp_url": "https://youtube.com/watch?v=abc123&t=0s",
    "transcript_source": "caption",
    "source_type": "youtube",
    "chunk_index": 0,
    "total_chunks": 5,
    "text": "Athletes who perform under pressure consistently focus on process.",
}


def make_openai_mock(embedding=None):
    if embedding is None:
        embedding = FAKE_EMBEDDING
    mock = MagicMock()
    mock.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=embedding)]
    )
    return mock


def make_qdrant_hit(score=0.87, payload=None):
    if payload is None:
        payload = SAMPLE_PAYLOAD
    hit = MagicMock()
    hit.score = score
    hit.payload = payload
    return hit


def make_qdrant_mock(hits=None):
    if hits is None:
        hits = [make_qdrant_hit()]
    mock = MagicMock()
    mock.search.return_value = hits
    return mock


# ── search ────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_list_of_results(self):
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        results = search("test query", qdrant, openai_client, "kb", top_k=5)
        assert isinstance(results, list)

    def test_result_contains_score_and_payload(self):
        qdrant = make_qdrant_mock([make_qdrant_hit(score=0.87)])
        openai_client = make_openai_mock()
        results = search("test query", qdrant, openai_client, "kb", top_k=5)
        assert len(results) == 1
        assert "score" in results[0]
        assert "payload" in results[0]
        assert results[0]["score"] == pytest.approx(0.87)

    def test_embeds_query_with_correct_model(self):
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        search("test query", qdrant, openai_client, "kb", top_k=5)
        openai_client.embeddings.create.assert_called_once()
        call_kwargs = openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["input"] == "test query"

    def test_calls_qdrant_search_with_top_k(self):
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        search("test query", qdrant, openai_client, "kb", top_k=7)
        qdrant.search.assert_called_once()
        call_kwargs = qdrant.search.call_args.kwargs
        assert call_kwargs.get("limit") == 7

    def test_calls_qdrant_search_with_collection(self):
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        search("test query", qdrant, openai_client, "my_collection", top_k=5)
        call_kwargs = qdrant.search.call_args.kwargs
        assert call_kwargs.get("collection_name") == "my_collection"

    def test_no_channel_filter_passes_no_filter(self):
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        search("query", qdrant, openai_client, "kb", top_k=5, channel_slug=None)
        call_kwargs = qdrant.search.call_args.kwargs
        assert call_kwargs.get("query_filter") is None

    def test_channel_filter_passed_to_qdrant(self):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        search("query", qdrant, openai_client, "kb", top_k=5, channel_slug="testchannel")
        call_kwargs = qdrant.search.call_args.kwargs
        query_filter = call_kwargs.get("query_filter")
        assert query_filter is not None
        # Filter should be a qdrant Filter object
        assert isinstance(query_filter, Filter)

    def test_channel_filter_targets_channel_slug_field(self):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        search("query", qdrant, openai_client, "kb", top_k=5, channel_slug="testchannel")
        call_kwargs = qdrant.search.call_args.kwargs
        query_filter = call_kwargs.get("query_filter")
        # Must filter on channel_slug field with correct value
        condition = query_filter.must[0]
        assert isinstance(condition, FieldCondition)
        assert condition.key == "channel_slug"
        assert condition.match.value == "testchannel"

    def test_multiple_results_returned(self):
        hits = [make_qdrant_hit(score=0.9), make_qdrant_hit(score=0.8), make_qdrant_hit(score=0.7)]
        qdrant = make_qdrant_mock(hits)
        openai_client = make_openai_mock()
        results = search("query", qdrant, openai_client, "kb", top_k=3)
        assert len(results) == 3
        assert results[0]["score"] == pytest.approx(0.9)

    def test_payload_fields_preserved(self):
        qdrant = make_qdrant_mock([make_qdrant_hit(payload=SAMPLE_PAYLOAD)])
        openai_client = make_openai_mock()
        results = search("query", qdrant, openai_client, "kb", top_k=5)
        payload = results[0]["payload"]
        assert payload["video_id"] == "abc123"
        assert payload["text"] == SAMPLE_PAYLOAD["text"]
        assert payload["timestamp_url"] == SAMPLE_PAYLOAD["timestamp_url"]

    def test_empty_results_returns_empty_list(self):
        qdrant = make_qdrant_mock(hits=[])
        openai_client = make_openai_mock()
        results = search("query", qdrant, openai_client, "kb", top_k=5)
        assert results == []
