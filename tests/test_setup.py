import os
import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from search import setup_cmd


# ── helpers ───────────────────────────────────────────────────────────────────

def invoke_setup(runner, api_key="sk-valid", qdrant_url="http://localhost:6333", collection="kb"):
    return runner.invoke(
        setup_cmd,
        ["--openai-api-key", api_key, "--qdrant-url", qdrant_url, "--collection", collection],
        catch_exceptions=False,
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestSetupCmd:
    def test_valid_inputs_writes_config(self, tmp_path):
        runner = CliRunner()
        env_file = tmp_path / ".env"

        with patch("search.openai.OpenAI") as mock_openai_cls, \
             patch("search.QdrantClient") as mock_qdrant_cls, \
             patch("search.os.path.expanduser", return_value=str(tmp_path)), \
             patch("search.os.makedirs"):

            mock_openai_cls.return_value.models.list.return_value = MagicMock()
            mock_qdrant_cls.return_value.get_collections.return_value = MagicMock()

            result = invoke_setup(runner, api_key="sk-test-key", qdrant_url="http://qdrant:6333", collection="mykb")

        assert result.exit_code == 0, result.output
        assert "Setup complete" in result.output
        assert env_file.exists()
        content = env_file.read_text()
        assert "OPENAI_API_KEY=sk-test-key" in content
        assert "QDRANT_URL=http://qdrant:6333" in content
        assert "KB_SEARCH_COLLECTION=mykb" in content

    def test_invalid_openai_key_exits_with_1(self, tmp_path):
        runner = CliRunner()

        with patch("search.openai.OpenAI") as mock_openai_cls, \
             patch("search.QdrantClient") as mock_qdrant_cls, \
             patch("search.os.path.expanduser", return_value=str(tmp_path)), \
             patch("search.os.makedirs"):

            mock_openai_cls.return_value.models.list.side_effect = Exception("Invalid API key")
            mock_qdrant_cls.return_value.get_collections.return_value = MagicMock()

            result = runner.invoke(
                setup_cmd,
                ["--openai-api-key", "sk-bad", "--qdrant-url", "http://localhost:6333", "--collection", "kb"],
            )

        assert result.exit_code == 1
        assert "Error" in result.output or "invalid" in result.output.lower()
        # .env must NOT be written on failure
        assert not (tmp_path / ".env").exists()

    def test_merges_with_existing_config(self, tmp_path):
        runner = CliRunner()
        env_file = tmp_path / ".env"
        # Pre-populate with an unrelated key
        env_file.write_text("SOME_OTHER_KEY=keep_me\nOPENAI_API_KEY=old-key\n")

        with patch("search.openai.OpenAI") as mock_openai_cls, \
             patch("search.QdrantClient") as mock_qdrant_cls, \
             patch("search.os.path.expanduser", return_value=str(tmp_path)), \
             patch("search.os.makedirs"):

            mock_openai_cls.return_value.models.list.return_value = MagicMock()
            mock_qdrant_cls.return_value.get_collections.return_value = MagicMock()

            result = invoke_setup(runner, api_key="sk-new-key", collection="updated")

        assert result.exit_code == 0, result.output
        content = env_file.read_text()
        # Unrelated key must be preserved
        assert "SOME_OTHER_KEY=keep_me" in content
        # OPENAI_API_KEY updated
        assert "OPENAI_API_KEY=sk-new-key" in content
        # Old value gone
        assert "old-key" not in content
        # New collection written
        assert "KB_SEARCH_COLLECTION=updated" in content
