import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from search import cli


# ── helpers ───────────────────────────────────────────────────────────────────

def invoke_check(*args, env=None, **kwargs):
    runner = CliRunner()
    return runner.invoke(cli, ["check", *args], env=env, catch_exceptions=False)


# ── OPENAI_API_KEY check ──────────────────────────────────────────────────────

class TestCheckOpenAIKey:
    def test_key_set_prints_ok(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-test"})
        assert "OK  OPENAI_API_KEY is set" in result.output

    def test_key_missing_prints_fail(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("auth error")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": ""})
        assert "FAIL  OPENAI_API_KEY is not set" in result.output

    def test_key_missing_increments_failures(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("auth error")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": ""})
        assert "check(s) failed" in result.output


# ── OpenAI validity check ─────────────────────────────────────────────────────

class TestCheckOpenAIValidity:
    def test_valid_key_prints_ok(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        assert "OK  OpenAI API key is valid" in result.output

    def test_invalid_key_prints_fail(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("Invalid API key")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-bad"})
        assert "FAIL  OpenAI API key is invalid" in result.output

    def test_invalid_key_includes_error_message(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("Invalid API key")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-bad"})
        assert "Invalid API key" in result.output


# ── Qdrant reachability check ─────────────────────────────────────────────────

class TestCheckQdrant:
    def test_reachable_prints_ok(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        assert "OK  Qdrant is reachable" in result.output

    def test_unreachable_prints_fail(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.side_effect = Exception("Connection refused")
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        assert "FAIL  Qdrant is not reachable" in result.output

    def test_unreachable_includes_error_message(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.side_effect = Exception("Connection refused")
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        assert "Connection refused" in result.output

    def test_custom_qdrant_url_is_used(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant_cls:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant_cls.return_value.get_collections.return_value = MagicMock()
            result = invoke_check("--qdrant-url", "http://myhost:9999",
                                  env={"OPENAI_API_KEY": "sk-valid"})
        mock_qdrant_cls.assert_called_once_with(url="http://myhost:9999")

    def test_default_qdrant_url_is_localhost(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant_cls:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant_cls.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        mock_qdrant_cls.assert_called_once_with(url="http://localhost:6333")


# ── Summary line ──────────────────────────────────────────────────────────────

class TestCheckSummary:
    def test_all_pass_prints_all_checks_passed(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        assert "All checks passed." in result.output

    def test_one_failure_prints_correct_count(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.side_effect = Exception("down")
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        assert "1 check(s) failed." in result.output

    def test_multiple_failures_prints_correct_count(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("bad key")
            mock_qdrant.return_value.get_collections.side_effect = Exception("down")
            result = invoke_check(env={"OPENAI_API_KEY": ""})
        assert "3 check(s) failed." in result.output

    def test_exit_code_zero_on_all_pass(self):
        with patch("search.OpenAI") as mock_openai, \
             patch("search.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-valid"})
        assert result.exit_code == 0
