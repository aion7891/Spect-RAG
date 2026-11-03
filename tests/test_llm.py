import json
import urllib.request

from spectrag.utils import questions
from spectrag.utils.llm import clean_reply, ollama_llm


def test_clean_reply_removes_thinking():
    assert clean_reply("<think>\nlet me see...\n</think>\n\nPorto ") == "Porto"
    assert clean_reply("Porto") == "Porto"


def test_old_import_still_works():
    assert questions.ollama_llm is ollama_llm


class FakeResponse:
    def read(self):
        return json.dumps({"response": "<think>hmm</think> Porto"}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_ollama_llm_cleans_the_reply(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    assert ollama_llm("qwen3:4b")("question") == "Porto"


def test_context_window_grows_with_the_prompt(monkeypatch):
    from spectrag.utils.llm import context_size
    from spectrag.utils.tokens import count_tokens
    assert context_size("short prompt") == 4096
    long_prompt = "word " * 12000                                     # about 12k tokens
    size = context_size(long_prompt)
    assert size >= count_tokens(long_prompt) * 1.15 + 1024 and size & (size - 1) == 0   # power of 2
    assert size // 2 < count_tokens(long_prompt) * 1.15 + 1024                          # smallest one
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ollama_llm("m")("word " * 12000)
    ollama_llm("m", num_ctx=8192)("hi")
    assert sent[0]["options"]["num_ctx"] == size and sent[1]["options"]["num_ctx"] == 8192


def test_retries_after_a_timeout_then_gives_a_clear_error(monkeypatch):
    import pytest
    calls = []

    def flaky(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("timed out")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    assert ollama_llm("m", retries=2)("hi") == "Porto" and len(calls) == 3

    def always_down(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", always_down)
    with pytest.raises(ConnectionError, match="failed after 2 tries"):
        ollama_llm("m", retries=1)("hi")
