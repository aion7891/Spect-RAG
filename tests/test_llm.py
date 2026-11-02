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
