import json
import os
import urllib.request

import pytest

from spectrafuse.utils.questions import (add_utility_questions, make_question_fn, ollama_llm,
                                      parse_questions)


def test_parse_questions_strips_numbers_and_bullets():
    reply = "Here you go:\n1. Who founded Aurora Labs?\n- When?\n* Where is it\n2) What is S1?"
    assert parse_questions(reply) == ["Who founded Aurora Labs?", "When?", "What is S1?"]


def test_make_question_fn_uses_prompt_and_limit():
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return "A?\nB?\nC?\nD?"

    question_fn = make_question_fn(fake_llm, count=2)
    assert question_fn("Aurora Labs makes sensors.") == ["A?", "B?"]
    assert "Aurora Labs makes sensors." in prompts[0]
    assert "2 short questions" in prompts[0]


def test_add_utility_questions_off_by_default():
    chunks = [{"text": "x"}]
    assert add_utility_questions(chunks, None)[0]["utility_questions"] == []


class FakeResponse:
    def read(self):
        return json.dumps({"response": "Who?"}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_ollama_llm_request(monkeypatch):
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ollama_llm("my-model")("hello") == "Who?"
    assert sent[0]["model"] == "my-model"
    assert sent[0]["options"]["temperature"] == 0
    assert sent[0]["stream"] is False


@pytest.mark.skipif(not os.environ.get("SPECTRAFUSE_LIVE_LLM"),
                    reason="set SPECTRAFUSE_LIVE_LLM=1 to call a real Ollama LLM")
def test_live_ollama_questions():
    model = os.environ.get("SPECTRAFUSE_LLM_MODEL", "gemma3:4b")
    question_fn = make_question_fn(ollama_llm(model), count=3)
    questions = question_fn("Aurora Labs was founded in 2014 in Lisbon by Maya Chen.")
    assert 1 <= len(questions) <= 3
    assert all(q.endswith("?") for q in questions)
